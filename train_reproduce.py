#!/usr/bin/env python3
"""Matched OPD/OPD2 reproduction for Qwen3-1.7B.

The implementation deliberately keeps the experiment contract in config.json so
the baseline and delta branches differ by one reviewed field.  It implements the
token-level policy-gradient objective directly because the paper's announced
TRL modifications were not public at the time of this reproduction.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(message: str, *, rank: int = 0, force: bool = False) -> None:
    if force or rank == 0:
        print(message, flush=True)


def distributed_setup() -> tuple[int, int, torch.device]:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


def barrier() -> None:
    dist.barrier()


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32 - 1))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def take_unique(stream: Iterable[dict[str, Any]], field: str, count: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in stream:
        value = str(row.get(field, "")).strip()
        if not value or value == "-" or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= count:
            break
    if len(result) < count:
        raise RuntimeError(f"Only found {len(result)} unique prompts in {field}; need {count}")
    return result


def prepare_prompt_cache(path: Path, seed: int) -> None:
    """Build the paper's 1:1:1 question-only mixture without redistributing it."""
    # The public code split has 15,703 embedded unique prompts.  Fifteen
    # thousand per domain is still larger than the 8,534/domain maximum consumed
    # by 100 x 256 examples, so every training question remains unique.
    per_domain = 15_000
    specs = [
        ("math", "nvidia/OpenMathReasoning", "default", "additional_problems", "problem"),
        ("science", "nvidia/OpenScienceReasoning-2", "default", "train", "input"),
        ("code", "nvidia/OpenCodeReasoning", "split_0", "split_0", "input"),
    ]
    prompts_by_domain: dict[str, list[str]] = {}
    for offset, (domain, repo, config, split, field) in enumerate(specs):
        stream = load_dataset(repo, config, split=split, streaming=True)
        stream = stream.select_columns([field])
        stream = stream.shuffle(seed=seed + offset, buffer_size=20_000)
        prompts = take_unique(stream, field, per_domain)
        prompts_by_domain[domain] = prompts
        print(f"DATASET domain={domain} prompts={len(prompts)} source={repo}/{config}/{split}", flush=True)
    rows = [
        {"domain": domain, "prompt": prompts_by_domain[domain][index]}
        for index in range(per_domain)
        for domain in ("math", "science", "code")
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def load_prompts(rank: int, seed: int) -> list[dict[str, str]]:
    path = Path("/tmp/opdd_prompt_mix.json")
    if rank == 0 and not path.exists():
        prepare_prompt_cache(path, seed)
    barrier()
    return json.loads(path.read_text(encoding="utf-8"))


def render_prompt(tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_ids(tokenizer: Any, question: str, max_prompt_tokens: int) -> torch.Tensor:
    rendered = render_prompt(tokenizer, question)
    ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    return ids


@torch.inference_mode()
def generate_completions(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    *,
    max_new_tokens: int | None = None,
    do_sample: bool = True,
    batch_size: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    model.eval()
    max_new = int(max_new_tokens or cfg["max_completion_tokens"])
    target_batch = int(batch_size or cfg["generation_batch_size"])
    outputs: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    index = 0
    while index < len(questions):
        current_batch = min(target_batch, len(questions) - index)
        while True:
            batch_questions = questions[index : index + current_batch]
            raw_ids = [prompt_ids(tokenizer, q, int(cfg["max_prompt_tokens"])) for q in batch_questions]
            width = max(x.numel() for x in raw_ids)
            padded = torch.full(
                (len(raw_ids), width), int(tokenizer.pad_token_id), dtype=torch.long, device=device
            )
            mask = torch.zeros_like(padded)
            for row, ids in enumerate(raw_ids):
                padded[row, -ids.numel() :] = ids.to(device)
                mask[row, -ids.numel() :] = 1
            torch.manual_seed(seed + index)
            try:
                generated = model.generate(
                    input_ids=padded,
                    attention_mask=mask,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=float(cfg["temperature"]) if do_sample else None,
                    top_p=1.0 if do_sample else None,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=int(tokenizer.eos_token_id),
                    use_cache=True,
                )
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
                target_batch = current_batch
                log(f"GENERATION_BACKOFF batch_size={current_batch}", force=True)
        for row, pids in enumerate(raw_ids):
            completion = generated[row, width:].detach().cpu()
            if tokenizer.eos_token_id in completion:
                eos_at = (completion == tokenizer.eos_token_id).nonzero(as_tuple=False)[0, 0].item()
                completion = completion[: eos_at + 1]
            completion = completion[completion != tokenizer.pad_token_id]
            if completion.numel() == 0:
                completion = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
            text = tokenizer.decode(completion, skip_special_tokens=True)
            outputs.append((pids.cpu(), completion, text))
        index += len(raw_ids)
    return outputs


def model_hidden(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return output.last_hidden_state


def token_objective(
    student: Any,
    teacher: Any,
    teacher_base: Any | None,
    pids: torch.Tensor,
    cids: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equations 6-8, evaluated in bounded vocabulary chunks."""
    full = torch.cat([pids, cids]).unsqueeze(0).to(device)
    attention = torch.ones_like(full)
    first_position = int(pids.numel()) - 1
    positions = torch.arange(first_position, full.shape[1] - 1, device=device)
    targets = full[0, first_position + 1 :]

    student_h = model_hidden(student, full, attention)[0]
    with torch.inference_mode():
        teacher_h = model_hidden(teacher, full, attention)[0]
        base_h = model_hidden(teacher_base, full, attention)[0] if teacher_base is not None else None

    chunk_size = int(cfg["logit_chunk_size"])
    top_k = int(cfg["top_k"])
    loss_sum = torch.zeros((), device=device)
    token_count = 0
    active_count = 0
    reward_sum = 0.0
    opd_adv_sum = 0.0

    for start in range(0, positions.numel(), chunk_size):
        pos = positions[start : start + chunk_size]
        tgt = targets[start : start + chunk_size]
        s_logits = student.lm_head(student_h[pos]).float()
        with torch.inference_mode():
            t_logits = teacher.lm_head(teacher_h[pos]).float()
            b_logits = teacher_base.lm_head(base_h[pos]).float() if teacher_base is not None else None

        s_logp = F.log_softmax(s_logits, dim=-1)
        with torch.inference_mode():
            t_logp = F.log_softmax(t_logits, dim=-1)
            b_logp = F.log_softmax(b_logits, dim=-1) if b_logits is not None else None
            top_values, top_ids = torch.topk(s_logits.detach(), k=top_k, dim=-1)
            top_probs = F.softmax(top_values, dim=-1)
            s_top = s_logp.detach().gather(1, top_ids)
            t_top = t_logp.gather(1, top_ids)
            s_taken = s_logp.detach().gather(1, tgt[:, None]).squeeze(1)
            t_taken = t_logp.gather(1, tgt[:, None]).squeeze(1)
            opd_reward = t_taken - s_taken
            opd_expected = (top_probs * (t_top - s_top)).sum(dim=-1)
            opd_adv = opd_reward - opd_expected

            if b_logp is None:
                advantage = opd_adv
                active = torch.ones_like(advantage, dtype=torch.bool)
            else:
                b_top = b_logp.gather(1, top_ids)
                b_taken = b_logp.gather(1, tgt[:, None]).squeeze(1)
                delta_reward = t_taken - b_taken
                delta_expected = (top_probs * (t_top - b_top)).sum(dim=-1)
                delta_adv = delta_reward - delta_expected
                active = delta_adv * opd_adv > 0
                advantage = torch.where(active, delta_adv, torch.zeros_like(delta_adv))

        sampled_student_logp = s_logp.gather(1, tgt[:, None]).squeeze(1)
        scaled_advantage = float(cfg["reward_scale"]) * advantage
        loss_sum = loss_sum - (scaled_advantage * sampled_student_logp).sum()
        token_count += int(tgt.numel())
        active_count += int(active.sum().item())
        reward_sum += float(advantage.sum().item())
        opd_adv_sum += float(opd_adv.sum().item())

        del s_logits, t_logits, s_logp, t_logp
        if b_logits is not None:
            del b_logits, b_logp

    loss = loss_sum / max(1, token_count)
    stats = {
        "tokens": float(token_count),
        "active_fraction": active_count / max(1, token_count),
        "mean_advantage": reward_sum / max(1, token_count),
        "mean_opd_advantage": opd_adv_sum / max(1, token_count),
    }
    return loss, stats


def sync_gradients(model: Any, world: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)


def cosine_lr(step: int, cfg: dict[str, Any]) -> float:
    total = int(cfg["steps"])
    warmup = max(1, round(total * float(cfg["warmup_ratio"])))
    minimum = float(cfg["minimum_lr_ratio"])
    if step < warmup:
        multiplier = (step + 1) / warmup
    else:
        progress = (step - warmup + 1) / max(1, total - warmup)
        multiplier = minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(cfg["learning_rate"]) * multiplier


def reduce_stats(stats: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = sorted(stats)
    values = torch.tensor([stats[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {k: float(v) for k, v in zip(keys, values.tolist())}


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return numbers[-1].strip() if numbers else text.strip().splitlines()[-1] if text.strip() else ""


def math_correct(prediction: str, answer: str) -> bool:
    try:
        from math_verify import parse, verify

        parsed_answer = parse(f"\\boxed{{{answer}}}")
        parsed_prediction = parse(prediction)
        if parsed_answer and parsed_prediction:
            return bool(verify(parsed_answer, parsed_prediction))
    except Exception:
        pass
    normalize = lambda x: re.sub(r"[\s$,]", "", x).lower()
    return normalize(extract_boxed(prediction)) == normalize(str(answer))


def prepare_eval_cache(path: Path) -> None:
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    aime = load_dataset("HuggingFaceH4/aime_2024", split="train")
    rows = {
        "math500": [
            {"problem": str(row["problem"]), "answer": str(row["answer"])} for row in math500
        ],
        "aime24": [
            {"problem": str(row["problem"]), "answer": str(row["answer"])} for row in aime
        ],
    }
    path.write_text(json.dumps(rows), encoding="utf-8")


def evaluate(
    model: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    device: torch.device,
    rank: int,
    world: int,
    label: str,
) -> dict[str, float]:
    path = Path("/tmp/opdd_eval.json")
    if rank == 0 and not path.exists():
        prepare_eval_cache(path)
    barrier()
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    samples: list[dict[str, str]] = []
    settings = [("math500", 1), ("aime24", int(cfg["aime_repetitions"]))]
    for benchmark, repetitions in settings:
        local_items: list[tuple[int, dict[str, str]]] = []
        for repetition in range(repetitions):
            for idx, row in enumerate(data[benchmark]):
                flat_idx = repetition * len(data[benchmark]) + idx
                if flat_idx % world == rank:
                    local_items.append((flat_idx, row))
        correct = 0
        total = 0
        eval_batch = int(cfg["eval_generation_batch_size"])
        for start in range(0, len(local_items), eval_batch):
            batch = local_items[start : start + eval_batch]
            generated = generate_completions(
                model,
                tokenizer,
                [row["problem"] for _, row in batch],
                cfg,
                device,
                int(cfg["seed"]) + 9_000_000 + start + rank,
                max_new_tokens=int(cfg["eval_max_completion_tokens"]),
                do_sample=True,
                batch_size=eval_batch,
            )
            for (_, row), (_, _, text) in zip(batch, generated):
                is_correct = math_correct(text, row["answer"])
                correct += int(is_correct)
                total += 1
                if len(samples) < 2:
                    samples.append(
                        {
                            "benchmark": benchmark,
                            "answer": row["answer"],
                            "prediction": extract_boxed(text)[:120],
                        }
                    )
        counts = torch.tensor([correct, total], device=device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        accuracy = 100.0 * counts[0].item() / max(1, counts[1].item())
        metrics[benchmark] = accuracy
        log(
            f"EVAL label={label} benchmark={benchmark} correct={counts[0].item()} "
            f"total={counts[1].item()} pass_at_1={accuracy:.3f}",
            rank=rank,
        )
    if rank == 0:
        log(f"EVAL_SAMPLES label={label} json={json.dumps(samples, sort_keys=True)}")
    barrier()
    return metrics


def load_model(model_id: str, device: torch.device, trainable: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rank, world, device = distributed_setup()
    seed_everything(int(cfg["seed"]), rank)
    if world != 8:
        raise RuntimeError(f"Paper setting requires 8 GPUs; torchrun started {world}")
    method = str(cfg["method"]).lower()
    if method not in {"opd", "opd2"}:
        raise ValueError(f"Unknown method: {method}")

    job_started = time.perf_counter()
    log("=== OPD DELTA REPRODUCTION CONFIG ===", rank=rank)
    log(json.dumps(cfg, sort_keys=True), rank=rank)
    log(
        "COMPUTE backend=kubernetes gpu_model=NVIDIA_RTX_PRO_6000_Blackwell "
        f"gpu_count={world} method={method}",
        rank=rank,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = load_prompts(rank, int(cfg["seed"]))

    log("MODEL_LOAD student", rank=rank)
    student = load_model(cfg["student_model"], device, trainable=True)
    log("MODEL_LOAD teacher", rank=rank)
    teacher = load_model(cfg["teacher_model"], device, trainable=False)
    teacher_base = None
    if method == "opd2":
        log("MODEL_LOAD teacher_base", rank=rank)
        teacher_base = load_model(cfg["teacher_base_model"], device, trainable=False)
    barrier()
    setup_seconds = time.perf_counter() - job_started

    pre_eval_started = time.perf_counter()
    original_metrics = evaluate(student, tokenizer, cfg, device, rank, world, "original")
    pre_eval_seconds = time.perf_counter() - pre_eval_started

    optimizer = AdamW(student.parameters(), lr=float(cfg["learning_rate"]), fused=True)
    local_batch = int(cfg["global_batch_size"]) // world
    if local_batch * world != int(cfg["global_batch_size"]):
        raise ValueError("global_batch_size must divide world size")
    train_started = time.perf_counter()
    consumed_domains = {"math": 0, "science": 0, "code": 0}
    trajectory: list[dict[str, float]] = []

    for step in range(int(cfg["steps"])):
        step_started = time.perf_counter()
        global_start = step * int(cfg["global_batch_size"])
        local_rows = [prompts[global_start + rank + world * j] for j in range(local_batch)]
        for row in local_rows:
            consumed_domains[row["domain"]] += 1
        torch.manual_seed(int(cfg["seed"]) + step * world + rank)
        rollouts = generate_completions(
            student,
            tokenizer,
            [row["prompt"] for row in local_rows],
            cfg,
            device,
            int(cfg["seed"]) + step * 10_000 + rank * 100,
        )
        student.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"tokens": 0.0, "active_fraction": 0.0, "mean_advantage": 0.0, "mean_opd_advantage": 0.0}
        completion_lengths: list[int] = []
        local_loss = 0.0
        for pids, cids, _ in rollouts:
            loss, stats = token_objective(student, teacher, teacher_base, pids, cids, cfg, device)
            (loss / local_batch).backward()
            local_loss += float(loss.detach().item()) / local_batch
            for key in sums:
                sums[key] += stats[key] / local_batch
            completion_lengths.append(int(cids.numel()))
        sync_gradients(student, world)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["gradient_clip"]))
        lr = cosine_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize(device)

        local_stats = {
            **sums,
            "loss": local_loss,
            "completion_tokens": float(sum(completion_lengths)),
            "mean_completion_length": float(np.mean(completion_lengths)),
            "max_completion_length": float(max(completion_lengths)),
            "grad_norm": float(grad_norm.item()),
            "step_seconds": time.perf_counter() - step_started,
        }
        stats = reduce_stats(local_stats, device)
        stats["step"] = float(step + 1)
        stats["lr"] = lr
        trajectory.append(stats)
        log(
            "TRAIN " + " ".join(
                [f"step={step + 1}", f"loss={stats['loss']:.6f}", f"adv={stats['mean_advantage']:.6f}",
                 f"active={stats['active_fraction']:.4f}", f"mean_len={stats['mean_completion_length']:.1f}",
                 f"max_len={stats['max_completion_length']:.0f}", f"grad_norm={stats['grad_norm']:.4f}",
                 f"lr={lr:.8g}", f"seconds={stats['step_seconds']:.2f}"]
            ),
            rank=rank,
        )
    train_seconds = time.perf_counter() - train_started

    post_eval_started = time.perf_counter()
    final_metrics = evaluate(student, tokenizer, cfg, device, rank, world, "final")
    post_eval_seconds = time.perf_counter() - post_eval_started
    total_seconds = time.perf_counter() - job_started
    domain_tensor = torch.tensor(
        [consumed_domains["math"], consumed_domains["science"], consumed_domains["code"]],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(domain_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        result = {
            "schema_version": 1,
            "method": method,
            "compute": {
                "backend": "kubernetes",
                "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
                "gpu_count": world,
            },
            "config": cfg,
            "data": {
                "consumed_prompts": int(cfg["steps"]) * int(cfg["global_batch_size"]),
                "domain_counts": dict(zip(["math", "science", "code"], domain_tensor.tolist())),
            },
            "metrics": {"original": original_metrics, "final": final_metrics},
            "delta_points": {k: final_metrics[k] - original_metrics[k] for k in final_metrics},
            "timing_seconds": {
                "setup": setup_seconds,
                "pre_eval": pre_eval_seconds,
                "training": train_seconds,
                "post_eval": post_eval_seconds,
                "total": total_seconds,
            },
            "trajectory": trajectory,
            "limitations": [
                "Direct token-level PyTorch implementation because the announced author repository was unavailable.",
                "Transformers SDPA rollout generation replaces the paper's TRL-integrated vLLM colocate backend.",
                "Evaluation uses one MATH-500 repetition and four AIME-2024 repetitions rather than the paper's broad 14-benchmark repeated suite.",
                "OpenMathReasoning prompts are sampled from its prompt-only additional_problems split.",
            ],
        }
        print("=== FINAL_RESULT_JSON ===", flush=True)
        print(json.dumps(result, sort_keys=True), flush=True)
        print("=== END_FINAL_RESULT_JSON ===", flush=True)
        print(
            f"RUNTIME method={method} backend=kubernetes gpu_model='NVIDIA RTX PRO 6000 Blackwell' "
            f"gpu_count={world} training_hours={train_seconds / 3600:.6f} total_hours={total_seconds / 3600:.6f}",
            flush=True,
        )
        print(
            f"CLAIM_RESULT method={method} math500={final_metrics['math500']:.3f} "
            f"aime24={final_metrics['aime24']:.3f} original_math500={original_metrics['math500']:.3f} "
            f"original_aime24={original_metrics['aime24']:.3f}",
            flush=True,
        )
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
