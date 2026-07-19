# On-Policy Delta Distillation: a matched Qwen3-1.7B reproduction

![Paper and observed OPD versus OPD2 benchmark scores](images/headline_benchmarks.png)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/rehaanahmad2013/on-policy-delta-distillation/blob/main/notebooks/opd2_reproduction.py)

The paper asks whether a student should imitate everything its teacher prefers, or only what the teacher *learned during reasoning post-training*. Its answer, On-Policy Delta Distillation (OPD2), scores each student-generated token by the log-probability difference between an instruction-tuned teacher and that teacher's base checkpoint. This reproduction tests the paper's Qwen3-1.7B non-thinking comparison against ordinary on-policy distillation (OPD).

**Assessment: partially reproduced.** Across two matched seeds, ordinary OPD reached **71.7% MATH-500 / 7.92% AIME 2024**, while OPD2 reached **72.0% / 9.17%**. The paper reports 81.0% / 36.7% for OPD and 83.9% / 41.0% for OPD2. OPD2 led in the two-seed mean, but by only +0.3 and +1.25 points, and the ordering reversed between seeds. That is directionally aligned but inconclusive under this bounded setup.

This was a Kubernetes reproduction on **NVIDIA RTX PRO 6000 Blackwell** GPUs, with **16 GPUs peak concurrent**. The full campaign occupied the cluster for **7.28 elapsed hours**; a matched 100-step pair completed in at most 3.33 hours. The public [tutorial notebook](../../notebooks/opd2_reproduction.py) embeds the evidence and opens directly in Molab without rerunning training.

## What was tested

The controlled variable was the token advantage. For a student rollout token \(y_t\), ordinary OPD uses the centered teacher-minus-student reward. OPD2 instead centers the teacher-minus-teacher-base reward and keeps it only when its sign agrees with the centered OPD advantage:

\[
R_{OPD}=\log p_T(y_t)-\log p_S(y_t),\qquad
R_{\Delta}=\log p_T(y_t)-\log p_{T_0}(y_t)
\]

\[
A_{OPD}=R_{OPD}-\mathbb E_{p_S}[R_{OPD}],\qquad
A_{OPD^2}=\begin{cases}A_\Delta & A_\Delta A_{OPD}>0\\0&\text{otherwise.}\end{cases}
\]

The expectation is approximated over the student's top 1,024 tokens, matching the paper. In the implementation, the consequential branch is deliberately small:

```python
delta_reward = teacher_taken - teacher_base_taken
delta_expected = (student_top_probs *
                  (teacher_top - teacher_base_top)).sum(-1)
delta_advantage = delta_reward - delta_expected
active = delta_advantage * opd_advantage > 0
advantage = torch.where(active, delta_advantage, 0.0)
```

The student is `Qwen/Qwen3-1.7B`; the teacher and teacher base are `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-4B-Base`. Every arm used 100 steps, one completion per question, temperature 0.7, AdamW at 5e-6 with cosine decay and 10% warmup, gradient clipping at 1.0, reward scale 0.1, and zero KL penalty. Manual eight-rank data parallelism kept student updates synchronized while each rank generated its own on-policy slice.

## Faithfulness and bounded substitutions

The announced author repository was not public, so this is a direct PyTorch/Transformers implementation of the paper's TRL objective. The paper's vLLM-colocated rollout backend was replaced with Transformers SDPA generation. A full-setting scout measured about 644 seconds per OPD step and 651 seconds per OPD2 step, projecting roughly 18 hours for 100 steps—past the hard compute deadline. The completed comparison therefore made two explicit reductions:

| Item | Paper | Reproduction | Consequence |
|---|---:|---:|---|
| Global batch per step | 256 | 64 | 6,400 rather than 25,600 consumed prompts |
| Maximum training completion | 8,192 | 4,096 tokens | The cap bound many later rollouts |
| Training steps | 100 | 100 | The full optimization horizon was retained |
| Training mixture | 1:1:1 math/science/code | 1:1:1 math/science/code | Public prompt-only splits; exact domain balance |
| Evaluation | 14 benchmarks, repeated | MATH-500 once; AIME 2024 four times | Focused evidence; AIME remains high variance |
| Hardware | 8× H100 | 8 GPUs/run, RTX PRO 6000 Blackwell | Throughput is not directly comparable |

Each seed began from the same checkpoint and used matched prompts, sampling seed, optimization, and evaluation in its OPD/OPD2 pair. Seed two changed only the shared seed. This is a faithful mechanism test, not a full-scale numerical replication.

## Benchmark evidence

![Change from the original checkpoint](images/within_run_gains.png)

| Evidence | Original | OPD | OPD2 | OPD2 − OPD | Assessment |
|---|---:|---:|---:|---:|---|
| Paper, MATH-500 | 68.6 | 81.0 | 83.9 | +2.9 | OPD2 advantage reported |
| Observed, MATH-500, two-seed mean | 69.9 | 71.7 | 72.0 | +0.3 | Directionally aligned; small and seed-sensitive |
| Paper, AIME 2024 | 14.2 | 36.7 | 41.0 | +4.3 | OPD2 advantage reported |
| Observed, AIME 2024, two-seed mean | 10.83 | 7.92 | 9.17 | +1.25 | Directionally aligned; high variance |

Seed one put OPD2 below OPD on MATH-500 (71.2 vs 71.4) and AIME (6.67 vs 7.50); seed two put OPD2 above OPD (72.8 vs 72.0 and 11.67 vs 8.33). Thus neither benchmark showed a seed-stable ordering. The observed mean gaps are 10% and 29% of the paper's reported MATH and AIME gaps, respectively. AIME used 120 sampled answers per checkpoint, so one correct answer changes the score by 0.83 points.

The before/after comparison is also informative. Relative to each matched original checkpoint, OPD gained 1.8 MATH points and lost 2.92 AIME points on average; OPD2 gained 2.1 MATH points and lost 1.67 AIME points. Because the two methods generate different rollouts as learning progresses, exact prompt and seed matching controls the starting conditions but cannot make their stochastic trajectories identical.

## What happened during training

![Completion length and OPD2 sign-agreement rate](images/training_dynamics.png)

Both objectives optimized stably for all 100 steps. OPD2's sign condition retained **72.91%** of token rewards on average, so teacher-base subtraction materially changed the gradient rather than behaving like ordinary OPD. The active fraction was stable across training. Mean completion length was 2,249 tokens for OPD and 2,489 for OPD2; at least one completion hit the 4,096-token cap in 61/66 OPD steps and 73/78 OPD2 steps across seeds one/two. That ceiling is the most consequential departure from the paper: it can truncate long reasoning and alter both the on-policy distribution and the dense token signal.

The training curves are diagnostics rather than benchmark evidence. Loss values can cross zero because this is an advantage-weighted policy objective; finite gradients, completed optimizer steps, matched checkpoint evaluations, and held-out accuracy are the relevant checks.

## Runtime claim

![Paper H100 and observed RTX PRO 6000 runtime](images/runtime_comparison.png)

The paper reports 4.4 hours for OPD and 5.5 hours for OPD2 on eight H100s, a 25% increase. Here, seed-level end-to-end means were **3.272 hours for OPD** and **3.322 hours for OPD2**, an overhead of **1.52%**. Training-only means were 3.050 and 3.093 hours (+1.43%). A separate paper-scale batch-256/8K one-step profile measured 644.17 seconds for OPD and 651.33 for OPD2 (+1.11%).

This supports the qualitative claim that the extra teacher-base forward pass still fits a short post-training window under the bounded recipe. It does not establish H100-equivalent throughput: GPU architecture, SDPA rather than vLLM, shorter rollouts, and smaller batches all differ. All reported times are actual harness wall times from Kubernetes logs, including model setup and both evaluations.

## Claim-by-claim assessment

| Target claim | Paper result | Observed result | Assessment | Compute |
|---|---|---|---|---|
| Teacher-minus-base delta rewards outperform ordinary OPD for Qwen3-1.7B reasoning | MATH-500 +2.9 pp and AIME24 +4.3 pp for OPD2 over OPD | Two-seed mean: +0.3 pp MATH, +1.25 pp AIME; seed-level signs split | **Inconclusive under this setup** | Two matched seeds; each run 8× RTX PRO 6000 Blackwell on Kubernetes |
| 100-step OPD2 remains a short post-training run despite an extra forward pass | 5.5 h OPD2 vs 4.4 h OPD on 8× H100 | 3.322 h mean total, 1.52% over matched OPD | **Aligned** | Peak 16 GPUs concurrent; 7.28 h campaign wall time |

The performance claim requires the paper-scale 25,600 rollouts at 8K, the authors' exact modified TRL/vLLM stack, and the broader repeated benchmark suite for a stronger test. The present run should not be used to conclude that the paper's claim is generally incorrect; it says only that two reduced-scale matched seeds produced a smaller, seed-sensitive mean advantage.

## Provenance

The exact command on every experiment node was `bash run.sh`; hyperparameters live in committed `config.json` rather than command-line overrides.

- [Full-setting OPD scout](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd-balanced-consumed-prompts) and [OPD2 scout](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd2-balanced-consumed-prompts): established that the 8K/batch-256 setting would miss the deadline.
- [OPD seed one](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd-64-batch-4k-completion) and [OPD2 seed one](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd2-64-batch-4k-completion): first complete matched pair.
- [OPD seed two](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd-replicate-seed) and [OPD2 seed two](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd2-replicate-seed): robustness pair.
- [Paper 2607.15161](https://arxiv.org/abs/2607.15161), [public code](https://github.com/rehaanahmad2013/on-policy-delta-distillation), and [self-contained Molab notebook](https://molab.marimo.io/github/rehaanahmad2013/on-policy-delta-distillation/blob/main/notebooks/opd2_reproduction.py).

Kubernetes was used for every experiment. The GPU model was NVIDIA RTX PRO 6000 Blackwell, the peak concurrent allocation was 16 GPUs, and actual elapsed wall time was 7.28 hours (2026-07-19 09:13:54–16:30:42 UTC).
