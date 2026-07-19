# On-Policy Delta Distillation — Qwen3-1.7B reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/rehaanahmad2013/on-policy-delta-distillation/blob/main/notebooks/opd2_reproduction.py)

This public repository reproduces the central claim of [*On-Policy Delta Distillation* (arXiv:2607.15161)](https://arxiv.org/abs/2607.15161): centered teacher-minus-base rewards (OPD2) should outperform ordinary on-policy distillation (OPD) for Qwen3-1.7B reasoning post-training, while retaining a short 100-step runtime.

The first matched seed did **not show the reported quality ordering under the bounded setup**. The paper reports OPD2 vs OPD at 83.9 vs 81.0 on MATH-500 (+2.9 points) and 41.0 vs 36.7 on AIME24 (+4.3). We observed 71.2 vs 71.4 (−0.2) and 6.67 vs 7.5 (−0.83). A second matched seed is running because these observed differences are small. The runtime claim aligned: OPD2 completed in 3.327 hours end to end versus 3.263 hours for OPD, only 1.96% slower.

We kept the paper's student and teachers, 100 steps, optimizer, sampling temperature, top-1,024 centering, reward scale, prompt mixture, and matched prompt order. To fit the hard wall-time window, global batch was reduced from 256 to 64 and maximum completion length from 8,192 to 4,096. The unavailable author repository also required a direct PyTorch/Transformers implementation rather than the paper's TRL-integrated vLLM path. Evaluation was narrowed to MATH-500 and four AIME24 repetitions.

All formal runs used the OpenResearch Kubernetes backend on NVIDIA RTX PRO 6000 Blackwell GPUs: 8 GPUs per method and 16 GPUs at peak concurrency. See the [illustrated report](reports/opd2-qwen17b/report.md), [self-contained marimo tutorial](notebooks/opd2_reproduction.py), and [structured result data](results/reproduction_summary.json).

## Experiment log

Commands below are copied verbatim from `orx exp status`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public README, report, notebook, metadata | Not run as an experiment (publication surface) | Presentation only | No experiment compute |
| [`orx/opd-balanced-consumed-prompts`](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd-balanced-consumed-prompts) | Full paper batch-256 / 8K OPD profile | `bash run.sh` | Step 1: 644.17 s; original MATH 69.8, AIME 10.83; intentionally stopped after useful profile | Kubernetes, 8× RTX PRO 6000, 20m13s |
| [`orx/opd2-balanced-consumed-prompts`](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd2-balanced-consumed-prompts) | Full paper batch-256 / 8K OPD2 profile | `bash run.sh` | Step 1: 651.33 s (+1.11%); gate 73.74%; intentionally stopped | Kubernetes, 8× RTX PRO 6000, 20m12s |
| [`orx/opd-64-batch-4k-completion`](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd-64-batch-4k-completion) | Seed 260715161 matched OPD control | `bash run.sh` | MATH 71.4; AIME 7.5; 3.263 h total | Kubernetes, 8× RTX PRO 6000, 3h16m |
| [`orx/opd2-64-batch-4k-completion`](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd2-64-batch-4k-completion) | Seed 260715161; one-field OPD2 treatment | `bash run.sh` | MATH 71.2; AIME 6.67; 3.327 h total; quality direction not observed | Kubernetes, 8× RTX PRO 6000, 3h20m |
| [`orx/opd-replicate-seed`](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd-replicate-seed) | Seed 260715162 OPD robustness control | `bash run.sh` | Running | Kubernetes, 8× RTX PRO 6000 |
| [`orx/opd2-replicate-seed`](https://github.com/rehaanahmad2013/on-policy-delta-distillation/tree/orx/opd2-replicate-seed) | Seed 260715162 OPD2 robustness treatment | `bash run.sh` | Running | Kubernetes, 8× RTX PRO 6000 |

Early failed launch and public-dataset discovery branches are retained in OpenResearch provenance, but omitted here because they do not add scientific evidence beyond explaining setup lineage.

## Run locally or on the configured cluster

The formal experiment command is:

```bash
bash run.sh
```

`config.json` selects OPD2 on `main`; the linked experiment branches freeze every matched variant. The Kubernetes manifest is in `.orx/k8s.yaml`. Training was launched only through `orx exp run --backend k8s`.

To explore the already-produced evidence without training:

```bash
uvx marimo edit notebooks/opd2_reproduction.py
uvx marimo run notebooks/opd2_reproduction.py
```

The notebook embeds the small public result table and figures, so it opens directly in Molab without fetching local experiment artifacts.
