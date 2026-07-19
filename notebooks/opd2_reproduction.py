# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "matplotlib>=3.9.0",
#   "numpy>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # On-policy delta distillation, reproduced

    **Already-computed evidence from two matched seed pairs of 100-step Qwen3-1.7B runs.**
    This notebook explains the central claim of [*On-Policy Delta Distillation*](https://arxiv.org/abs/2607.15161)
    without asking you to rerun an expensive experiment. The formal runs used Kubernetes,
    8 NVIDIA RTX PRO 6000 Blackwell GPUs per method, and 16 GPUs at peak concurrency.

    OPD2 led in the two-seed mean, but the ordering reversed between seeds and the gap was
    much smaller than the paper's. The extra base-model forward pass consistently preserved
    a short post-training window. The Kubernetes campaign used 7.28 elapsed hours.
    """)
    return


@app.cell
def _(mo, np, plt):
    benchmark_names = ["MATH-500", "AIME 2024"]
    paper_opd = [81.0, 36.7]
    paper_opd2 = [83.9, 41.0]
    observed_opd = [71.7, 7.9166666667]
    observed_opd2 = [72.0, 9.1666666667]

    headline_fig, headline_ax = plt.subplots(figsize=(9.2, 4.6))
    headline_x = np.arange(2)
    headline_width = 0.18
    headline_series = [
        ("Paper OPD", paper_opd, "#5065A8", "//"),
        ("Paper OPD2", paper_opd2, "#E07A5F", "//"),
        ("Observed OPD", observed_opd, "#5065A8", None),
        ("Observed OPD2", observed_opd2, "#E07A5F", None),
    ]
    for headline_i, (headline_label, headline_values, headline_color, headline_hatch) in enumerate(headline_series):
        headline_bars = headline_ax.bar(
            headline_x + (headline_i - 1.5) * headline_width,
            headline_values,
            headline_width,
            label=headline_label,
            color=headline_color,
            hatch=headline_hatch,
        )
        headline_ax.bar_label(headline_bars, fmt="%.1f", padding=2)
    headline_ax.set_xticks(headline_x, benchmark_names)
    headline_ax.set_ylim(0, 100)
    headline_ax.set_ylabel("Pass@1 (%)")
    headline_ax.set_title("Two-seed mean: OPD2 led modestly, with seed-sensitive ordering")
    headline_ax.legend(frameon=False, ncol=2)
    headline_ax.grid(axis="y", alpha=0.18)
    headline_fig.tight_layout()

    mo.vstack(
        [
            mo.as_html(headline_fig),
            mo.callout(
                mo.md(
                    "**Observed comparison:** OPD2 − OPD averaged **+0.3 percentage points** "
                    "on MATH-500 and **+1.25 points** on AIME24. Seed one favored OPD "
                    "(−0.2/−0.83); seed two favored OPD2 (+0.8/+3.33), so the quality "
                    "claim remains inconclusive under this reduced recipe."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The idea in one equation

    Ordinary OPD rewards a sampled token when the teacher likes it more than the teacher's
    centered top-k alternative. OPD2 first subtracts the base model:

        \[
        A_{\Delta}(y_t) =
        \big(\log p_T(y_t)-\log p_B(y_t)\big)
        - \sum_{v \in \operatorname{topk}(p_\theta)} \bar p_\theta(v)
        \big(\log p_T(v)-\log p_B(v)\big).
        \]

    A joint sign gate keeps this delta reward only where it agrees with the ordinary OPD
    advantage. Across the two seeds, that gate retained **72.91% of tokens on average**, so
    the mechanism was active and stable rather than degenerating into ordinary OPD.

    ```python
        top_ids = topk(student_logits, 1024)
        top_probs = softmax(student_logits[top_ids])
        opd_reward = teacher_logp[token] - student_logp[token]
        opd_center = sum(top_probs * (teacher_logp[top_ids] - student_logp[top_ids]))
        opd_adv = opd_reward - opd_center
        delta = teacher_logp - base_logp
        delta_adv = delta[token] - sum(top_probs * delta[top_ids])
        active = sign(delta_adv) == sign(opd_adv)
        advantage = 0.1 * delta_adv * active
    ```
    """)
    return


@app.cell
def _(mo, np, plt):
    runtime_methods = ["OPD", "OPD2"]
    paper_runtime = [4.4, 5.5]
    observed_runtime = [3.272469, 3.322339]
    runtime_x = np.arange(2)
    runtime_fig, runtime_ax = plt.subplots(figsize=(8.6, 4.3))
    runtime_paper_bars = runtime_ax.bar(
        runtime_x - 0.18, paper_runtime, 0.36, label="Paper: 8× H100", color="#8D99AE", hatch="//"
    )
    runtime_observed_bars = runtime_ax.bar(
        runtime_x + 0.18,
        observed_runtime,
        0.36,
        label="Observed: 8× RTX PRO 6000",
        color="#2A9D8F",
    )
    runtime_ax.bar_label(runtime_paper_bars, fmt="%.2f h", padding=3)
    runtime_ax.bar_label(runtime_observed_bars, fmt="%.2f h", padding=3)
    runtime_ax.set_xticks(runtime_x, runtime_methods)
    runtime_ax.set_ylabel("End-to-end hours")
    runtime_ax.set_title("Both 100-step runs fit a short post-training window")
    runtime_ax.legend(frameon=False)
    runtime_ax.grid(axis="y", alpha=0.18)
    runtime_fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Runtime claim"),
            mo.as_html(runtime_fig),
            mo.md(
                "Across seeds, OPD2 averaged **3.093 training hours / 3.322 total hours**, versus "
                "**3.050 / 3.272 hours** for OPD: **+1.43% training** and **+1.52% total**. "
                "A separate full-batch, 8K one-step profile measured only **+1.11%**."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why this is a bounded, partial reproduction

    | Choice | Paper | Here |
    |---|---:|---:|
    | Global batch | 256 | 64 |
    | Maximum rollout | 8,192 | 4,096 tokens |
    | Training steps | 100 | 100 |
    | Student | Qwen3-1.7B | Qwen3-1.7B |
    | Quality evaluation | 14 benchmarks | MATH-500 + AIME24 |
    | Engine | TRL + colocated vLLM | direct PyTorch + Transformers SDPA |

    The rollout cap was consequential: at least one sequence hit 4,096 tokens in **61/66
    OPD steps** and **73/78 OPD2 steps** across seeds one/two. Average completions were also
    longer for OPD2 (2,489 vs 2,249 tokens). A full claim-level reproduction still needs the paper's batch,
    8K rollouts, complete evaluation suite, and the authors' implementation when released.

    ## Assessment

    - **Quality advantage:** directionally aligned in the two-seed mean but inconclusive under this setup; the ordering reversed by seed.
    - **Short runtime despite the extra pass:** aligned.

    See the [illustrated report](https://github.com/rehaanahmad2013/on-policy-delta-distillation/blob/main/reports/opd2-qwen17b/report.md)
    and [public source repository](https://github.com/rehaanahmad2013/on-policy-delta-distillation) for exact configurations and provenance.
    """)
    return


if __name__ == "__main__":
    app.run()
