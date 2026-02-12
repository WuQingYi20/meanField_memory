import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    if n == 0:
        return float("nan")
    return (gt - lt) / n


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(np.mean(sample))
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def eta_squared_oneway(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    grand_mean = df[value_col].mean()
    ss_between = 0.0
    ss_total = np.sum((df[value_col] - grand_mean) ** 2)
    for _, g in df.groupby(group_col):
        ss_between += len(g) * (g[value_col].mean() - grand_mean) ** 2
    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


def main() -> None:
    input_csv = Path("data/experiments/cognitive_lockin_pure_scan_full.csv")
    out_dir = Path("data/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input file: {input_csv}")

    df = pd.read_csv(input_csv)

    # Main summary by memory type
    summary = df.groupby("memory_type").agg(
        runs=("name", "count"),
        convergence_rate=("converged", "mean"),
        mean_conv_tick=("convergence_tick", "mean"),
        median_conv_tick=("convergence_tick", "median"),
        std_conv_tick=("convergence_tick", "std"),
        mean_final_majority=("final_majority_fraction", "mean"),
        mean_final_trust=("final_mean_trust", "mean"),
    ).reset_index()

    # Only converged rows for convergence-time inference
    conv = df[df["converged"] == True].copy()

    # Omnibus tests
    groups = [g["convergence_tick"].dropna().values for _, g in conv.groupby("memory_type")]
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    eta2 = eta_squared_oneway(conv.dropna(subset=["convergence_tick"]), "memory_type", "convergence_tick")

    # Pairwise tests
    pair_rows = []
    mem_types = sorted(conv["memory_type"].unique().tolist())
    for a, b in combinations(mem_types, 2):
        xa = conv.loc[conv["memory_type"] == a, "convergence_tick"].dropna().values
        xb = conv.loc[conv["memory_type"] == b, "convergence_tick"].dropna().values
        u_stat, p_val = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        delta = cliffs_delta(xa, xb)
        pair_rows.append({
            "group_a": a,
            "group_b": b,
            "n_a": len(xa),
            "n_b": len(xb),
            "mean_a": float(np.mean(xa)),
            "mean_b": float(np.mean(xb)),
            "median_a": float(np.median(xa)),
            "median_b": float(np.median(xb)),
            "mannwhitney_u": float(u_stat),
            "p_value": float(p_val),
            "cliffs_delta": float(delta),
        })
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_bonf"] = np.minimum(pairwise["p_value"] * len(pairwise), 1.0)

    # Stratified robustness checks: within each stratum, does dynamic beat fixed and decay?
    strata_cols = ["initial_strategy_0_fraction", "alpha", "beta", "initial_trust"]
    strata_rows = []
    for keys, g in conv.groupby(strata_cols):
        ticks_by_mem = {m: g.loc[g["memory_type"] == m, "convergence_tick"].dropna().values for m in ["dynamic", "fixed", "decay"]}
        if any(len(v) == 0 for v in ticks_by_mem.values()):
            continue

        means = {m: float(np.mean(v)) for m, v in ticks_by_mem.items()}
        medians = {m: float(np.median(v)) for m, v in ticks_by_mem.items()}
        rank = sorted(means.items(), key=lambda kv: kv[1])
        best = rank[0][0]

        # Kruskal inside this stratum
        h_s, p_s = stats.kruskal(ticks_by_mem["dynamic"], ticks_by_mem["fixed"], ticks_by_mem["decay"])

        strata_rows.append({
            "initial_strategy_0_fraction": keys[0],
            "alpha": keys[1],
            "beta": keys[2],
            "initial_trust": keys[3],
            "best_memory_by_mean": best,
            "mean_dynamic": means["dynamic"],
            "mean_fixed": means["fixed"],
            "mean_decay": means["decay"],
            "median_dynamic": medians["dynamic"],
            "median_fixed": medians["fixed"],
            "median_decay": medians["decay"],
            "kruskal_h": float(h_s),
            "kruskal_p": float(p_s),
        })

    strata = pd.DataFrame(strata_rows)

    # Bootstrap CIs for mean convergence tick
    ci_rows = []
    for mem, g in conv.groupby("memory_type"):
        vals = g["convergence_tick"].dropna().values
        lo, hi = bootstrap_mean_ci(vals)
        ci_rows.append({
            "memory_type": mem,
            "mean_convergence_tick": float(np.mean(vals)),
            "ci95_low": lo,
            "ci95_high": hi,
            "n": len(vals),
        })
    ci_df = pd.DataFrame(ci_rows)

    # Save tables
    summary_path = out_dir / "cognitive_lockin_pure_scan_full_summary.csv"
    pairwise_path = out_dir / "cognitive_lockin_pure_scan_full_pairwise.csv"
    strata_path = out_dir / "cognitive_lockin_pure_scan_full_strata.csv"
    ci_path = out_dir / "cognitive_lockin_pure_scan_full_ci.csv"
    report_path = out_dir / "cognitive_lockin_pure_scan_full_analysis.md"

    summary.to_csv(summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    strata.to_csv(strata_path, index=False)
    ci_df.to_csv(ci_path, index=False)

    # Compose markdown report
    dynamic_best_rate = float((strata["best_memory_by_mean"] == "dynamic").mean()) if len(strata) else float("nan")
    strata_sig_rate = float((strata["kruskal_p"] < 0.05).mean()) if len(strata) else float("nan")

    lines = []
    lines.append("# Pure Cognitive-Lockin Full Scan: Statistical Analysis")
    lines.append("")
    lines.append(f"- Input: `{input_csv}`")
    lines.append(f"- Total runs: `{len(df)}`")
    lines.append(f"- Converged runs used for convergence-time inference: `{len(conv)}`")
    lines.append("")
    lines.append("## Omnibus Tests (Convergence Tick by Memory Type)")
    lines.append(f"- One-way ANOVA: `F={f_stat:.3f}`, `p={f_p:.3e}`")
    lines.append(f"- Kruskal-Wallis: `H={h_stat:.3f}`, `p={h_p:.3e}`")
    lines.append(f"- Eta-squared (ANOVA): `eta^2={eta2:.3f}`")
    lines.append("")
    lines.append("## Key Robustness Checks")
    lines.append(f"- Dynamic is fastest (mean convergence tick) in `{dynamic_best_rate:.1%}` of parameter strata")
    lines.append(f"- Within-stratum memory-type difference significant (`p<0.05`) in `{strata_sig_rate:.1%}` of strata")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Summary table: `{summary_path}`")
    lines.append(f"- Pairwise tests: `{pairwise_path}`")
    lines.append(f"- Stratified tests: `{strata_path}`")
    lines.append(f"- Mean CI table: `{ci_path}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print(summary_path)
    print(pairwise_path)
    print(strata_path)
    print(ci_path)
    print(report_path)
    print("\nOmnibus:")
    print(f"ANOVA F={f_stat:.3f}, p={f_p:.3e}; Kruskal H={h_stat:.3f}, p={h_p:.3e}; eta2={eta2:.3f}")
    print(f"Dynamic fastest in {dynamic_best_rate:.1%} strata; significant in {strata_sig_rate:.1%} strata")


if __name__ == "__main__":
    main()
