import argparse
from pathlib import Path
from dataclasses import asdict
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.runner import ExperimentConfig, ExperimentRunner


def build_matched_normative_configs(off_df: pd.DataFrame, max_ticks: int = 10000) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    for _, r in off_df.iterrows():
        cfg = ExperimentConfig(
            name=f"normon_{r['name']}",
            num_agents=int(r["num_agents"]),
            memory_type=str(r["memory_type"]),
            memory_size=int(r.get("memory_size", 5)),
            decision_mode="cognitive_lockin",
            initial_trust=float(r["initial_trust"]),
            alpha=float(r["alpha"]),
            beta=float(r["beta"]),
            max_ticks=max_ticks,
            random_seed=int(r["random_seed"]),
            enable_normative=True,
            # Keep observation off to isolate normative-memory toggle.
            observation_k=0,
            initial_strategy_0_fraction=float(r["initial_strategy_0_fraction"]),
        )
        configs.append(cfg)
    return configs


def run_normative_on_for_matched_grid(
    off_csv: Path,
    output_dir: Path,
    n_workers: int,
    max_ticks: int,
) -> pd.DataFrame:
    off_df = pd.read_csv(off_csv)
    configs = build_matched_normative_configs(off_df, max_ticks=max_ticks)

    runner = ExperimentRunner(output_dir=str(output_dir), n_workers=n_workers)
    runner.run_experiments(configs, parallel=True, progress=True)
    on_df = runner.get_results_dataframe()
    runner.save_results("cognitive_lockin_normative_on_matched")
    return on_df


def analyze_delta(off_df: pd.DataFrame, on_df: pd.DataFrame, output_dir: Path) -> None:
    keys = [
        "num_agents",
        "memory_type",
        "memory_size",
        "initial_strategy_0_fraction",
        "initial_trust",
        "alpha",
        "beta",
        "random_seed",
    ]

    off = off_df.copy()
    off["condition"] = "off"
    on = on_df.copy()
    on["condition"] = "on"

    merged = off.merge(on, on=keys, suffixes=("_off", "_on"), how="inner")

    merged["delta_convergence_tick_on_minus_off"] = (
        merged["convergence_tick_on"] - merged["convergence_tick_off"]
    )

    both_conv = merged[(merged["converged_off"] == True) & (merged["converged_on"] == True)].copy()

    paired_stats = {}
    if len(both_conv) > 0:
        diffs = both_conv["delta_convergence_tick_on_minus_off"].dropna().values
        if len(diffs) > 0:
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            paired_stats = {
                "n_paired_converged": int(len(diffs)),
                "mean_delta_on_minus_off": float(np.mean(diffs)),
                "median_delta_on_minus_off": float(np.median(diffs)),
                "wilcoxon_w": float(w_stat) if pd.notna(w_stat) else np.nan,
                "wilcoxon_p": float(w_p) if pd.notna(w_p) else np.nan,
                "faster_on_rate": float(np.mean(diffs < 0)),
            }

    conv_comp = pd.DataFrame([
        {
            "condition": "off",
            "convergence_rate": off_df["converged"].mean(),
            "mean_convergence_tick": off_df["convergence_tick"].mean(),
            "median_convergence_tick": off_df["convergence_tick"].median(),
            "mean_final_majority_fraction": off_df["final_majority_fraction"].mean(),
        },
        {
            "condition": "on",
            "convergence_rate": on_df["converged"].mean(),
            "mean_convergence_tick": on_df["convergence_tick"].mean(),
            "median_convergence_tick": on_df["convergence_tick"].median(),
            "mean_final_majority_fraction": on_df["final_majority_fraction"].mean(),
        },
    ])

    by_memory = []
    for mem in sorted(merged["memory_type"].unique().tolist()):
        m = merged[merged["memory_type"] == mem]
        both = m[(m["converged_off"] == True) & (m["converged_on"] == True)]
        d = both["delta_convergence_tick_on_minus_off"].dropna().values
        by_memory.append({
            "memory_type": mem,
            "n_pairs": int(len(m)),
            "conv_rate_off": float(m["converged_off"].mean()),
            "conv_rate_on": float(m["converged_on"].mean()),
            "mean_tick_off": float(m["convergence_tick_off"].mean()),
            "mean_tick_on": float(m["convergence_tick_on"].mean()),
            "mean_delta_on_minus_off": float(np.mean(d)) if len(d) else np.nan,
            "median_delta_on_minus_off": float(np.median(d)) if len(d) else np.nan,
            "faster_on_rate": float(np.mean(d < 0)) if len(d) else np.nan,
            "mean_first_norm_tick_on": float(m["first_norm_tick_on"].mean()),
            "mean_normative_level_tick_on": float(m["normative_level_tick_on"].mean()),
            "mean_final_norm_adoption_on": float(m["final_norm_adoption_rate_on"].mean()),
            "mean_final_norm_level_on": float(m["final_norm_level_on"].mean()),
        })
    by_memory_df = pd.DataFrame(by_memory)

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / "cognitive_lockin_normative_delta_matched_pairs.csv", index=False)
    conv_comp.to_csv(out_dir / "cognitive_lockin_normative_delta_overall.csv", index=False)
    by_memory_df.to_csv(out_dir / "cognitive_lockin_normative_delta_by_memory.csv", index=False)

    report_lines = []
    report_lines.append("# Cognitive-Lockin: Normative OFF vs ON (Matched Grid)")
    report_lines.append("")
    report_lines.append(f"- Paired runs: `{len(merged)}`")
    report_lines.append(f"- Both converged pairs: `{len(both_conv)}`")
    if paired_stats:
        report_lines.append(f"- Mean delta convergence tick (on-off): `{paired_stats['mean_delta_on_minus_off']:.3f}`")
        report_lines.append(f"- Median delta convergence tick (on-off): `{paired_stats['median_delta_on_minus_off']:.3f}`")
        report_lines.append(f"- Faster-on rate: `{paired_stats['faster_on_rate']:.1%}`")
        report_lines.append(
            f"- Wilcoxon signed-rank: `W={paired_stats['wilcoxon_w']:.3f}`, `p={paired_stats['wilcoxon_p']:.3e}`"
        )
    report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append("- `data/experiments/cognitive_lockin_normative_on_matched.csv`")
    report_lines.append("- `data/experiments/cognitive_lockin_normative_delta_matched_pairs.csv`")
    report_lines.append("- `data/experiments/cognitive_lockin_normative_delta_overall.csv`")
    report_lines.append("- `data/experiments/cognitive_lockin_normative_delta_by_memory.csv`")

    (out_dir / "cognitive_lockin_normative_delta_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved delta analysis artifacts to", out_dir)
    print("\nOverall:")
    print(conv_comp.to_string(index=False))
    print("\nBy memory:")
    print(by_memory_df.round(3).to_string(index=False))
    if paired_stats:
        print("\nPaired:")
        for k, v in paired_stats.items():
            print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and analyze matched normative ON/OFF comparison for cognitive_lockin.")
    parser.add_argument("--off-csv", type=str, default="data/experiments/cognitive_lockin_pure_scan_full.csv")
    parser.add_argument("--output-dir", type=str, default="data/experiments")
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser.add_argument("--n-workers", type=int, default=default_workers)
    parser.add_argument("--max-ticks", type=int, default=10000)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    off_csv = Path(args.off_csv)
    output_dir = Path(args.output_dir)

    off_df = pd.read_csv(off_csv)

    on_csv = output_dir / "cognitive_lockin_normative_on_matched.csv"

    if not args.skip_run:
        on_df = run_normative_on_for_matched_grid(
            off_csv=off_csv,
            output_dir=output_dir,
            n_workers=args.n_workers,
            max_ticks=args.max_ticks,
        )
    else:
        on_df = pd.read_csv(on_csv)

    analyze_delta(off_df=off_df, on_df=on_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
