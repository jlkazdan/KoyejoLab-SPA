#!/usr/bin/env python3
"""
Plot cross-family question-level error correlations from local response CSVs.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FAMILY_MAP = {
    "openai.gpt-oss-20b-1:0": "GPT-OSS",
    "openai.gpt-oss-120b-1:0": "GPT-OSS",
    "qwen.qwen3-32b-v1:0": "Qwen",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen",
    "google/gemma-3-4b-it": "Gemma",
}

FAMILY_ORDER = ["GPT-OSS", "Qwen", "Gemma"]
ANSWER_ORDER = ["Truth"] + FAMILY_ORDER
FAMILY_COLORS = {
    "GPT-OSS": "#4c78a8",
    "Qwen": "#f58518",
    "Gemma": "#54a24b",
}

DATASET_LABELS = {
    "cais/hle": "HLE",
    "google/boolq": "BoolQ",
    "tasksource/com2sense": "Com2Sense",
    "kyssen/predict-the-futurebench-cutoff-June25": "Futurebench",
}


def load_responses(data_dir: Path) -> pd.DataFrame:
    usecols = [
        "config_model_id",
        "config_dataset_name",
        "question_idx",
        "response_type",
        "extracted_answer",
        "true_answer",
        "config_temperature",
        "config_experiment_type",
    ]
    frames = []
    for path in data_dir.glob("*_responses.csv"):
        frames.append(pd.read_csv(path, usecols=usecols))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def filter_responses(df: pd.DataFrame, experiment: str, temperature: Optional[float]) -> pd.DataFrame:
    df = df[df["response_type"] == "direct_answer"].copy()
    df["extracted_answer"] = df["extracted_answer"].astype(str).str.lower()
    df["true_answer"] = df["true_answer"].astype(str).str.lower()
    df = df[df["extracted_answer"].notna() & df["true_answer"].notna()]
    df = df[df["extracted_answer"] != "unclear"]

    if experiment:
        df = df[df["config_experiment_type"] == experiment]
    if temperature is not None:
        df = df[df["config_temperature"] == temperature]

    df["correct"] = df["extracted_answer"] == df["true_answer"]
    return df


def compute_family_question_errors(df: pd.DataFrame) -> pd.DataFrame:
    # Per model per question error rate
    q_stats = (
        df.groupby(["config_dataset_name", "config_model_id", "question_idx"])
        .agg(error_rate=("correct", lambda x: 1.0 - x.mean()))
        .reset_index()
    )
    q_stats["family"] = q_stats["config_model_id"].map(FAMILY_MAP)
    q_stats = q_stats[q_stats["family"].notna()]

    # Average across models within family for each question
    family_q = (
        q_stats.groupby(["config_dataset_name", "family", "question_idx"])
        .agg(error_rate=("error_rate", "mean"))
        .reset_index()
    )
    return family_q


def map_binary_answer(answer: str) -> Optional[int]:
    if answer in ("yes", "true"):
        return 1
    if answer in ("no", "false"):
        return 0
    return None


def compute_family_question_yes_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["answer_positive"] = df["extracted_answer"].map(map_binary_answer)
    df = df.dropna(subset=["answer_positive"])

    q_yes = (
        df.groupby(["config_dataset_name", "config_model_id", "question_idx"])
        .agg(yes_rate=("answer_positive", "mean"))
        .reset_index()
    )
    q_yes["family"] = q_yes["config_model_id"].map(FAMILY_MAP)
    q_yes = q_yes[q_yes["family"].notna()]

    family_yes = (
        q_yes.groupby(["config_dataset_name", "family", "question_idx"])
        .agg(yes_rate=("yes_rate", "mean"))
        .reset_index()
    )
    return family_yes


def compute_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    truth = df[["config_dataset_name", "question_idx", "true_answer"]].drop_duplicates()
    truth["truth_label"] = truth["true_answer"].map(map_binary_answer)
    truth = truth.dropna(subset=["truth_label"])
    return truth


def correlation_by_dataset(family_q: pd.DataFrame) -> dict:
    corr_by_dataset = {}
    for dataset, dgroup in family_q.groupby("config_dataset_name"):
        pivot = dgroup.pivot(index="question_idx", columns="family", values="error_rate")
        pivot = pivot.dropna()
        if pivot.empty:
            continue
        corr = pivot.corr()
        corr_by_dataset[dataset] = {
            "corr": corr,
            "n_questions": len(pivot),
        }
    return corr_by_dataset


def pooled_zscore_correlation(family_q: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    zscores = []
    total_questions = 0
    for dataset, dgroup in family_q.groupby("config_dataset_name"):
        pivot = dgroup.pivot(index="question_idx", columns="family", values="error_rate")
        pivot = pivot.dropna()
        if pivot.empty:
            continue
        total_questions += len(pivot)
        # Normalize within each dataset so pooled correlations aren't driven by base error rates.
        z = (pivot - pivot.mean()) / pivot.std(ddof=0)
        zscores.append(z)
    pooled = pd.concat(zscores) if zscores else pd.DataFrame()
    if pooled.empty:
        return pd.DataFrame(), 0
    return pooled.corr(), total_questions


def answer_correlation_with_truth_by_dataset(
    family_yes: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict:
    corr_by_dataset = {}
    for dataset, dgroup in family_yes.groupby("config_dataset_name"):
        pivot = dgroup.pivot(index="question_idx", columns="family", values="yes_rate")
        truth_subset = truth[truth["config_dataset_name"] == dataset].set_index("question_idx")
        if not truth_subset.empty:
            pivot = pivot.join(truth_subset["truth_label"].rename("Truth"), how="inner")
        pivot = pivot.dropna()
        if pivot.empty:
            continue
        corr = pivot.corr()
        corr_by_dataset[dataset] = {
            "corr": corr,
            "n_questions": len(pivot),
        }
    return corr_by_dataset


def answer_correlation_by_dataset(family_yes: pd.DataFrame) -> dict:
    corr_by_dataset = {}
    for dataset, dgroup in family_yes.groupby("config_dataset_name"):
        pivot = dgroup.pivot(index="question_idx", columns="family", values="yes_rate")
        pivot = pivot.dropna()
        if pivot.empty:
            continue
        corr = pivot.corr()
        corr_by_dataset[dataset] = {
            "corr": corr,
            "n_questions": len(pivot),
        }
    return corr_by_dataset


def pooled_answer_correlation_with_truth(
    family_yes: pd.DataFrame,
    truth: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    zscores = []
    total_questions = 0
    for dataset, dgroup in family_yes.groupby("config_dataset_name"):
        pivot = dgroup.pivot(index="question_idx", columns="family", values="yes_rate")
        truth_subset = truth[truth["config_dataset_name"] == dataset].set_index("question_idx")
        if not truth_subset.empty:
            pivot = pivot.join(truth_subset["truth_label"].rename("Truth"), how="inner")
        pivot = pivot.dropna()
        if pivot.empty:
            continue
        total_questions += len(pivot)
        z = (pivot - pivot.mean()) / pivot.std(ddof=0)
        zscores.append(z)
    pooled = pd.concat(zscores) if zscores else pd.DataFrame()
    if pooled.empty:
        return pd.DataFrame(), 0
    return pooled.corr(), total_questions


def reindex_corr(corr: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    return corr.reindex(index=order, columns=order)


def plot_heatmap(corr: pd.DataFrame, title: str, out_path: Path, order: List[str]) -> None:
    corr = reindex_corr(corr, order)
    mask = corr.isna()
    plt.figure(figsize=(4.2, 3.6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        mask=mask,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_dataset_grid(corr_by_dataset: dict, out_path: Path, order: List[str]) -> None:
    datasets = [d for d in DATASET_LABELS.keys() if d in corr_by_dataset]
    if not datasets:
        return

    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.6, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, dataset in zip(axes, datasets):
        corr = reindex_corr(corr_by_dataset[dataset]["corr"], order)
        mask = corr.isna()
        label = DATASET_LABELS.get(dataset, dataset)
        n_q = corr_by_dataset[dataset]["n_questions"]
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar=False,
            mask=mask,
            ax=ax,
        )
        ax.set_title(f"{label} (n={n_q})")

    # Hide any unused subplots
    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_answer_agreement_vs_truth_panels(
    family_yes: pd.DataFrame,
    answer_corr_by_dataset: dict,
    out_path: Path,
) -> None:
    datasets = [d for d in DATASET_LABELS.keys() if d in answer_corr_by_dataset]
    if not datasets:
        return

    nrows = len(datasets)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(8.6, 3.2 * nrows))
    axes = np.array(axes).reshape(nrows, 2)

    family_corr_by_dataset = answer_correlation_by_dataset(family_yes)

    for i, dataset in enumerate(datasets):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        family_payload = family_corr_by_dataset.get(dataset)
        if not family_payload:
            ax_left.axis("off")
            ax_right.axis("off")
            continue

        corr = reindex_corr(family_payload["corr"], FAMILY_ORDER)
        mask = corr.isna()
        label = DATASET_LABELS.get(dataset, dataset)
        n_q = family_payload["n_questions"]

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar=(i == 0),
            cbar_kws={"label": "Pearson r"} if i == 0 else None,
            mask=mask,
            ax=ax_left,
        )
        ax_left.set_title(f"{label} (n={n_q})")

        truth_corr = answer_corr_by_dataset[dataset]["corr"]
        labels = []
        values = []
        colors = []
        for fam in FAMILY_ORDER:
            if "Truth" in truth_corr.index and fam in truth_corr.columns:
                val = truth_corr.loc["Truth", fam]
            elif "Truth" in truth_corr.columns and fam in truth_corr.index:
                val = truth_corr.loc[fam, "Truth"]
            else:
                val = np.nan
            if pd.isna(val):
                continue
            labels.append(fam)
            values.append(val)
            colors.append(FAMILY_COLORS.get(fam, "#888888"))

        y_pos = np.arange(len(labels))
        ax_right.barh(y_pos, values, color=colors, alpha=0.85)
        ax_right.axvline(0, color="black", linewidth=1)
        ax_right.set_yticks(y_pos)
        ax_right.set_yticklabels(labels)
        ax_right.set_xlim(-1, 1)
        ax_right.set_xlabel("Correlation with Truth")
        if i == 0:
            ax_right.set_title("Truth alignment (r)")

        for y_i, val in zip(y_pos, values):
            offset = 0.03 if val >= 0 else -0.03
            ha = "left" if val >= 0 else "right"
            ax_right.text(val + offset, y_i, f"{val:.2f}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_summary_csv(
    corr_by_dataset: dict,
    pooled_corr: pd.DataFrame,
    pooled_n: int,
    out_path: Path,
    experiment: str,
    temperature: Optional[float],
) -> None:
    rows = []
    for dataset, payload in corr_by_dataset.items():
        corr = reindex_corr(payload["corr"], FAMILY_ORDER)
        for i in corr.index:
            for j in corr.columns:
                if pd.isna(corr.loc[i, j]):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "family_i": i,
                        "family_j": j,
                        "correlation": float(corr.loc[i, j]),
                        "n_questions": payload["n_questions"],
                        "experiment": experiment,
                        "temperature": temperature,
                    }
                )
    if not pooled_corr.empty:
        pooled_corr = reindex_corr(pooled_corr, FAMILY_ORDER)
        for i in pooled_corr.index:
            for j in pooled_corr.columns:
                if pd.isna(pooled_corr.loc[i, j]):
                    continue
                rows.append(
                    {
                        "dataset": "pooled_zscore",
                        "family_i": i,
                        "family_j": j,
                        "correlation": float(pooled_corr.loc[i, j]),
                        "n_questions": pooled_n,
                        "experiment": experiment,
                        "temperature": temperature,
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_answer_summary_csv(
    corr_by_dataset: dict,
    pooled_corr: pd.DataFrame,
    pooled_n: int,
    out_path: Path,
    experiment: str,
    temperature: Optional[float],
) -> None:
    rows = []
    for dataset, payload in corr_by_dataset.items():
        corr = reindex_corr(payload["corr"], ANSWER_ORDER)
        for i in corr.index:
            for j in corr.columns:
                if pd.isna(corr.loc[i, j]):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model_i": i,
                        "model_j": j,
                        "correlation": float(corr.loc[i, j]),
                        "n_questions": payload["n_questions"],
                        "experiment": experiment,
                        "temperature": temperature,
                    }
                )
    if not pooled_corr.empty:
        pooled_corr = reindex_corr(pooled_corr, ANSWER_ORDER)
        for i in pooled_corr.index:
            for j in pooled_corr.columns:
                if pd.isna(pooled_corr.loc[i, j]):
                    continue
                rows.append(
                    {
                        "dataset": "pooled_zscore",
                        "model_i": i,
                        "model_j": j,
                        "correlation": float(pooled_corr.loc[i, j]),
                        "n_questions": pooled_n,
                        "experiment": experiment,
                        "temperature": temperature,
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-family error correlation heatmaps.")
    parser.add_argument("--data-dir", type=Path, default=Path("notebooks/01_spa/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("notebooks/01_spa/results"))
    parser.add_argument("--experiment", type=str, default="surprisingly_popular")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = load_responses(args.data_dir)
    if df.empty:
        raise SystemExit("No response CSVs found.")

    df = filter_responses(df, args.experiment, args.temperature)
    family_q = compute_family_question_errors(df)
    corr_by_dataset = correlation_by_dataset(family_q)
    pooled_corr, pooled_n = pooled_zscore_correlation(family_q)

    family_yes = compute_family_question_yes_rates(df)
    truth = compute_truth_labels(df)
    answer_corr_by_dataset = answer_correlation_with_truth_by_dataset(family_yes, truth)
    pooled_answer_corr, pooled_answer_n = pooled_answer_correlation_with_truth(family_yes, truth)

    sns.set_style("white")

    if not pooled_corr.empty:
        plot_heatmap(
            pooled_corr,
            "Cross-family error correlation (pooled, z-scored)",
            args.out_dir / "cross_family_error_correlation_heatmap.png",
            FAMILY_ORDER,
        )

    plot_dataset_grid(
        corr_by_dataset,
        args.out_dir / "cross_family_error_correlation_by_dataset.png",
        FAMILY_ORDER,
    )

    if not pooled_answer_corr.empty:
        plot_heatmap(
            pooled_answer_corr,
            "Cross-family answer correlation with Truth (pooled, z-scored)",
            args.out_dir / "cross_family_answer_correlation_with_truth_heatmap.png",
            ANSWER_ORDER,
        )

    plot_dataset_grid(
        answer_corr_by_dataset,
        args.out_dir / "cross_family_answer_correlation_with_truth_by_dataset.png",
        ANSWER_ORDER,
    )

    plot_answer_agreement_vs_truth_panels(
        family_yes,
        answer_corr_by_dataset,
        args.out_dir / "cross_family_answer_agreement_vs_truth_panels.png",
    )

    save_summary_csv(
        corr_by_dataset,
        pooled_corr,
        pooled_n,
        args.results_dir / "cross_family_error_correlation_summary.csv",
        args.experiment,
        args.temperature,
    )

    save_answer_summary_csv(
        answer_corr_by_dataset,
        pooled_answer_corr,
        pooled_answer_n,
        args.results_dir / "cross_family_answer_correlation_with_truth_summary.csv",
        args.experiment,
        args.temperature,
    )

    print("Wrote:")
    print(f"- {args.out_dir / 'cross_family_error_correlation_heatmap.png'}")
    print(f"- {args.out_dir / 'cross_family_error_correlation_by_dataset.png'}")
    print(f"- {args.out_dir / 'cross_family_answer_correlation_with_truth_heatmap.png'}")
    print(f"- {args.out_dir / 'cross_family_answer_correlation_with_truth_by_dataset.png'}")
    print(f"- {args.out_dir / 'cross_family_answer_agreement_vs_truth_panels.png'}")
    print(f"- {args.results_dir / 'cross_family_error_correlation_summary.csv'}")
    print(f"- {args.results_dir / 'cross_family_answer_correlation_with_truth_summary.csv'}")


if __name__ == "__main__":
    main()
