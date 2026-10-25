import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import pandas as pd
import re
from collections.abc import Iterable, Sequence
from typing import Any
from matplotlib.figure import Figure

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/experiment1"
OUT_DIR = REPOSITORY_ROOT / "outputs"



def show_and_save(fig: Figure, filename):
    if SHOW:
        plt.show(block=True)

    filename = OUT_DIR / filename

    if not ASK_SAVE:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Saved to {filename}")
    else:
        if input(f"Save figure to {filename}? [y/n]: ").strip().lower() == "y":
            fig.savefig(filename, bbox_inches="tight")
            print(f"Saved to {filename}")

    if SHOW:
        plt.close(fig)


def concat_files_to_df() -> pd.DataFrame:
    files = sorted(Path(DATA_FOLDER).glob("*.csv"))
    dfs = []
    for run_id, f in enumerate(files):
        df = pd.read_csv(f)
        df["run"] = f.name.split("_", 2)[1]
        df["regime"] = f.name.split("_", 1)[0]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Read {run_id} files, resulting dataframe of size {len(df)}")
    return df
    
def aggregate_results(
    df, mean_std_patterns, mean_only_patterns,
):
    group_cols = ["regime", "sample_index"]

    mean_std_cols = [
        col
        for col in df.columns
        if any(re.search(pattern, col) for pattern in mean_std_patterns)
    ]

    mean_only_cols = [
        col
        for col in df.columns
        if any(re.search(pattern, col) for pattern in mean_only_patterns)
    ]

    mean_std = (
        df.groupby(group_cols, as_index=False)[mean_std_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    mean_only = (
        df.groupby(group_cols, as_index=False)[mean_only_cols]
        .mean()
    )

    return mean_std, mean_only


def collect_value_threshold_pairs(
    value_columns: Iterable[Any], threshold_columns: Iterable[Any], value_patterns: Sequence[str], threshold_patterns: Sequence[str],
) -> list[tuple[Any, Any]]:
    
    def first_key(column: Any) -> Any:
        return column[0] if isinstance(column, tuple) else column

    def extract(
        columns: Iterable[Any],
        patterns: Sequence[str],
    ) -> list[dict[str, Any]]:
        compiled_patterns = [re.compile(pattern) for pattern in patterns]
        result = []

        for column in columns:
            key = first_key(column)

            for pattern in compiled_patterns:
                match = pattern.fullmatch(key)
                if match:
                    groups = match.groupdict()
                    result.append(
                        {
                            "key": key,
                            "uuid": groups.get("uuid"),
                            "name": groups.get("name"),
                        }
                    )
                    break

        return result

    values = extract(value_columns, value_patterns)
    thresholds = extract(threshold_columns, threshold_patterns)

    values = list({value["key"]: value for value in values}.values())
    thresholds = list(
        {threshold["key"]: threshold for threshold in thresholds}.values()
    )

    result = []

    for value in values:
        if value["uuid"]:
            candidates = [
                threshold
                for threshold in thresholds
                if threshold["uuid"] == value["uuid"]
                and threshold["name"] == value["name"]
            ]
        else:
            # without UUID, only match another non-UUID threshold by name.
            candidates = [
                threshold
                for threshold in thresholds
                if threshold["uuid"] is None
                and threshold["name"] == value["name"]
            ]

        if not candidates:
            continue

        threshold = candidates[0]

        result.append(
            (value["key"], threshold["key"])
        )

        # Consume the threshold key. Since we've deduplicated by first key,
        # this removes all MultiIndex columns belonging to that threshold.
        thresholds.remove(threshold)

    return result


def plot_detector_signal(stats_thresholds, names=None):
    if len(names) != len(stats_thresholds):
        names = [st[0].split(".")[0] for st in stats_thresholds]
        print(f"Names: {names}")


    for j, (stat, threshold) in enumerate(stats_thresholds):
        fig = plt.figure(figsize=(10, 6))

        runs = []

        for i in df["run"].unique():
            data = (
                    df[(df["regime"] == REGIME) & (df["run"] == str(i))]
                    .sort_values("sample_index")
                    .reset_index(drop=True)
            )

            data = data[["sample_index", stat]].dropna(subset=[stat])
            runs.append(data)

        # Combine all runs
        data = pd.concat(runs, ignore_index=True)

        # Mean/std across runs at each sample_index
        stats = (
            data.groupby("sample_index")[stat]
            .agg(
                    mean="mean",
                    median="median",
                    q10=lambda x: x.quantile(0.10),
                    q90=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )

        x = stats["sample_index"]
        mean = stats["mean"]
        median = stats["median"]
        q10 = stats["q10"]
        q90 = stats["q90"]

        plt.plot(x, mean, label=f"{names[j]} mean")
        plt.plot(x, median, label=f"{names[j]} median")
        plt.fill_between(
            x,
            q10,
            q90,
            alpha=0.2,
            label="10-90th percentile",
        )

        plt.ylim(bottom=mean.mean() - 1.1*mean.mean(), top = 2*mean.mean())
        plt.xlim(0, FINAL_INDEX)
        plt.hlines(mo[threshold].mean(), *plt.xlim(), linestyle="--", linewidth=0.7, color="black", label="mean threshold")
        plt.vlines(TRUE_CHANGE, *plt.ylim(), linestyles="-.", color="black", label="true change")
        plt.title(f"{names[j]} detector metric, {REGIME} drift")
        plt.legend()
        show_and_save(fig, f"{EXPERIMENT_NAME}_{REGIME}_detector_stat_{names[j]}.pdf")
    return names


def plot_detections(df, stats_thresholds, names):
    detection_times = []

    metrics_after_detection = []
    metrics_before_change = []

    for stat, threshold in stats_thresholds:
        
        tds = np.zeros(len(df["run"].unique()))
        mad = np.zeros(len(df["run"].unique()))
        mbc = np.zeros(len(df["run"].unique()))

        for i in df["run"].unique():
            data = (
                    df[(df["regime"] == REGIME) & (df["run"] == str(i))]
                    .sort_values("sample_index")
                    .reset_index(drop=True)
                    )

            mask = ~data[stat].isna()
            y = data[stat][mask]

            crossing = data["sample_index"][data[stat] >= data[threshold].mean()]
            if len(crossing) == 0:
                tds[int(i)] = data["sample_index"].max()
            else:
                tds[int(i)] = crossing.iloc[0]


            start = int(tds[int(i)])
            end = min(start + 1000, data["sample_index"].max())

            mask = (data["sample_index"] >= start) & (data["sample_index"] <= end)

            acc = data["metrics.accuracy_rw50"][mask].mean()
            rec = data["metrics.recall_rw50"][mask].mean()
            f1 = data["metrics.f1_rw50"][mask].mean()
            mad[int(i)] = ((acc + rec + f1)/3)


            mask = (data["sample_index"] >= 0) & (data["sample_index"] < TRUE_CHANGE)
            
            acc = data["metrics.accuracy_rw50"][mask].mean()
            rec = data["metrics.recall_rw50"][mask].mean()
            f1 = data["metrics.f1_rw50"][mask].mean()
            mbc[int(i)] = ((acc + rec + f1)/3)

        detection_times.append(tds)
        metrics_after_detection.append(mad)
        metrics_before_change.append(mbc)



    # Detection time - performance
    H_LIM = sum([mbc.mean() for mbc in metrics_before_change]) / len(metrics_before_change)

    fig = plt.figure(figsize=(7,5))
    #names = ["Conformal", "MMD"]
    markers = ["s", "o", "v", "D"]
    for i in range(len(stats_thresholds)):
        mask = detection_times[i] < FINAL_INDEX

        plt.scatter(detection_times[i][mask], 1-metrics_after_detection[i][mask], label=names[i], marker=markers[i], edgecolors='black')
        plt.scatter(detection_times[i][~mask], 1-metrics_after_detection[i][~mask], marker=markers[i], edgecolors='black', color="gray")

    plt.title(f"Detection impact, {REGIME} drift")
    plt.xlabel("Detection time [sample index]")
    plt.ylabel("Performance degradation after change [1-metric]")
    plt.vlines(TRUE_CHANGE, *plt.ylim(), linestyles="-.", color="black", label="True change")
    plt.hlines(1-H_LIM, *plt.xlim(), linestyles="--", color="red", label="Mean performance before change")
    plt.fill_between( [TRUE_CHANGE, FINAL_INDEX], plt.ylim()[1], 1-H_LIM, alpha=0.15, label="High impact region")
    plt.legend(fancybox=False, loc="best")
    plt.xlim(0, N_samples)
    plt.tight_layout()
    show_and_save(fig, f"{EXPERIMENT_NAME}_{REGIME}_detection_impact.pdf")
    


    # Empirical cumulative distibution
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, td in enumerate(detection_times):

        td = np.asarray(td)
        td = td[np.isfinite(td)]
        x = np.sort(td)
        y = np.arange(1, len(x) + 1) / len(x)

        # Explicitly start the ecdf at zero
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0)

        ax.step( x, y, where="post", label=names[i] )

    # True change point
    ax.axvline( TRUE_CHANGE, linestyle="-.", label="True change", color="black" )

    ax.set_xlabel("Detection time [sample index]")
    ax.set_ylabel("Fraction detected")
    ax.set_ylim(0, 1)

    ax.set_title(f"Empirical CDF of detection times, {REGIME} drift")
    ax.legend(fancybox=False)

    plt.tight_layout()
    plt.xlim(0, FINAL_INDEX + 50)
    show_and_save(fig, f"{EXPERIMENT_NAME}_{REGIME}_ecdf.pdf")
    return detection_times


def calculate_value_se(detection_times):
    values = []
    ses = []

    for i in range(len(stats_thresholds)):
        tpr, tpr_se = true_pos_rate( TRUE_CHANGE, detection_times[i], FINAL_INDEX )
        fpr, fpr_se = false_positive_rate( TRUE_CHANGE, detection_times[i] )
        miss, miss_se = missed( TRUE_CHANGE, detection_times[i], FINAL_INDEX )

        values.append([ tpr, fpr, miss ])
        ses.append([ tpr_se, fpr_se, miss_se ])

    values = np.array(values)
    ses = np.array(ses)
    return values, ses


def plot_metrics(detection_times):
    metrics = [
        "True positive rate",
        "False positive rate",
        "Missed"
    ]

    # Compute metrics and their standard errors for each dds[i]
    values, ses = calculate_value_se(detection_times)

    # X positions for the metrics
    x = np.arange(len(metrics))

    # Width of each bar
    width = 0.8 / len(stats_thresholds)

    fig, ax = plt.subplots(figsize=(5, 4))

    print("Means and standard errors: \n")

    # One set of bars per dds[i]
    for i in range(len(stats_thresholds)):
        offset = i * width - (len(stats_thresholds) - 1) * width / 2

        ax.bar(
            x + offset,
            values[i],
            width,
            yerr=ses[i],
            capsize=3,
            label=f"{names[i]}",
            edgecolor="black",
        )
        print(f"{names[i]}: \n" \
                f"\t tpr:  {values[i][0]:.3f} +- {ses[i][0]:.3f},\n" \
                f"\t fpr:  {values[i][1]:.3f} +- {ses[i][1]:.3f},\n" \
                f"\t Miss: {values[i][2]:.3f} +- {ses[i][2]:.3f}\n")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Metric")
    ax.set_title("Change-point detection performance")

    ax.legend(fancybox=False, loc="center right")

    plt.tight_layout()
    show_and_save(fig, f"{EXPERIMENT_NAME}_{REGIME}_detection_metrics.pdf")


def plot_secondary_metrics(stats_thresholds):

    values, ses = calculate_value_se(stats_thresholds)
    
    # values
    vm2 = np.zeros((len(stats_thresholds), 3))
    vm2_se = np.zeros_like(vm2)

    for i in range(len(stats_thresholds)):
        tp, fp, fn = values[i]
        r, ser = recall(tp*60, fn*60)
        p, sep = precision(tp*60, fp*60)
        f, sef = fdr(tp*60, fp*60)

        vm2[i,:] = np.array([r, p, f])
        vm2_se[i,:] = np.array([ser, sep, sef])

    metrics = [
        "Recall",
        "Precision",
        "FDR"
    ]
    x = np.arange(len(metrics))

    # Width of each bar
    width = 0.8 / len(stats_thresholds)

    fig, ax = plt.subplots(figsize=(5, 4))

    print("Means and standard errors: \n")

    # One set of bars per dds[i]
    for i in range(len(stats_thresholds)):
        offset = i * width - (len(stats_thresholds) - 1) * width / 2

        ax.bar(
            x + offset,
            vm2[i],
            width,
            yerr=vm2_se[i],
            capsize=3,
            label=f"{names[i]}",
            edgecolor="black",
        )
        print(f"{names[i]}: \n" \
                f"\t recall   :  {vm2[i][0]:.3f} +- {vm2_se[i][0]:.3f},\n" \
                f"\t precision:  {vm2[i][1]:.3f} +- {vm2_se[i][1]:.3f},\n" \
                f"\t FPR      :  {vm2[i][2]:.3f} +- {vm2_se[i][2]:.3f}\n")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value")
    #ax.set_xlabel("Metric")
    #ax.set_title("Change-point detection performance")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.legend(fancybox=False, loc="center right")

    show_and_save(fig, f"{EXPERIMENT_NAME}_{REGIME}_detection_metrics2.pdf")


def plot_detection_delay(detection_times):
    x = np.arange(1)

    # Width of each bar
    width = 0.8 / len(stats_thresholds)

    fig, ax = plt.subplots(figsize=(5, 4))
    print("Means and standard deviations: \n")
    for i in range(len(stats_thresholds)):
        td = detection_times[i]
        mu, std, n = ttd(TRUE_CHANGE, td, FINAL_INDEX)

        print(f"{names[i]}:\n \ttd: {mu:.3f} +- {std:.3f},\n\tn: {n}\n")

        offset = i * width - (len(stats_thresholds) - 1) * width / 2

        ax.bar(
            x + offset,
            mu,
            width,
            yerr=std,
            capsize=3,
            label=f"{names[i]}",
            edgecolor="black"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Detector"])
    ax.set_ylabel("Detection delay [samples]")
    ax.set_xlabel("")
    #ax.set_title("Change-point detection delay")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    show_and_save(fig, f"{EXPERIMENT_NAME}_{REGIME}_detection_delay.pdf")


def plot_pilot_n_trials():
    z_1minusalpha = 1.644  # 90% two-sided CI
    E = 144
    sigma_squared = (0.3*126 + 0.7*908)**2
    print(sigma_squared**0.5)
    n = (z_1minusalpha**2 * sigma_squared) / E**2

    print(f"Number of trials needed: {np.ceil(n):.0f}")

    fig = plt.figure(figsize=(7,5))
    x = np.arange(50, 200, 10)
    y = (z_1minusalpha**2 * sigma_squared) / x**2
    #plt.yscale("log")
    plt.plot(x,y, label=f"std = {sigma_squared**0.5:.1f}, 90% CI")
    plt.scatter([144], [60], label="Experiment 1", marker="o", edgecolors="black")
    plt.grid()
    plt.grid(which="minor", color="0.9")
    plt.xlabel("Desired standard error [samples]")
    plt.ylabel("Number of trials required")
    plt.annotate(f"E = 144, N = 60", (144,60), xytext=(144, 75))
    plt.legend()
    show_and_save(fig, f"pilot_{REGIME}_number_of_trials.pdf")


# DETECTION METRICS

def ttd(tau: int, td: np.ndarray, td_max):
    """Mean and std. dev. of detection delay for successful detections."""
    mask = (td >= tau) & (td < td_max)
    delays = td[mask] - tau

    if delays.size == 0:
        return np.nan, np.nan, 0

    return delays.mean(), delays.std(ddof=0), delays.size


def true_pos_rate(tau: int, td: np.ndarray, td_max):
    """Probability of detecting after the change."""
    mask = (td >= tau) & (td < td_max)
    n = mask.size
    p = np.mean(mask)

    # Standard error of the estimated probability
    se = np.sqrt(p * (1 - p) / n)

    return p, se


def false_positive_rate(tau: int, td: np.ndarray):
    """Probability of detecting before the change."""
    mask = td < tau
    n = mask.size
    p = np.mean(mask)

    se = np.sqrt(p * (1 - p) / n)

    return p, se

def missed(tau: int, td: np.ndarray, td_max):
    """Probability of never detecting within the monitoring horizon."""
    mask = td >= td_max
    n = mask.size
    p = np.mean(mask)

    se = np.sqrt(p * (1 - p) / n)

    return p, se


def recall(tp, fn):
    """
    Returns recall and its standard error.

    inputs are counts
    """
    n = tp + fn
    p = tp / n
    se = np.sqrt(p * (1 - p) / n)
    return p, se


def precision(tp, fp):
    """
    Returns precision and its standard error.

    inputs are counts
    """
    n = tp + fp
    p = tp / n
    se = np.sqrt(p * (1 - p) / n)
    return p, se


def fdr(tp, fp):
    """
    Returns false discovery rate and its standard error.

    inputs are counts
    """
    n = tp + fp
    p = fp / n
    se = np.sqrt(p * (1 - p) / n)
    return p, se



if __name__ == "__main__":

    EXPERIMENT_NAME = "experiment1"
    DATA_FOLDER = EXPERIMENT_ROOT / "temp/20260815_200018/"
    N_samples = 5000
    FINAL_INDEX = 4999
    TRUE_CHANGE = int(N_samples*0.3)
    REGIME = "gradual"

    ASK_SAVE = False
    SHOW = False

    stat_patterns = [
        r"^metrics[a-fA-F0-9]{0,4}\.",
        r"^mmd[a-fA-F0-9]{0,4}\.mmd2u$",
        r"^conformal[a-fA-F0-9]{0,4}\.stat$",
        r"^c2st[a-fA-F0-9]{0,4}\.roc_aucV$",
        r"^energydistance[a-fA-F0-9]{0,4}\.res$"
    ]
    threshold_patterns = [
        r"^conformal[a-fA-F0-9]{0,4}\.threshold$",
        r"^mmd[a-fA-F0-9]{0,4}\.perc$",
        r"^c2st[a-fA-F0-9]{0,4}\.rand_roc_aucT$",
        r"^energydistance[a-fA-F0-9]{0,4}\.perc$"
    ]

    names = [
        ""
    ]

    df = concat_files_to_df()
    ms, mo = aggregate_results(df, stat_patterns, threshold_patterns)
    stats_thresholds = collect_value_threshold_pairs(ms.columns, mo.columns, stat_patterns, threshold_patterns)

    names = plot_detector_signal(stats_thresholds, names)
    detection_times = plot_detections(df, stats_thresholds, names)

    plot_detection_delay(detection_times)
    plot_metrics(detection_times)
    plot_secondary_metrics(detection_times)

