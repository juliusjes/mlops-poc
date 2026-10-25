import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/experiment2"
TEMP_DIR = EXPERIMENT_ROOT / "temp"
OUT_DIR = REPOSITORY_ROOT / "outputs"
RESULT_FILE = EXPERIMENT_ROOT / "result.json"

RESULT = {}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def show_and_save(fig: Figure, filename):
    plt.show(block=True)

    filename = OUT_DIR / filename

    if input(f"Save figure to {filename}? [y/n]: ").strip().lower() == "y" or not ASK_SAVE:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Saved to {filename}")

    plt.close(fig)



def find_detection_from_csv(filepath: Path, filename: str):
    file = filepath / filename

    df = pd.read_csv(file)

    regex = r"^mmd.*\.(mmd2u|perc)$|^sample_index$"
    valid_df = df.dropna(subset=df.filter(regex=regex).columns)

    detection_index = None

    for c in valid_df.filter(regex=r"^mmd.*\.mmd2u$").columns:
        perc_col = c.rsplit(".", 1)[0] + ".perc"

        mask = valid_df[c] > valid_df[perc_col]

        if mask.any():
            detection_index = int(valid_df.loc[mask, "sample_index"].iloc[0])
            break
    return detection_index


def gather_state(regime) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(TEMP_DIR / "state.json", "r") as f:
        state: dict = json.load(f)

    state = state["completed"]
    subsets = state.get("subset")
    regimes = state.get("regime")
    log_files = state.get("associated_log_files")
    models = state.get("model")

    metrics_ = []
    regimesW = []
    subsets_ = []
    variants = []
    accuracy = []
    recall   = []
    f1       = []

    for i, (k, v) in enumerate(state.get("validation").items()):
        _, regimeW, subset, variant = k.split("_")
        regimeW = regimeW.strip("W200")  # noqa: B005
        value = v
        regimesW.append(regimeW)
        subsets_.append(subset)
        variants.append(variant)

        match i % 3:
            case 0: f1.append(value)
            case 1: recall.append(value)
            case 2: accuracy.append(value)

    df0 = pd.DataFrame(
        { "subset": subsets, "regime": regimes, "log_file": log_files, "model": models, }
        ) # 100 records

    df1 = pd.DataFrame(
        { "regime": regimesW, "subset": subsets_, "variant": variants, }
    ).drop_duplicates() # 267 records

    df2 = pd.DataFrame(
        { "f1": f1, "recall": recall, "accuracy": accuracy, }
    ) # 267 records

    # join df1 and df2 by concat, then result and df0 by subset and regime
    result = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

    df0["subset"] = pd.to_numeric(df0["subset"])
    result["subset"] = pd.to_numeric(result["subset"])

    # Join with df0 using subset and regime
    result = df0.merge(
        result,
        on=["subset", "regime"],
        how="left"
    )

    tp = TEMP_DIR
    result["final_filename"] = (
        result["regime"].astype(str) + "_" +
        result["subset"].astype(str) + "_" +
        result["log_file"].astype(str)
    )

    result["detection_index"] = [
        find_detection_from_csv(
            filepath=tp,
            filename=log_file
        )
        for  log_file in result["final_filename"]
    ]
    result["mean_metric"] = (result["f1"] + result["recall"] + result["accuracy"]) / 3

    cols = ["detection_index", "f1", "accuracy", "mean_metric", "recall", "regime", "subset"]

    baseline_pre = result.loc[
            (result["regime"] == regime) &
            (result["variant"] == "prebase"),
            cols
        ].dropna(subset=["detection_index"])

    baseline_post = result.loc[
        (result["regime"] == regime) &
        (result["variant"] == "postbase"),
        cols
    ].dropna(subset=["detection_index"])

    retrained = result.loc[
        (result["regime"] == regime) &
        (result["variant"] == "postretr"),
        cols
    ].dropna(subset=["detection_index"])

    return baseline_pre, baseline_post, retrained


def t_test(delta):
    result = ttest_1samp(delta, 0, alternative="greater")

    ci95 = result.confidence_interval(confidence_level=0.95)

    RESULT["n"] = len(delta)
    RESULT["mean_delta"] = np.mean(delta)
    RESULT["std_delta"] = np.std(delta, ddof=1)

    RESULT["t_stat"] = result.statistic
    RESULT["p_value"] = result.pvalue
    RESULT["df"] = result.df

    RESULT["ci95_low"] = ci95.low
    RESULT["ci95_high"] = ci95.high

    RESULT["reject_null"] = result.pvalue < 0.05

def plot_metric_delta_delay(baseline_pre, baseline_post, retrained, outlier_quantile=None, N=3000, drift_start=0.3):

    keys = ["regime", "subset", "detection_index"]

    baseline_pre = baseline_pre.rename(columns={
        "f1": "f1_pre",
        "accuracy": "accuracy_pre",
        "recall": "recall_pre",
        "mean_metric": "mean_metric_pre",
    })

    baseline_post = baseline_post.rename(columns={
        "f1": "f1_post",
        "accuracy": "accuracy_post",
        "recall": "recall_post",
        "mean_metric": "mean_metric_post",
    })

    retrained = retrained.rename(columns={
        "f1": "f1_retrained",
        "accuracy": "accuracy_retrained",
        "recall": "recall_retrained",
        "mean_metric": "mean_metric_retrained",
    })

    result = (
        baseline_pre
        .merge(baseline_post, on=keys, how="inner")
        .merge(retrained, on=keys, how="inner")
    )


    delta = (result["mean_metric_retrained"] - result["mean_metric_post"]).to_numpy()
    detection_delay = (result["detection_index"].to_numpy() - N*drift_start)

    t_test(delta)

    plot_excluded = False
    if outlier_quantile:
        lq, uq = np.quantile(delta, [outlier_quantile, 1 - outlier_quantile])
        mask = (delta >= lq) & (delta <= uq)
        excluded_delta = delta[~mask]
        delta = delta[mask]
        excluded_delay = detection_delay[~mask]
        detection_delay = detection_delay[mask]
        plot_excluded = True

    tp_delta = delta[detection_delay >= 0]
    tp_dd = detection_delay[detection_delay >= 0]
    fp_delta = delta[detection_delay < 0]
    fp_dd = detection_delay[detection_delay < 0]

    fig1, ax = plt.subplots()
    ax.scatter(tp_dd, tp_delta, color="blue" ,label="TP", alpha=0.7)
    ax.scatter(fp_dd, fp_delta, color="red" , marker="x",label="FP", alpha=0.7)

    if plot_excluded:
        ax.scatter(excluded_delay, excluded_delta, color="gray", alpha=0.7, label=f"Outside P{100 - 2*outlier_quantile*100:.0f}")

    extra = 0.02*(detection_delay.max() - detection_delay.min())
    lp, up = detection_delay.min() - extra, detection_delay.max() + extra
    ax.set_xlim(lp, up)
    ax.hlines(
        0, 
        lp, up,
        linestyles="--",
        color="black",
        label="No effect boundary"
        )
    ax.hlines(delta.mean(), lp, up, color="green", alpha = 0.7, label=f"Mean={delta.mean():.3f}")
    ax.set_ylabel("ΔP")
    ax.set_xlabel("Detection delay")
    plt.legend(fancybox=False)

    print(f"TP: {tp_dd.shape[0] + excluded_delta.shape[0]}, FP: {fp_dd.shape[0]}, ")

    return fig1


def plot_future_past(baseline_pre, baseline_post, retrained, metric="mean_metric"):

    keys = ["regime", "subset", "detection_index"]

    baseline_pre = baseline_pre.rename(columns={
        "f1": "f1_pre",
        "accuracy": "accuracy_pre",
        "recall": "recall_pre",
        "mean_metric": "mean_metric_pre",
    })

    baseline_post = baseline_post.rename(columns={
        "f1": "f1_post",
        "accuracy": "accuracy_post",
        "recall": "recall_post",
        "mean_metric": "mean_metric_post",
    })

    retrained = retrained.rename(columns={
        "f1": "f1_retrained",
        "accuracy": "accuracy_retrained",
        "recall": "recall_retrained",
        "mean_metric": "mean_metric_retrained",
    })

    result = (
        baseline_pre
        .merge(baseline_post, on=keys, how="inner")
        .merge(retrained, on=keys, how="inner")
    )


    pp = result[f"{metric}_pre"].to_numpy()
    fp_re = result[f"{metric}_retrained"].to_numpy() 
    fp_nr = result[f"{metric}_post"].to_numpy() 

    fig1, ax = plt.subplots()

    # Scatter plot
    x = fp_re - pp#fp_nr
    y = fp_nr - pp
    c = result["detection_index"].to_numpy() - 900

    sc = ax.scatter(x, y, c=c, cmap="viridis",alpha=0.8)# color="blue", alpha=0.8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(sc, cax=cax)

    #plt.colorbar(sc)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    ax.fill_between([2*xmin, 0], 0, 2*ymax, color="red", alpha=0.15)   # top left
    ax.fill_between([2*xmin, 0], 2*ymin, 0, color="yellow", alpha=0.15)  # bottom left
    ax.fill_between([0, 2*xmax], 2*ymin, 0, color="green", alpha=0.15)     # bottom right
    ax.fill_between([0, 2*xmax], 0, 2*ymax, color="yellow", alpha=0.15)  # top right

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_aspect("equal")
    ax.set_xlabel("ΔP w/ retraining")
    ax.set_ylabel("ΔP w/o retraining")
    ax.set_ylim(-0.15, 0.1)#ymin*1.05, ymax*1.05)
    ax.set_xlim(-0.38, 0.1)#xmin*1.05, xmax*1.05)
    plt.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.scatter(pp, fp_nr, label="Non-retrained", color="red", marker="^", alpha=0.7)
    ax2.scatter(pp, fp_re, label="Retrained", color="blue", alpha=0.7)
    ax2.set_xlabel("Pre detection performance")
    ax2.set_ylabel("Post detection performance")
    plt.legend(fancybox=False)

    count_tl = ((x < 0) & (y >= 0)).sum()
    count_tr = ((x >= 0) & (y >= 0)).sum()
    count_bl = ((x < 0) & (y < 0)).sum()
    count_br = ((x >= 0) & (y < 0)).sum()

    print(f"TL: {count_tl}, TR: {count_tr}, BL: {count_bl}, BR: {count_br}")

    return fig1, fig2


def find_pref_from_csv(filepath: Path, filename: str):
    file = filepath / filename

    df = pd.read_csv(file)

    regex = r"^metrics2e38.*\.(f1_rw50|accuracy_rw50|recall_rw50)$|^sample_index$"
    valid_df = df.dropna(subset=df.filter(regex=regex).columns)

    valid_df["pref"] = (
        valid_df["metrics2e38.accuracy_rw50"] +
        valid_df["metrics2e38.recall_rw50"] +
        valid_df["metrics2e38.f1_rw50"]
        ) / 3

    return valid_df["pref"].to_numpy()


def plot_performance_across_runs():
    prefs = []
    tp = TEMP_DIR

    with open(tp / "state.json", "r") as f:
        state: dict = json.load(f)["completed"]

    filenames = [
        REGIME + "_" + str(sbs) + "_" + lf
        for lf, reg, sbs in zip(
            state.get("associated_log_files", []),
            state.get("regime", []),
            state.get("subset", [])
        )
        if reg == REGIME
    ]

    print(len(filenames))

    for filename in filenames:
        pref = find_pref_from_csv(tp, filename)
        prefs.append(pref)

    all_perfs = np.stack(prefs)

    # Statistics across runs, for each point in the performance curve
    mean = np.mean(all_perfs, axis=0)
    q10, q25, q50, q75, q90 = np.quantile(all_perfs, [0.1, 0.25, 0.50, 0.75, 0.9], axis=0)

    x = np.linspace(0, 3000, all_perfs.shape[1])

    fig, ax = plt.subplots()

    ax.fill_between(
            x,
            q10,
            q90,
            alpha=0.2,
            label="10-90th percentile",
        )

    # Interquartile range
    ax.fill_between(
        x,
        q25,
        q75,
        alpha=0.2,
        label="25-75th percentile",
    )

    # Mean
    ax.plot(
        x,
        mean,
        linewidth=1,
        label="Mean",
        color="blue"
    )

    # Optional: median
    ax.plot(
        x,
        q50,
        linestyle="--",
        linewidth=1,
        label="Median",
    )
    ax.vlines(900, 1.2, 0, color="black", linestyles="dashdot", label="True change point")
    ax.set_ylim(0.65, 1.0)
    ax.set_xlim(0, 3000)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("P")
    ax.set_title(f"Performance across subsets ({REGIME})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    ASK_SAVE = False
    REGIME = "gradual"

    perf_fig = plot_performance_across_runs()

    baseline_pre, baseline_post, retrained = gather_state(regime=REGIME)
    metric_delta_fig = plot_metric_delta_delay(
        baseline_pre,
        baseline_post,
        retrained,
        outlier_quantile=0.025
        )

    train_gain_fig, pre_post_fig = plot_future_past(
        baseline_pre,
        baseline_post,
        retrained
        )

    filenames = [
        f"experiment2_metric_delta_{REGIME}.pdf",
        f"experiment2_train_gain_{REGIME}.pdf",
        f"experiment2_pre_post_{REGIME}.pdf",
        f"experiment2_perf_across_runs_{REGIME}.pdf"
        ]
    figs = [metric_delta_fig, train_gain_fig, pre_post_fig, perf_fig]
    for filename, fig in zip(filenames, figs):
        show_and_save(filename=filename, fig=fig)

    import json
    with open(RESULT_FILE, "w") as f:
        json.dump(RESULT, f, cls=NumpyEncoder)
