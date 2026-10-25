import dotenv
import mlflow
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import f1_score, accuracy_score, recall_score
import torch

from paths import ENV_FILE
from data_processing.data_processing import lakeFS_parquet_to_df
from data_processing.augmentation_funcs import draw_scratch
from orchestrator.services.mlflow_service import MLflowService
dotenv.load_dotenv(ENV_FILE)
import matplotlib.pyplot as plt



def plot_performance_with_levels(scores, levels1, levels2):
    """
    Parameters
    ----------
    scores : ndarray of shape (3, n_levels)
        scores[0] = curvature
        scores[1] = blur
        scores[2] = curvature + blur

    levels1 : ndarray
        Curvature levels (x-axis)

    levels2 : ndarray
        Blur levels (displayed on secondary x-axis)
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(levels1, scores[0], marker="o", label="Curvature")
    ax.plot(levels1, scores[1], marker="s", label="Blur")
    ax.plot(levels1, scores[2], marker="^", label="Curvature + Blur")

    ax.set_xlabel("Curvature level")
    ax.set_ylabel("Metric mean")
    ax.set_xticks(levels1)
    ax.grid(True, alpha=0.3)

    # Secondary x-axis (blur sigma)
    secax = ax.secondary_xaxis("top")

    secax.set_xlabel("Gaussian blur σ")
    secax.set_xticks(curvature)
    secax.set_xticklabels([f"{s:.2f}" for s in sigmas])

    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_sample_grid(sample_images, curvature_levels, blur_levels):
    """
    Parameters
    ----------
    sample_images : list[list[pd.Series]]
        Outer list indexes drift level.
        Inner list contains:
            0 Curve class0
            1 Curve class1
            2 Blur class0
            3 Blur class1
            4 Combined class0
            5 Combined class1

    curvature_levels : iterable
    blur_levels : iterable
    """

    n_levels = len(sample_images)
    n_cols = 6

    fig, axes = plt.subplots(
        n_levels,
        n_cols,
        figsize=(2.2 * n_cols, 2.0 * n_levels),
        squeeze=False,
    )

    column_titles = [
        "Curve\nClass 0",
        "Curve\nClass 1",
        "Blur\nClass 0",
        "Blur\nClass 1",
        "Curve + Blur\nClass 0",
        "Curve + Blur\nClass 1",
    ]

    # Top column labels
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, fontsize=11)

    for i, row in enumerate(sample_images):

        for j, sample in enumerate(row):

            ax = axes[i, j]

            ax.imshow(sample["waferMap"], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

        # Label on right-hand side
        axes[i, -1].text(
            1.08,
            0.5,
            f"Curvature = {curvature_levels[i]:.2f}\n"
            f"Blur σ = {blur_levels[i]:.2f}",
            transform=axes[i, -1].transAxes,
            va="center",
            ha="left",
            fontsize=10,
        )

    plt.subplots_adjust(
        left=0.03,
        right=0.88,
        top=0.93,
        bottom=0.03,
        wspace=0.05,
        hspace=0.05,
    )

    plt.show()


def apply_blur_with_level(samples, sigma):
    return [blur(sample, sigma) for sample in samples]


def blur(grid, sigma):
    k = int(np.ceil(6 * sigma))
    k += (k + 1) % 2
    grid = grid.astype(np.float32)
    return cv2.GaussianBlur(grid, (k, k), sigma)


def apply_curvature_with_level(samples, level):
    return [
        draw_scratch(
            sample,
            **scratch_kwargs,
            curvature_range=level,
            verbose=False
        )
        for sample in samples
    ]


def evaluate_subset(model, subset):
    x = np.stack(subset["waferMap"].to_numpy())
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(1)

    y_true = subset["failure_label"].to_numpy()

    with torch.no_grad():
        logits, _ = model(x)

    y_pred = logits.argmax(dim=1).cpu().numpy()

    return metrics_mean(y_true, y_pred)


def metrics_mean(y_true, y_pred):
    return ( accuracy_score(y_true, y_pred) + recall_score(y_true, y_pred) + f1_score(y_true, y_pred) ) / 3


if __name__ == "__main__":

    scratch_kwargs = {
        "n_scratches_range": 4,
        "width_range": 0.01,
        "degree": 4,
        "length_range": 0.6,
        "circularity_range": (0.0, 1.0),
        "angle_range": (0, 2 * np.pi),
    }


    model_uri = "models:/simple_resnet_pilot/20"
    verfication_data_location = "masters-thesis/exp14b7a3d631f4a575/expdata/0/online/"
    mlflow_service = MLflowService()
    model = mlflow_service.get_model(model_uri)

    df = lakeFS_parquet_to_df(verfication_data_location, deser_col=["waferMap"])

    n_samples = 100

    class0 = df[df.failure_label == 0].copy()
    #class1_pre = df[df.failure_label == 1].copy()
    class1 = df[df.failure_label == 0].copy()

    df_class0 = class0.sample(n_samples // 2, random_state=0)
    df_class1 = class1.sample(n_samples // 2, random_state=1)
            

    sigmas = np.array([ 0.70, 0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
    curvature = np.linspace(0.0, 0.9, sigmas.shape[0])
    scores = np.zeros((3, curvature.shape[0]))
    sample_images = []


    for i in range(curvature.shape[0]):
        print(f"Round {i}/{curvature.shape[0]}")
        # input data is in column "waferMap"
        
        # generate a curavature subset where only class 1 is affected with apply_curvature with level curvature[i], concat to one big dataset and shuffle
        curve_class1 = df_class1.copy()

        curve_class1["waferMap"] = apply_curvature_with_level(
            curve_class1["waferMap"].tolist(),
            curvature[i]
        )
        curve_class1["failure_label"] = 1

        curve_subset = (
            pd.concat([df_class0, curve_class1], ignore_index=True)
            .sample(frac=1, random_state=1)
            .reset_index(drop=True)
        )


        # generate blur subset with same approach but blur affects both classes
        blur_class0 = df_class0.copy()
        blur_class1 = df_class1.copy()

        blur_class1["waferMap"] = apply_curvature_with_level(
                    curve_class1["waferMap"].tolist(),
                    0.0
                )
        blur_class1["failure_label"] = 1

        blur_class0["waferMap"] = apply_blur_with_level(
            blur_class0["waferMap"].tolist(),
            sigmas[i]
        )

        blur_class1["waferMap"] = apply_blur_with_level(
            blur_class1["waferMap"].tolist(),
            sigmas[i]
        )

        blur_subset = (
            pd.concat([blur_class0, blur_class1], ignore_index=True)
            .sample(frac=1, random_state=1)
            .reset_index(drop=True)
        )
        # generate subset where first scrathes are drawn to class 1 and after blur is applied
        combined_class0 = df_class0.copy()
        combined_class1 = df_class1.copy()

        combined_class1["waferMap"] = apply_curvature_with_level(
            combined_class1["waferMap"].tolist(),
            curvature[i]
        )
        combined_class1["failure_label"] = 1

        combined_class0["waferMap"] = apply_blur_with_level(
            combined_class0["waferMap"].tolist(),
            sigmas[i]
        )

        combined_class1["waferMap"] = apply_blur_with_level(
            combined_class1["waferMap"].tolist(),
            sigmas[i]
        )

        combined_subset = (
            pd.concat([combined_class0, combined_class1], ignore_index=True)
            .sample(frac=1, random_state=1)
            .reset_index(drop=True)
        )

        with torch.no_grad():
            # predict each subset and caluculate metrics mean by using "failure_label" as the true label column

            curve_score = evaluate_subset(model, curve_subset)
            blur_score = evaluate_subset(model, blur_subset)
            combined_score = evaluate_subset(model, combined_subset)


        scores[:, i] = np.array([curve_score, blur_score, combined_score])

        sample_images.append([
            curve_subset[curve_subset["failure_label"] == 0].iloc[10],
            curve_subset[curve_subset["failure_label"] == 1].iloc[10],
            blur_subset[blur_subset["failure_label"] == 0].iloc[10],
            blur_subset[blur_subset["failure_label"] == 1].iloc[10],
            combined_subset[combined_subset["failure_label"] == 0].iloc[10],
            combined_subset[combined_subset["failure_label"] == 1].iloc[10],
        ])


    plot_performance_with_levels(scores, curvature, sigmas)
    plot_sample_grid(
        sample_images,
        curvature,
        sigmas,
    )