import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from ml_platform.shared.utils import info
from ml_platform.shared.exceptions import DataformatException

def pad(grid: np.ndarray, h=1, w=1) -> np.ndarray:
    """
    Pad a numpy array with zeros to match the target aspect ratio. Ratio is interpreted as width:height (w:h).
    Padding is applied symmetrically where possible.
    """
    if h <= 0 or w <= 0:
        raise ValueError("Ratio values must be > 0")

    current_h, current_w = grid.shape[:2]

    target_ratio = w / h
    current_ratio = current_w / current_h

    if np.isclose(current_ratio, target_ratio):
        return grid

    if current_ratio < target_ratio:
        target_w = int(np.ceil(current_h * target_ratio))
        pad_left = (target_w - current_w) // 2
        pad_right = (target_w - current_w) - pad_left
        pad_top = pad_bottom = 0

    else:
        target_h = int(np.ceil(current_w / target_ratio))
        pad_top = (target_h - current_h) // 2
        pad_bottom = (target_h - current_h) - pad_top
        pad_left = pad_right = 0

    return cv2.copyMakeBorder(
        grid,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def scale(grid: np.ndarray, target_size=None) -> np.ndarray:
    """
    Resize grid to target_size using nearest-neighbor interpolation. If target_size is None, return input unchanged.

    target_size format:
        (width, height)
    """
    if target_size is None:
        return grid
    target_w, target_h = target_size
    return cv2.resize(grid, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def is_size(grid, w_max=None, w_min=None, h_max=None, h_min=None) -> bool:
    """
    Check whether a grid falls within the given width/height bounds.
    Any bound can be None to disable that constraint.
    """
    h, w = grid.shape[:2]
    return (
        (w_min is None or w >= w_min)
        and (w_max is None or w <= w_max)
        and (h_min is None or h >= h_min)
        and (h_max is None or h <= h_max)
    )


def LSWMD_pad(df, **kwargs):
    for i in tqdm(range(len(df)), desc=info(return_string=True)):
        df.at[i, "waferMap"] = pad(df.at[i, "waferMap"], **kwargs)
    return df


def LSWMD_scale(df, **kwargs):
    for i in tqdm(range(len(df)), desc=info(return_string=True)):
        df.at[i, "waferMap"] = scale(df.at[i, "waferMap"], **kwargs)
    return df


def preprocess_wafermaps(
    df: pd.DataFrame, target_hw: tuple[int, int], limits: tuple[int, int, int, int]
) -> pd.DataFrame:
    """
    Filter wafer maps by size, pad to target aspect ratio,
    then resize to target resolution.
    target_hw format:
        (height, width)
    limits format:
        (w_max, w_min, h_max, h_min)
    """
    target_h, target_w = target_hw
    df = filter_by_grid_size(df, "waferMap", *limits)

    df["waferMap"] = (
        df["waferMap"]
        .apply(lambda grid: pad(grid, h=target_h, w=target_w))
        .apply(lambda grid: scale(grid, (target_w, target_h)))
    )
    return df


def filter_by_grid_size(
    df: pd.DataFrame, grid_col: str, w_max=None, w_min=None, h_max=None, h_min=None
) -> pd.DataFrame:
    """
    Filter rows by grid-col dimensions.
    """
    if grid_col not in df.columns:
        raise DataformatException(
            f"'{grid_col}' column not found in columns: {df.columns.tolist()}"
        )

    mask = df[grid_col].apply(
        lambda grid: is_size(grid, w_max=w_max, w_min=w_min, h_max=h_max, h_min=h_min)
    )
    removed = (~mask).sum()
    info(
        f"Filtered out {removed} rows out of {len(df)} included ({w_min}<w<{w_max}, {h_min}<h<{h_max})"
    )
    return df.loc[mask].copy().reset_index(drop=True)