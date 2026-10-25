import base64
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm

from ml_platform.shared.exceptions import DataformatException
from ml_platform.shared.paths import DATA_FILE_PATH, DOWNLOAD_SCRIPT_PATH
from ml_platform.shared.utils import array_to_bytes, bytes_to_array, info, warning


def ensure_data_file(
        data_file=DATA_FILE_PATH, download_file=DOWNLOAD_SCRIPT_PATH
    ) -> Path:
    """Return the canonical LSWMD data file, downloading it when absent.

    ``None`` is accepted for data-pipeline configurations so the canonical
    repository paths are selected instead of a machine-specific source path.
    """
    if data_file is None:
        data_file = DATA_FILE_PATH
    elif not isinstance(data_file, Path):
        data_file = Path(data_file)
        if not data_file.parent.is_dir():
            data_file = DATA_FILE_PATH
            
    if not isinstance(download_file, Path):
        download_file = Path(download_file)

    if not data_file.is_file():
        if not download_file.is_file():
            raise FileNotFoundError(f"Download script not found: {download_file}")

        data_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["bash", str(download_file), str(data_file.parent)],
            check=True,
        )

        if not data_file.is_file():
            raise FileNotFoundError(
                f"Dataset was not found after download: {data_file}"
            )

    return data_file


def numpy_to_string(x) -> str:
    try:
        return x[0][0]
    except Exception:
        return pd.NA


def LSWMD_to_df(file: Path) -> pd.DataFrame:
    sys.modules["pandas.indexes"] = pd.core.indexes

    with file.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def lakeFS_parquet_to_df(location, deser_col=None) -> pd.DataFrame:
    from ml_platform.shared.services.lakefs_service import LakeFSService
    return LakeFSService().lakeFS_parquet_to_df(location, deser_col)



def df_to_lakeFS_parquet( df, location, chunk_size=10_000, disable=False, ser_col=None) -> str:
    from ml_platform.shared.services.lakefs_service import LakeFSService
    return LakeFSService().df_to_lakeFS_parquet(df, location, chunk_size, disable, ser_col)



def LSWMD_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the LSWMD dataframe for downstream processing.

    - Extract failure_label from failureType
    - Extract split_label from trianTestLabel
    - Replace "none" failure labels with NA
    - Drop the original array-backed columns
    - Convert textual columns to pandas string dtype

    Returns:
        A cleaned dataframe
    """

    required_columns = {"failureType", "trianTestLabel"}

    missing = required_columns - set(df.columns)
    if missing:
        raise DataformatException(f"Missing required columns: {missing}")

    df = df.copy()

    # Extract labels
    df["failure_label"] = df["failureType"].apply(numpy_to_string)
    df["split_label"] = df["trianTestLabel"].apply(numpy_to_string)

    # Normalize missing values
    df["failure_label"] = df["failure_label"].mask(df["failure_label"] == "none", pd.NA)

    # Remove original nested-array columns
    df = df.drop(columns=["failureType", "trianTestLabel"])
    # Normalize string columns
    string_columns = ["lotName", "failure_label", "split_label"]

    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def filter_by_failure_type(
    df: pd.DataFrame, failure_types: str | list[str] | None = None
) -> pd.DataFrame:
    """
    Filter dataframe rows by failure_type from failure_label column.
    """
    if not isinstance(failure_types, (str, list, type(None))):
        raise TypeError(
            f"Invalid type for failure_types: {type(failure_types)} "
            "(expected str, list, or None)"
        )

    if failure_types is None:
        warning("No filtering performed, failure_type was None")
        return df

    if isinstance(failure_types, str):
        failure_types = [failure_types]

    if "failure_label" not in df.columns:
        raise DataformatException(
            f"'failure_label' not found in columns: {df.columns.tolist()}"
        )

    available_failures = set(df["failure_label"].unique())
    requested_failures = set(failure_types)

    missing = requested_failures - available_failures
    if missing:
        warning(
            f"Missing failure_types: {sorted(missing)} "
            f"(available: {sorted(available_failures)})"
        )

    mask = df["failure_label"].isin(failure_types)
    removed = (~mask).sum()
    info(f"Filtered out {removed} rows out of {len(df)} (included {failure_types})")
    return df.loc[mask]

def _draw_scratch(row, seed, verbose, **scratch_kwargs):
    from ml_platform.shared.data_processing.augmentation_funcs import draw_scratch

    row_rng = default_rng(SeedSequence([seed, int(row.name)]))

    return draw_scratch(
        row["waferMap"], # TODO: fixed column name
        rng=row_rng,
        verbose=verbose,
        **scratch_kwargs,
    )

def add_scratches_to_rows( df: pd.DataFrame, row_mask=None, seed: int = 0, verbose: bool = True, workers=None, **scratch_kwargs, ):
    required_columns = {"waferMap", "failure_label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise DataformatException(f"Missing columns: {sorted(missing)}")

    df = df.copy()

    if row_mask is None:
        row_mask = np.ones(len(df), dtype=bool)

    df.loc[row_mask, "waferMap"] = parallel_apply(
        df.loc[row_mask],
        _draw_scratch,
        workers=workers,
        args=(seed, verbose),
        **scratch_kwargs,
    )

    df.loc[row_mask, "failure_label"] = "Scratch"

    if verbose:
        info(f"Added Bezier scratches to {row_mask.sum()} rows out of {len(df)}")

    return df

def map_to_int( df: pd.DataFrame, target_column: str, mapping: dict[str, int], default_value: int) -> pd.DataFrame:
    df[target_column] = df[target_column].map(
        lambda x: mapping.get(x, default_value)
    )
    return df

def index_list_to_mask(df, subsets, ratio=0.5):
    union_idx = pd.Index([])
    for idx in subsets:
        n = int(len(idx) * ratio)
        union_idx = union_idx.union(idx[:n])
    return df.index.isin(union_idx)


def overlap_matrix(indices: list[pd.Index]) -> np.ndarray:
    """Compute a matrix of pairwise overlaps between pandas Index objects."""
    n = len(indices)

    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            matrix[i, j] = len(indices[i].intersection(indices[j]))

    return matrix


def LSWMD_experiment_datasets_to_lakeFS(
    df, N, seeds, l1, l2, class_ratio, location, scratch_kwargs, mapping_kwargs
):

    assert isinstance(seeds, list) and len(seeds) > 0, (
        "Seeds must be a list with atleast 1 element"
    )

    if len(seeds) > N:
        warning(f"Received {len(seeds)} seeds, truncating to {N}", tf=1)
        seeds = seeds[:N]

    if len(seeds) < N or len(set(seeds)) < len(seeds):  # check for uniqueness
        warning(
            f"Invalid seed list: {len(seeds)} seeds, {len(set(seeds))} unique", tf=1
        )
        seeds = list(range(seeds[0], seeds[0] + N))
        info(f"New seeding now ranges from {seeds[0]} to {seeds[-1]}", tf=1)


    # Keep the source pool deterministic, but do not modify it after this point.
    df = df.sample(frac=1, random_state=seeds[0])

    splits = N_splits(df, N, l1, l2, seeds=seeds)

    subsets0 = [s[0] for s in splits]  # offline indices
    subsets1 = [s[1] for s in splits]  # online indices

    # Full coverage masks: these are only for diagnostics, not augmentation.
    offline_union_mask = index_list_to_mask(df, subsets0, ratio=1.0)
    online_union_mask = index_list_to_mask(df, subsets1, ratio=1.0)

    overlap_subsets0 = overlap_matrix(subsets0)
    overlap_subsets1 = overlap_matrix(subsets1)

    # Fraction of the source dataframe occurring in at least one subset.
    coverage_fraction = ( offline_union_mask | online_union_mask ).sum() / len(df)

    info(f"Subsets cover {float(coverage_fraction) * 100:.2f} % of the superset", tf=1)

    cross_role_overlap_fraction = (
        offline_union_mask & online_union_mask
    ).sum() / len(df)

    info(
        f"Offline/online cross-seed overlap is "
        f"{float(cross_role_overlap_fraction) * 100:.2f} % of the superset",
        tf=1,
    )

    if not 0.0 <= class_ratio <= 1.0:
        raise ValueError( f"class_ratio must be between 0 and 1, got {class_ratio}" )

    for i, seed in tqdm(
        enumerate(seeds),
        total=len(seeds),
        desc=info(return_string=True),
    ):
        off_loc = location + f"{seed}/offline/"
        on_loc = location + f"{seed}/online/"

        off_idx = subsets0[i]
        on_idx = subsets1[i]

        offline_df = df.loc[off_idx].copy()
        online_df = df.loc[on_idx].copy()

        n_scratch = int(len(offline_df) * class_ratio)

        scratch_mask = np.zeros(len(offline_df), dtype=bool)

        if n_scratch > 0:
            rng = np.random.default_rng(seed)
            scratch_positions = rng.choice(
                len(offline_df),
                size=n_scratch,
                replace=False,
            )
            scratch_mask[scratch_positions] = True

            offline_df = add_scratches_to_rows( offline_df, row_mask=scratch_mask, seed=seed, **scratch_kwargs, )

        offline_df = map_to_int( offline_df, **mapping_kwargs, )
        online_df = map_to_int( online_df, **mapping_kwargs, )

        shuffled_off_df = ( offline_df .sample(frac=1, random_state=0) .reset_index(drop=True) )
        shuffled_on_df = ( online_df .sample(frac=1, random_state=1) .reset_index(drop=True) )


        assert len(off_idx.intersection(on_idx)) == 0, (
            f"Seed {seed}: offline and online indices overlap"
        )

        assert len(offline_df) == len(off_idx), (
            f"Seed {seed}: offline row count changed unexpectedly"
        )

        assert len(online_df) == len(on_idx), (
            f"Seed {seed}: online row count changed unexpectedly"
        )

        df_to_lakeFS_parquet(
            shuffled_off_df,
            off_loc,
            disable=True,
            ser_col=["waferMap"],
        )

        df_to_lakeFS_parquet(
            shuffled_on_df,
            on_loc,
            disable=True,
            ser_col=["waferMap"],
        )

    return location



def N_splits(df: pd.DataFrame, N: int, l1: int, l2: int, seeds: list) -> list[tuple]:
    assert isinstance(seeds, list)
    assert N == len(seeds)

    idx_pairs = []
    for seed in seeds:
        idx = nonoverlapping_subset_index(df, l1, l2, seed=seed)
        idx_pairs.append(idx)

    return idx_pairs


def nonoverlapping_subset_index(
        df: pd.DataFrame, l1: int = 1000, l2: int = 1000, seed: int = 2026
    ) -> tuple[pd.Index, pd.Index]:
    """Subset into two non-overlapping subsets and return their indices."""
    l_tot = l1 + l2

    if l_tot > len(df):
        warning(f"Requested subset sizes ({l1}, {l2}) are are larger than available dataset" 
                f"({len(df)}), defaulting to using all available samples with same ratio")
        
        target = len(df)
        scale = target / (l1 + l2)
        l1 = int(l1 * scale)
        l2 = target - l1
        l_tot = l1 + l2


    selected = df.sample(n=l_tot, random_state=seed).index
    i1, i2 = selected[:l1], selected[l1:]

    assert i1.intersection(i2).empty
    return i1, i2


def metadata_to_lakeFS(arg, metadata=None):
    from ml_platform.shared.services.lakefs_service import LakeFSService
    metadata = {} if not metadata else metadata
    LakeFSService().metadata_to_lakeFS(metadata, location=arg)


def generate_simple_data(N, dim=8, n_classes=2):
    """
    Generate N vectors of dimension `dim` from n_classes Gaussian clusters.
    """
    n_per_class = N // n_classes
    X = []
    y = []
    for c in range(n_classes):
        mean = np.ones(dim) * ((c + 1) * 5)
        samples = np.random.normal( loc=mean, scale=1.0, size=(n_per_class, dim), )

        for sample in samples:
            X.append(sample)
            y.append(c)

    return pd.DataFrame( { "X": X, "y": y, } )


def simple_experiment_datasets_to_lakeFS(df, location, N=1,):
    seeds = list(range(N))
    splits = N_splits(df, N, l1=500, l2=500, seeds=seeds)  # list of tuples
    subsets0 = [s[0] for s in splits]  # offline Index
    subsets1 = [s[1] for s in splits]  # online Index

    for i in range(N):
        off_loc = location + f"{i}/offline/"
        on_loc = location + f"{i}/online/"
        off_idx = subsets0[i]
        on_idx = subsets1[i]

        df_to_lakeFS_parquet(
            df.loc[off_idx],
            off_loc,
            disable=False,
            ser_col=["X"],
        )
        df_to_lakeFS_parquet(
            df.loc[on_idx],
            on_loc,
            disable=False,
            ser_col=["X"],
        )


def encode_for_json(x):
    """
    Encode numpy array for jsons
    """
    return (
        base64.b64encode(array_to_bytes(x)).decode("utf-8")
        if isinstance(x, np.ndarray)
        else x
    )


def decode_from_json(x):
    """
    decode numpy array from json utf-8
    """
    if isinstance(x, str):
        return bytes_to_array(base64.b64decode(x))
    return x



def _apply_chunk(args):
    chunk, func, axis, apply_args, apply_kwargs = args
    return chunk.apply(
        func,
        axis=axis,
        args=apply_args,
        **apply_kwargs,
    )

def parallel_apply(df, func, axis=1, workers=None, args=(), **kwargs):
    """
    Parallel version of pandas.DataFrame.apply().
    """
    from multiprocess import Pool, cpu_count
    workers = workers or cpu_count()

    chunks = np.array_split(df, workers)

    with Pool(workers) as pool:
        results = pool.map(
            _apply_chunk,
            [
                (chunk, func, axis, args, kwargs)
                for chunk in chunks
            ],
        )

    return pd.concat(results)

if __name__ == "__main__":
    pass
    # df = pd.DataFrame({"a": np.random.uniform(size=(10000)), "b": np.random.uniform(size=(10000))})

    # def score(row, multiplier, offset=0):
    #     return (row["a"] + row["b"]) * multiplier + offset

    
    # result = parallel_apply(
    #     df,
    #     score,
    #     args=(2,),       # multiplier
    #     offset=10,        # keyword argument
    # )
    

    # import matplotlib.pyplot as plt

    # def make_disk_mask(h=128, w=128, radius=None):
    #     if radius is None:
    #         radius = min(h, w) // 2 - 4

    #     yy, xx = np.ogrid[:h, :w]
    #     cy, cx = h // 2, w // 2

    #     mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

    #     grid = np.zeros((h, w), dtype=np.float32)
    #     grid[mask] = 1.0

    #     return grid

    # grid = make_disk_mask(128, 128)

    # scratched = draw_scratch(
    #     grid.copy(),
    #     n_scratches_range=(1),
    #     length_range=(0.5, 0.5),
    #     curvature_range=(0.0, 0.5),
    #     angle_range=("normal", 1, 0.05),
    #     circularity_range=(0.0, 1.0),
    #     width_range=(0.01, 0.05),
    #     degree=5,
    # )

    # plt.figure(figsize=(5, 5))
    # plt.imshow(scratched)
    # plt.title("Scratch Test")
    # plt.axis("off")
    # plt.show()
