import hashlib
import time

import numpy as np
import pandas as pd
import torch
from ml_platform.shared.services.lakefs_service import LakeFSService
from ml_platform.workers.trainer.plugin_api.data_source import DataSource
from ml_platform.workers.trainer.training_config import (
    ColumnTypes,
    DatasetDescription,
    TargetInterpret,
)
from torch.utils.data import IterableDataset


class ArrowIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        columns=None,
        batch_size=1024,
        prep=None,
        prep_cols=None,
        split=None,
        num_splits=None,
        seed=42,
        max_samples=-1
    ):
        self.dataset = dataset
        self.columns = columns
        self.batch_size = batch_size
        self.prep = prep
        self.prep_cols = prep_cols or []

        self.split = split
        self.num_splits = num_splits
        self.seed = seed
        self.max_samples = max_samples

        if split is not None:
            if num_splits is None:
                raise ValueError("num_splits must be provided when split is specified")
            if num_splits < 2:
                raise ValueError("num_splits must be >= 2")

        self._split_index = self._get_split_index()

    def _get_split_index(self):
        if self.split is None:
            return None
        value = int( hashlib.md5(f"{self.seed}:{self.split}".encode()).hexdigest(), 16, )
        return value % self.num_splits

    def _in_split(self, row_index):
        if self._split_index is None:
            return True
        value = int( hashlib.md5(f"{self.seed}:{row_index}".encode()).hexdigest(), 16, )
        return value % self.num_splits == self._split_index

    def __iter__(self):
        scanner = self.dataset.scanner(
            columns=self.columns,
            batch_size=self.batch_size,
        )

        count = 0
        row_index = 0

        for batch in scanner.to_batches():
            data = batch.to_pydict()

            for c in self.prep_cols:
                data[c] = [torch.from_numpy(self.prep(x)).to(torch.float) for x in data[c]]

            n = batch.num_rows

            for i in range(n):
                current_index = row_index
                row_index += 1

                if not self._in_split(current_index):
                    continue

                if self.max_samples > 0 and count >= self.max_samples:
                    return

                count += 1
                yield {key: value[i] for key, value in data.items()}


class LakeFSSource(DataSource):
    def __init__(self, location, dataset_desciption: DatasetDescription):
        self.lakefs_service = LakeFSService(verbose=True)

        self.dsd = dataset_desciption
        self.dataset, self.decoder = self.lakefs_service.get_arrow_dataset(
            location, self.dsd
        )

    def prepare(self, x):
        return x if self.decoder is None else self.decoder(x)

    def get_torch_dataset(self, split:str, num_splits:int=3, seed:int=0, max_samples=-1) -> IterableDataset:
        """
        Returns an iterable torch dataset. The split parameter defines the current split to use.
        The number of splits should be known before to ensure no information leakage between splits.
        If the underlying dataset is reorderd the same seed won't produce the same dataset.
        """
        return ArrowIterableDataset(
            self.dataset,
            [self.dsd.sample_column[0], self.dsd.target_column[0]],
            prep=self.prepare,
            prep_cols=[self.dsd.sample_column[0]],
            split=split,
            num_splits=num_splits,
            seed=seed,
            max_samples=max_samples
        )

    def get_list_dataset(self) -> list:
        raise NotImplementedError

    def get_numpy_dataset(self) -> np.ndarray:
        raise NotImplementedError

    def get_pandas_dataset(self) -> pd.DataFrame:
        df: pd.DataFrame = self.dataset.to_table(
            columns=[self.dsd.sample_column[0], self.dsd.target_column[0]]
        ).to_pandas()
        df[self.dsd.sample_column[0]] = df[self.dsd.sample_column[0]].apply(
            self.prepare
        )
        return df


if __name__ == "__main__":
    desc = DatasetDescription(
        sample_column=["waferMap", ColumnTypes.BYTES],
        sample_shape=[1,128,128],
        target_column=["failure_label", ColumnTypes.INT],
        target_shape=[1],
        order_column=["write_timestamp", ColumnTypes.DATETIME],
        target_interp=TargetInterpret.CLASSIFICATION,
    )

    location = "masters-thesis/expedc681c7d6c3d260/expdata/0/offline/"

    source = LakeFSSource(location, desc)

    ids = source.get_torch_dataset(split="train", num_splits=3)

    from torch.utils.data import DataLoader

    loader = DataLoader(ids, batch_size=64)

    for batch in loader:

        print(batch["waferMap"].shape)


    

    
