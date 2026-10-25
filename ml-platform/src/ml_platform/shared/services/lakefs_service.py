import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import dotenv
import lakefs
import lakefs_sdk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lakefs.client import Client
from lakefs_spec import LakeFSFileSystem
from tqdm import tqdm

from ml_platform.shared.data_processing.data_processing import (
    array_to_bytes,
    bytes_to_array,
)
from ml_platform.shared.paths import ENV_FILE
from ml_platform.shared.utils import get_healthy_endpoint, info

#from ml_platform.workers.trainer.training_config import DatasetDescription

pa_map = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
    "bytes": pa.binary(),
    "date": pa.date32(),
    "datetime": pa.timestamp("us"),
}


class LakeFSService:
    """
    Singleton service for communicating with LakeFS
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, verbose=False):
        self.verbose = verbose
        if self._initialized:
            return

        self._initialized = True

        self.verbose = verbose
        if self.verbose:
            print(f"Using ENV file: {ENV_FILE}")

        dotenv.load_dotenv(ENV_FILE)

        username = os.environ["CREDENTIALS_ACCESS_KEY_ID"]
        password = os.environ["CREDENTIALS_SECRET_ACCESS_KEY"]
        host = get_healthy_endpoint(
            [
                os.environ["SERVER_ENDPOINT_URL"],
                "http://localhost:8000",
                os.environ["SERVER_ENDPOINT_URL"].replace("lakefs", "ml-platform-lakefs-1"),
            ],
            health_path="/healthcheck",
            timeout=0.5
        )
        if self.verbose:
            print(f"LAKEFS using host: {host}")

        lakeFS_cfg = {
            "host": host,
            "username": username,
            "password": password,
        }
        self.fs: LakeFSFileSystem = LakeFSFileSystem(**lakeFS_cfg)
        self.client: Client = Client(**lakeFS_cfg)

        S3_kwargs = {
            "access_key": username,
            "secret_key": password,
            "endpoint_override": host,
        }
        self.pa_fs = pa.fs.S3FileSystem(**S3_kwargs, scheme="https")


    def is_path_lakefs(self, location) -> bool:
        """Return whether ``location`` is an existing LakeFS object path.

        The platform configuration convention is ``<repository>/<branch>``
        with an optional object path following the branch.  This intentionally
        does not accept local filesystem paths or ``lakefs://`` URIs: callers
        pass the raw path understood by ``LakeFSFileSystem``.

        Invalid syntax is rejected without contacting LakeFS.  A syntactically
        valid path is then checked through LakeFS so that this method retains
        its existing use as an existence check during dataset generation.
        """
        if not isinstance(location, str):
            return False

        path = location.strip()
        if (
            not path
            or path.startswith(("/", "lakefs://"))
            or "//" in path
        ):
            return False

        parts = path.strip("/").split("/")
        if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
            return False

        try:
            result = self.fs.ls(location)
            return len(result) > 0
        except FileNotFoundError:
            return False


    def get_repo(self, reponame)-> Any|None:
        repo = lakefs.repository(reponame.strip("/"), client=self.client)

        try:
            repo._client.sdk_client.repositories_api.get_repository(repo.id)
            return repo
        except lakefs_sdk.exceptions.NotFoundException:
            return 

        
    def create_repo(self, reponame):
        try:
            repo = lakefs.repository(reponame, client=self.client).create(
                storage_namespace=f"s3://lakefs-bucket/{reponame}/"
            )
        except lakefs.exceptions.ConflictException:
            return self.get_repo(reponame)
        return repo

    def create_branch(self, reponame, branchname, source_reference="main"):
        repo = self.get_repo(reponame)
        if repo is None:
            return
        
        try:
            branch = repo.branch(branchname.strip("/")).create(
                source_reference=source_reference,
                exist_ok=False
            )   
        except lakefs.exceptions.ConflictException:
            return self.get_branch(reponame, branchname)

        return branch

    def get_branch(self, reponame:str, branchname:str):
        repo = self.get_repo(reponame)
        if not repo:
            return
        branch = repo.branch(branchname.strip("/"))
        return branch

    def make_commit(self, branch, msg, metadata):
        if self.verbose:
            print(branch, msg, metadata)
        branch.commit(message=msg, metadata=metadata)

    def get_arrow_dataset(
        self, location: str, dataset_description
    ) -> tuple[ds.Dataset, Callable | None]:
        # if not self.is_path_lakefs(location):
        #     raise Exception(f"Invalid lakefs dataset location: {location}")

        sc, sct = dataset_description.sample_column
        tc, tct = dataset_description.target_column
        oc, oct = dataset_description.order_column
        target_interp = dataset_description.target_interp
        ss = tuple(dataset_description.sample_shape)
        ts = tuple(dataset_description.target_shape)

        schema = [
            (sc, pa_map[sct]),
            (tc, pa_map[tct]),
        ]

        if oc is not None:
            schema.append((oc, pa_map[oct]))

        schema = pa.schema(schema)

        dataset = ds.dataset(
            location,
            format="parquet",
            filesystem=self.pa_fs,
            schema=schema,
        )

        dataset = dataset.sort_by(oc)

        if sct == "bytes" or tct == "bytes":
            decoder = lambda x: np.reshape(bytes_to_array(x), ss)
        else:
            decoder = None

        return dataset, decoder

    def get_head_commit(self, location):
        """
        get commit id of dataset location where dataset location is <repo>/<branch>/...
        """
        if not self.is_path_lakefs(location):
            raise Exception(f"Invalid lakefs dataset location: {location}")

        parts = location.strip("/").split("/")
        repo = parts[0]
        branch = parts[1]
        return lakefs.repository(repo, client=self.client).branch(branch).head.id


    def df_to_lakeFS_parquet(
        self,
        df,
        location: str,
        chunk_size=10_000,
        disable=False,
        ser_col=None,
    ) -> str:
        """
        Write a dataframe to location as a .parquet file. Serialize any columns in ser_col to bytes.
        location is <repo>/branch/...
        """
        dataset_timestamp = datetime.now(timezone.utc)

        if ser_col is None:
            ser_col = []
        if not location.endswith("/"):
            location = location + "/"
        for i, start in tqdm(
            enumerate(range(0, len(df), chunk_size)),
            disable=disable,
            desc=info(return_string=True),
        ):
            chunk = df.iloc[start : start + chunk_size].copy()
            chunk["write_timestamp"] = dataset_timestamp

            for sc in ser_col:
                if sc in chunk.columns:
                    chunk[sc] = [ array_to_bytes(x) for x in chunk[sc] ]

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            chunk_location = f"{location}part-{i:05d}.parquet"

            pq.write_table(
                table,
                chunk_location,
                filesystem=self.pa_fs,
            )

        return location

    def lakeFS_parquet_to_df(self, location, deser_col=None) -> pd.DataFrame:
        """
        Read a Parquet dataset from lakeFS using PyArrow. Desearailize byte columns
        """
        if not self.is_path_lakefs(location):
            return
        if deser_col is None:
            deser_col = []

        dataset = pa.dataset.dataset(
            location,
            filesystem=self.pa_fs,
            format="parquet",
        )

        df = dataset.to_table().to_pandas()
        for col in deser_col:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: bytes_to_array(x) if x is not None else None
                )

        return df

    def metadata_to_lakeFS(self, metadata: dict, location: str):
        if location[-1] == "/":
            location = location + "metadata.json"
        with self.pa_fs.open_output_stream(location) as f:
            f.write(json.dumps(metadata, indent=2).encode("utf-8"))



if __name__ == "__main__":
    pass
