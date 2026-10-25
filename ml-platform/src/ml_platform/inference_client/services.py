import asyncio
import importlib
import json
import logging
import os

import httpx
import numpy as np
import pandas as pd
from ml_platform.inference_client import drifts as drift_module
from ml_platform.inference_client.config import DatasetDescription, Settings
from ml_platform.shared.data_processing.data_processing import (
    bytes_to_array,
    decode_from_json,
    encode_for_json,
    parallel_apply,
)
from ml_platform.shared.services.lakefs_service import LakeFSService

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


class InferenceClient:
    def __init__(self, settings: Settings):

        self.inference_endpoint = settings.inference_endpoint
        self.send_path = settings.send_path
        self.timeout = settings.inference_server_timeout
        self.max_timeouts = settings.max_timeouts
        self.label_delay = settings.label_delay # in samples
        self.current_sample_index = settings.current_sample_index
        self.last_end_index = self.current_sample_index - self.label_delay

        assert settings.sample_column != settings.target_column
        self.target_column_name = settings.target_column
        self.sample_column_name = settings.sample_column
        self.ss = tuple(settings.sample_shape)
        
        self.async_client = httpx.AsyncClient()

        self.drift_func = None
        self.set_drift_func(settings.drift_func_name, settings.drift_func_kwargs)
        self.data_source = None
        self.dsd = None
        self.raw_df:pd.DataFrame = None
        self.drifted_df = None
        self.drift = None
        self.predictions = None

        self.access_key = settings.s3_access_key
        self.secret_key = settings.s3_secret_key
        self.endpoint_override = settings.s3_endpoint_override

        success = self.set_data_source(settings.data_source, settings.dsd)

        # if success:
        logger.info(f"Inference endpoint: {self.inference_endpoint}/{self.send_path}")
        logger.info(f"Data source: {self.data_source}")
        # else:
        #     logger.error("Failed to launch")
            


    def set_drift_func(self, drift_func_name, kwargs):
        try:
            drift_cls = getattr(drift_module, drift_func_name)
            self.drift_func = drift_cls(**kwargs)
            logger.info(f"Drift function set to {drift_func_name}")
            return True
        
        except Exception as e:
            logger.error(f"Unable to set drift function: {e}")

            try:
                drift_cls = getattr(drift_module, drift_func_name)
                self.drift_func = drift_cls(**kwargs)
                logger.info(f"Drift function set to {drift_func_name}")
                return True
            
            except Exception as e:
                logger.error(f"Unable to set drift function: {e}")
        
        return False
        

    def set_data_source(self, location: str|None, dsd: DatasetDescription) -> bool:
        self.data_source = location
        self.dsd = dsd
        logger.info(f"Data source set to: {location}")
        if not self.data_source:
            return False
        
        if not self._get_lakefs_data():
            return False
        
        if not self.initialize():
            return False

        self.current_sample_index = 0
        
        logger.info("Data drifted and initialized")
        
        if self.dsd:
            if self.dsd.target_interp == "classification":
                target_values = self.drifted_df["y"]
                class_counts = target_values.value_counts()
                logger.info("Target class distribution:\n%s", class_counts)
            else:
                logger.info(f"target_interp = {self.dsd.target_interp}")

        return True


    def _get_lakefs_data(self):
        try:
            service = LakeFSService()

            dataset, decoder = service.get_arrow_dataset(self.data_source, self.dsd)
            self.raw_df = dataset.to_table().to_pandas()
        except Exception as e:
            logger.error(f"Unable to get Lakefs dataset: {e!s}")
            return False

        self.raw_df = self.raw_df.rename(columns={self.dsd.target_column[0]: "y", self.dsd.sample_column[0]: "X"})
        logger.info(f"Dataset loaded with {len(self.raw_df)} samples from {self.data_source}")
        if self.dsd.sample_column[1] == "bytes":
            self.raw_df["X"] = [decoder(x) for x in self.raw_df["X"]]

        # TODO: add target column decoder

        return True

    def generate_drifted_validation_dataset(
            self, new_location: str, size: int, start: int | None = None, allow_resample: bool = True
        ) -> bool:
        
        if self.drifted_df is None:
            logger.info(
                "No data source set for inference client, "
                "unable to generate drifted validation dataset"
            )
            return False

        if size <= 0:
            raise ValueError(f"size must be greater than 0, got {size}")

        try:
            repo, branchname = new_location.split("/", 2)[0], new_location.split("/", 2)[1]
        except ValueError:
            raise ValueError(
                f"Invalid LakeFS location '{new_location}'. "
                "Expected format '<repo>/<branch>'."
            )

        service = LakeFSService()

        start = ( max(0, start) if start is not None else self.current_sample_index )
        end = min(len(self.drifted_df), start + size)

        if start >= len(self.drifted_df):
            if not allow_resample:
                raise ValueError(
                    f"No data available from start index {start}."
                )
            raise ValueError(
                f"Start index {start} is outside the dataset "
                f"(length={len(self.drifted_df)})."
            )

        df = self.drifted_df.iloc[start:end].copy()

        original_size = len(df)
        resampled = False

        if original_size < size:
            if not allow_resample:
                raise ValueError( f"Only {original_size} rows are available from index {start}, but {size} were requested." )

            if original_size == 0:
                raise ValueError(f"No data available from start index {start}.")

            df = df.sample(n=size, replace=True, random_state=None)
            resampled = True

        # Rename columns to match the original dataset schema.
        df = df.rename(
            columns={
                "X": self.dsd.sample_column[0],
                "y": self.dsd.target_column[0],
            }
        )

        logger.info(
            "Generating drifted validation dataset: "
            "start=%d, end=%d, source_rows=%d, output_rows=%d, resampled=%s, "
            "columns=%s",
            start,
            end,
            original_size,
            len(df),
            resampled,
            list(df.columns),
        )

        service.df_to_lakeFS_parquet(
            df,
            new_location,
            ser_col=[self.dsd.sample_column[0]],
        )

        branch = service.get_branch(repo, branchname)

        service.make_commit( branch, "Generated validation dataset", metadata={"size": str(len(df))})
            #     "start_index": str(start),
            #     "end_index": str(end),
            #     "requested_size": str(size),
            #     "source_size": str(original_size),
            #     "output_size": str(len(df)),
            #     "resampled": str(resampled),
            # },
        # )

        logger.info(
            "Generated drifted validation dataset from %d to %d " "(%d rows written)",
            start, end, len(df)
            )

        return True
    
    # def _decoding(self):
    #     self.raw_df["X"] = [bytes_to_array(x) for x in self.raw_df["X"]]

    def _try_drift_func(self) -> bool:
        sample = self.raw_df.sample(n=10).copy()
        X_sample = sample["X"]
        y_sample = sample["y"]
        try:
            self.drift_func(X_sample, y_sample, test=True)
        except Exception as e:
            logger.error(f"The configured drift function '{self.drift_func}' failed the call:\n\t{e}")
            return False
        
        return True

    def _transform(self):
        X, y, self.drift = self.drift_func(self.raw_df["X"], self.raw_df["y"])

        self.drifted_df = pd.DataFrame({"X": X, "y": y})
        #self.drifted_df["X"] = [np.reshape(x, self.dsd.sample_shape) for x in self.drifted_df["X"]] # TODO: gimmick
        logger.info(f"Drift function applied to {len(self.drifted_df)} samples")


    def _shuffle(self):
        self.raw_df = self.raw_df.sample(frac=1, random_state=0).reset_index(drop=True)

    def initialize(self):
        #self._shuffle() # TODO: this makes the augmentatiotion parametere index fail for some reason

        if not self.drift_func:
            return False

        if not self._try_drift_func():
            return False

        self._transform()
        self.predictions = []

        return True


    async def _is_inference_service_available(self) -> bool:
        try:
            response = await self.async_client.get(f"{self.inference_endpoint}/state", timeout=30.0)
        except:
            return False
        return response.status_code == 200
    

    async def _sample_to_inference(self, sample, uuid, y=None):
        if self.inference_endpoint is None:
            raise ValueError("inference_endpoint is not set")
        
        if not await self._is_inference_service_available():
            raise TimeoutError("Inference service not available")
        data = {"X": sample, "y": y, "deployment_uuid": str(uuid)}
        response = await self.async_client.post( f"{self.inference_endpoint}/{self.send_path}", json=data, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        if "y_pred" not in data:
            raise ValueError("Response missing 'y_pred'")

        self.current_sample_index += 1
        y_pred = decode_from_json(data["y_pred"])
        return y_pred


    async def send_n_samples(self, n, uuid, label=True):
        # TODO: fix behavioer with no model in inference server, should not advance internal data counter
        if self.drifted_df is None:
            logger.info("No data source set for inference client, unable to send samples")
            return []

        timeouts = 0
        samples_sent = 0
        for _ in range(n):
            idx = self.current_sample_index

            if idx >= len(self.drifted_df):
                logger.warning(f"out of samples with index {idx}")
                return []

            if timeouts >= self.max_timeouts:
                logger.warning("maximum number fo timeouts reached")
                break

            sample = self.drifted_df["X"].iloc[idx]
            sample = encode_for_json(sample)
            y = float(self.drifted_df["y"].iloc[idx]) if label else None # TODO: make sure this works with label delay, harder than it seems actually

            try:
                async with asyncio.timeout(self.timeout):
                    y_pred = await self._sample_to_inference(sample,uuid, y=y)

                    self.predictions.extend(y_pred)
                    samples_sent += 1
                    
            except TimeoutError:
                logger.warning(f"Timed out for sample {idx}")
                self.predictions.append(None)
                timeouts += 1

            except ValueError as e:
                logger.warning(e)

            except httpx.HTTPStatusError as e:
                logger.warning(f"Prediction request caused HTTPStatusError: {e}")

            except Exception as e:
                logger.error(f"Unknown error {str(e)}")

        logger.info(f"{samples_sent} samples sent, timeouts: {timeouts}, sample idx: {idx}")

        # if samples_sent == 0:
        #     return []
            
        return self.predictions[-max(samples_sent, 1):]

    def is_out_of_samples(self):
        if self.drifted_df is None:
            return True, self.current_sample_index, -1
        return len(self.drifted_df) <= self.current_sample_index, self.current_sample_index, len(self.drifted_df)


dir_path = os.path.dirname(os.path.realpath(__file__))
settings_path = os.path.join(dir_path, "settings.json")

if os.path.exists(settings_path):
    logger.info("Using settings.json")
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = Settings.model_validate_json(f.read())
else:
    settings = Settings()

client = InferenceClient(
    settings
)