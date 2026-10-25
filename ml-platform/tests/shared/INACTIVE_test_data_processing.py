import dotenv
import numpy as np
import pandas as pd
import pytest
import os

from data_processing.data_processing import (
    array_to_bytes,
    bytes_to_array,
    decode_from_json,
    df_to_lakeFS_parquet,
    encode_for_json,
    filter_by_grid_size,
    generate_simple_data,
    lakeFS_parquet_to_df,
    nonoverlapping_subset_index,
    pad,
    scale
)
from ml.training_config import Sampling
from orchestrator.services.lakefs_service import LakeFSService
from paths import ENV_FILE
import requests




def service_is_up(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


class TestDataProcessing:

    dotenv.load_dotenv(ENV_FILE)
    lakefs_health_url = os.environ["SERVER_ENDPOINT_URL"] + "/health"

    def test_encode_decode(self):
        x = np.random.uniform(size=(100,100))
        decoded = decode_from_json(encode_for_json(x))
        assert (x == decoded).all()


    def test_array_bytes(self):
        x = np.random.uniform(size=(100,100))
        ret = array_to_bytes(x)
        assert isinstance(ret, bytes)
        assert len(ret) <= x.nbytes + 512 # arbitaray header size
        y = bytes_to_array(ret)
        assert (x==y).all()
        

    def test_non_overlapping_subset_index(self):
        df = pd.DataFrame({"X": np.arange(1000)})
        ret = nonoverlapping_subset_index(df, l1=500, l2=500, seed=0)

        assert len(ret) == 2 and isinstance(ret, tuple)
        i1, i2 = ret

        assert isinstance(i1, pd.Index) and isinstance(i2, pd.Index)
        assert set(i1.to_list()).isdisjoint(i2.to_list())
        assert len(i1) == 500


    def test_filter_by_size(self):
        df = pd.DataFrame({"X": [np.zeros((i,i)) for i in range(1, 101)]})
        df = filter_by_grid_size(df, "X", w_max=50, h_max=50)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    
    def test_pad(self):
        x = np.zeros((10, 20))
        ret = pad(x)
        assert isinstance(ret, np.ndarray)
        assert ret.shape[0] == 20 and ret.shape[1] == 20
        assert np.count_nonzero(ret) == 0
        with pytest.raises(ValueError):
            pad(x, h=0)


    def test_scale(self):
        x = np.zeros((10, 10))
        ret = scale(x, target_size=(100, 100))
        assert isinstance(ret, np.ndarray)
        assert ret.shape[0] == 100, ret.shape[1] == 100
        assert np.count_nonzero(ret) == 0

        ret1 = scale(x)
        np.testing.assert_array_equal(ret1, x)


    #------------------
    # end to end tests
    #------------------
    @pytest.mark.asyncio
    async def test_lakefs_dataset(self):
        if not service_is_up(self.lakefs_health_url):
            pytest.skip("lakefs is unavailable")

        location = "s3://pytest/main/test_lakefs_dataset"
        df1 = generate_simple_data(100)

        df_to_lakeFS_parquet(df1, location=location, ser_col=["X"])

        ds = await LakeFSService().get_torch_dataset(
            dataset_location=location[5:],
            n_samples=-1,
            sampling=Sampling.UNIFORM,
            sampling_seed=0,
        )
        df2 = ds.df

        df1 = df1.rename(columns={"y": "y_true"})

        # Compare all non-array columns
        pd.testing.assert_frame_equal(
            df1.drop(columns=["X"]),
            df2.drop(columns=["X"]),
            check_dtype=True,
            atol=1e-6,
            rtol=1e-6,
        )

        # Compare array column
        assert len(df1) == len(df2)

        for a, b in zip(df1["X"], df2["X"]):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


    def test_lakefs_dataframe_in_out(self):
        if not service_is_up(self.lakefs_health_url):
            pytest.skip("lakefs is unavailable")

        df1 = pd.DataFrame({
            "a": [float(x) for x in np.random.uniform(size=(10))],
            "b": [int(x) for x in np.random.uniform(size=(10))]
            })
        
        location = "s3://pytest/main/test_lakefs_dataframe_in_out"

        # in
        df_to_lakeFS_parquet(df1, location)

        # out
        df2 = lakeFS_parquet_to_df(location)
        df2 = df2.drop("write_timestamp", axis=1)

        pd.testing.assert_frame_equal(df1, df2)

