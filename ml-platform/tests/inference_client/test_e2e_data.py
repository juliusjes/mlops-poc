import time

import numpy as np
import pandas as pd
import pytest
import requests
from ml_platform.shared.data_processing.augmentation_funcs import f, blur
from ml_platform.shared.services.lakefs_service import LakeFSService

TEST_REPOSITORY = "pytest"


@pytest.fixture
def inf_url():
    inf_cli_url = "http://localhost:8013"
    r = requests.get(url = f"{inf_cli_url}/health")
    r.raise_for_status()
    return inf_cli_url


@pytest.fixture
def lkfs():
    service = LakeFSService()
    if service.get_repo(TEST_REPOSITORY) is None:
        service.create_repo(TEST_REPOSITORY)
    return service


def assert_dataframe_equal_ordered(expected, actual):
    """
    Compare only X and y.

    Row order must be identical.
    Each row matches when both X and y are equal.
    """
    assert len(expected) == len(actual), (
        f"Different number of rows: expected {len(expected)}, "
        f"got {len(actual)}"
    )

    assert list(expected[["X", "y"]].columns) == ["X", "y"]

    for i, (expected_row, actual_row) in enumerate(
        zip(expected[["X", "y"]].itertuples(index=False),
            actual[["X", "y"]].itertuples(index=False))
    ):
        np.testing.assert_array_equal( actual_row.X, expected_row.X, err_msg=f"X differs at row {i}", )

        assert actual_row.y == expected_row.y, (
            f"y differs at row {i}: "
            f"expected {expected_row.y}, got {actual_row.y}"
        )


def assert_dataframe_equal_unordered(expected, actual, tol=1e-6):
    """
    Compare X and y columns, ignoring row order.

    Rows are matched using X and y with numerical tolerance.

    Produces a match table when matching fails.
    """
    assert len(expected) == len(actual), (
        f"Different number of rows: expected {len(expected)}, "
        f"got {len(actual)}"
    )

    unmatched = list(actual.index)
    matches = []

    for expected_i, expected_row in enumerate(
        expected[["X", "y"]].itertuples(index=False)
    ):
        best_match = None
        best_distance = float("inf")

        for actual_i in unmatched:
            actual_row = actual.loc[actual_i]

            x_equal = np.allclose(actual_row["X"], expected_row.X, rtol=tol, atol=tol)

            y_equal = np.isclose(actual_row["y"], expected_row.y, rtol=tol, atol=tol)

            x_diff = np.max( np.abs( np.asarray(actual_row["X"]) - np.asarray(expected_row.X) ) )

            y_diff = abs(actual_row["y"] - expected_row.y)

            distance = x_diff# + y_diff

            if distance < best_distance:
                best_distance = distance
                best_match = {
                    "actual_index": actual_i,
                    "x_equal": x_equal,
                    "y_equal": y_equal,
                    "x_max_diff": x_diff,
                    "y_diff": y_diff,
                    "actual_X": actual_row["X"],
                    "expected_X": expected_row.X,
                    "actual_y": actual_row["y"],
                    "expected_y": expected_row.y,

                }

            if x_equal and y_equal:
                break

        if best_match and best_match["x_equal"] and best_match["y_equal"]:
            actual_i = best_match["actual_index"]
            unmatched.remove(actual_i)

            matches.append({
                "expected_index": expected_i,
                #"actual_index": actual_i,
                "actual_x": best_match["actual_X"],
                "expected_x": best_match["expected_X"],
                "status": "MATCH",
                "x_equal": True,
                "y_equal": True,
                "x_max_diff": best_match["x_max_diff"],
                "y_diff": best_match["y_diff"],
            })

        else:
            matches.append({
                "expected_index": expected_i,
                "actual_index": (
                    best_match["actual_index"]
                    if best_match else None
                ),
                "status": "NO MATCH",
                "x_equal": (
                    best_match["x_equal"]
                    if best_match else False
                ),
                "y_equal": (
                    best_match["y_equal"]
                    if best_match else False
                ),
                "x_max_diff": (
                    best_match["x_max_diff"]
                    if best_match else None
                ),
                "y_diff": (
                    best_match["y_diff"]
                    if best_match else None
                ),
                "actual_x": best_match["actual_X"],
                "expected_x": best_match["expected_X"],
                "actual_y": best_match["actual_y"],
                "expected_y": best_match["expected_y"],
            })

    match_table = pd.DataFrame(matches)

    if not match_table["status"].eq("MATCH").all() or unmatched:
        print("\nDataFrame match diagnostics:")
        print(match_table.to_string(index=False))

        if unmatched:
            print("\nUnmatched actual rows:")
            print(actual.loc[unmatched, ["X", "y"]].to_string())

        raise AssertionError(
            "DataFrames are not equivalent. "
            "See match diagnostics above."
        )


class TestE2EData:

    def test_inference_server_nodrift(self, lkfs, inf_url):

        # make some test data
        n = 15
        X = [np.full((2,), i) for i in range(n)]
        y = [i for i in range(n)]

        df = pd.DataFrame({"X": X, "y": y})

        reponame = TEST_REPOSITORY
        branchname = time.strftime("%Y%m%d_%H%M%S_nd")
        branch = lkfs.create_branch(reponame, branchname)

        location = reponame + "/" + branchname + "/" + "data"

        location_out = lkfs.df_to_lakeFS_parquet(df, location, ser_col=["X"])

        lkfs.make_commit(branch, "pytest autocommit", {})

        assert location == location_out[:-1]

        df_out = lkfs.lakeFS_parquet_to_df(location, deser_col=["X"])

        assert_dataframe_equal_ordered(df, df_out)

        data = {
            "drift_func_name": "DriftFunc",
            "drift_func_kwargs": {}
        }

        r = requests.post(url=f"{inf_url}/set_drift_func", json=data)
        r.raise_for_status()


        data = {
            "location": f"{location_out}",
            "dataset_description": {
                "target_column": ["y", "int"],
                "sample_column": ["X", "bytes"],
                "sample_shape": [2],
                "target_shape": [1],
                "order_column": ["write_timestamp", "datetime"],
                "target_interp": "classification"
            }
        }

        r = requests.post(url = f"{inf_url}/set_data_source", json=data)
        r.raise_for_status()
        assert r.json()["found"], "Inference client could not load the LakeFS data source"

        data = {
            "new_location": f"{location_out}" + "from_inf",
            "size": n,
            "start": 0
        }

        r = requests.post(url=f"{inf_url}/generate_drifted_dataset", json=data)
        r.raise_for_status()

        location_d_out = r.json()["new_location"]

        
        df_out = lkfs.lakeFS_parquet_to_df(location_d_out, deser_col=["X"])

        # check find matching rows. 

        assert_dataframe_equal_unordered(df, df_out)


    def test_inference_server_drift_f(self, lkfs, inf_url):

        # make some test data
        n = 20
        X = [np.ones((2,)) for _ in range(n)]
        y = [i for i in range(n)]
        for i, yy in enumerate(y):
            X[i][1] = X[i][1] + yy

        df = pd.DataFrame({"X": X, "y": y})

        reponame = TEST_REPOSITORY
        branchname = time.strftime("%Y%m%d_%H%M%S_d")
        branch = lkfs.create_branch(reponame, branchname)

        location = reponame + "/" + branchname + "/" + "data"

        location_out = lkfs.df_to_lakeFS_parquet(df, location, ser_col=["X"])

        lkfs.make_commit(branch, "pytest autocommit", {})

        assert location == location_out[:-1]

        df_out = lkfs.lakeFS_parquet_to_df(location, deser_col=["X"])

        assert_dataframe_equal_ordered(df, df_out)

        data = {
            "drift_func_name": "CPNDriftFunc",
            "drift_func_kwargs": {
                "augmentation_func": ["f"],
                "augmentation_kwargs": [
                    {
                        "constant": [1]
                    }
                ],
                "parameter_names": ["offset"],
                "param_values": [0.0, 1.0],
                "param_cps": [0.0, 1.0],
                "class_ratio": 1.0,
                "replace_all_labels": [False]
            }
        }

        r = requests.post(url=f"{inf_url}/set_drift_func", json=data)
        r.raise_for_status()


        data = {
            "location": f"{location_out}",
            "dataset_description": {
                "target_column": ["y", "int"],
                "sample_column": ["X", "bytes"],
                "sample_shape": [2],
                "target_shape": [1],
                "order_column": ["write_timestamp", "datetime"],
                "target_interp": "classification"
            }
        }

        r = requests.post(url = f"{inf_url}/set_data_source", json=data)
        r.raise_for_status()
        assert r.json()["found"], "Inference client could not load the LakeFS data source"

        data = {
            "new_location": f"{location_out}" + "from_inf",
            "size": n,
            "start": 0
        }

        r = requests.post(url=f"{inf_url}/generate_drifted_dataset", json=data)
        r.raise_for_status()

        location_d_out = r.json()["new_location"]

        df_out = lkfs.lakeFS_parquet_to_df(location_d_out, deser_col=["X"])

        # check find matching rows.

        offsets = np.linspace(0, 1, len(df), endpoint=False)

        df["X"] = [
                f(x, offset=offset)
                for x, offset in zip(df["X"], offsets)
            ]

        assert_dataframe_equal_unordered(df, df_out)


    def test_inference_server_drift_blur(self, lkfs, inf_url):

        # make some test data
        n = 11
        X = [np.ones((5,5)) for _ in range(n)]
        y = [i for i in range(n)]
        for i, yy in enumerate(y):
            X[i][1, 1] = X[i][1,1] + yy

        df = pd.DataFrame({"X": X, "y": y})

        reponame = TEST_REPOSITORY
        branchname = time.strftime("%Y%m%d_%H%M%S_blur")
        branch = lkfs.create_branch(reponame, branchname)

        location = reponame + "/" + branchname + "/" + "data"

        location_out = lkfs.df_to_lakeFS_parquet(df, location, ser_col=["X"])

        lkfs.make_commit(branch, "pytest autocommit", {})

        assert location == location_out[:-1]

        df_out = lkfs.lakeFS_parquet_to_df(location, deser_col=["X"])

        assert_dataframe_equal_ordered(df, df_out)

        data = {
            "drift_func_name": "CPNDriftFunc",
            "drift_func_kwargs": {
                "augmentation_func": ["blur"],
                "augmentation_kwargs": [
                    {}
                ],
                "parameter_names": ["sigma"],
                "param_values": [0.7, 1.2],
                "param_cps": [0.0, 1.0],
                "class_ratio": 1.0,
                "replace_all_labels": [False]
            }
        }

        r = requests.post(url=f"{inf_url}/set_drift_func", json=data)
        r.raise_for_status()

        data = {
            "location": f"{location_out}",
            "dataset_description": {
                "target_column": ["y", "int"],
                "sample_column": ["X", "bytes"],
                "sample_shape": [1,5,5],
                "target_shape": [1],
                "order_column": ["write_timestamp", "datetime"],
                "target_interp": "classification"
            }
        }

        r = requests.post(url = f"{inf_url}/set_data_source", json=data)
        r.raise_for_status()
        assert r.json()["found"], "Inference client could not load the LakeFS data source"

        data = {
            "new_location": f"{location_out}" + "from_inf",
            "size": n,
            "start": 0
        }

        r = requests.post(url=f"{inf_url}/generate_drifted_dataset", json=data)
        r.raise_for_status()

        location_d_out = r.json()["new_location"]
        
        df_out = lkfs.lakeFS_parquet_to_df(location_d_out, deser_col=["X"])

        # check find matching rows.

        sigmas = np.linspace(0.7, 1.2, n, endpoint=False)

        df["X"] = [ blur(x, sigma=s) for x, s in zip(df["X"], sigmas) ]

        assert_dataframe_equal_unordered(df, df_out)
