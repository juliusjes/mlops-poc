import numpy as np
import pandas as pd
import pytest
import cv2 as cv

from ml_platform.inference_client.drifts import CPNDriftFunc



def test_simple_augment_result():

    test_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    expected_grid = cv.GaussianBlur(
        test_grid,
        ksize=(7, 7),
        sigmaX=1
    )


    X = pd.Series([test_grid.copy() for _ in range(10)])
    y = pd.Series(np.ones(10))


    d = CPNDriftFunc(
            seed=0,
            class_ratio=1.0,
            augmentation_func="blur",
            augmentation_kwargs={},
            parameter_names="sigma",
            target_replacement=1,
            replace_all_labels=False,
            param_values=[1, 1],
            param_cps=[0.5, 0.6],
        )

    X_out, y_out, param = d._transform(X, y)

    assert param[0].sum() == 10
    assert y_out.sum() == 10

    for i in range(10):
        augmented_grid = X_out.iloc[i]
        np.testing.assert_array_equal(augmented_grid, expected_grid)





@pytest.mark.parametrize(
    "class_ratio, replace_all_labels, expected_label_changes, expected_augmented",
    [
        # Existing target samples are sufficient.
        # 50 target samples already exist, N = 50.
        (0.5, True, 0, 50),

        # No label replacement, existing target samples are sufficient.
        (0.5, False, 0, 50),

        # Need 75 target samples, but only 50 exist.
        # 25 non-target samples are selected.
        # With replacement enabled, those 25 become target labels.
        (0.75, True, 25, 75),

        # Same selection, but labels remain unchanged.
        # This simulates concept drift.
        (0.75, False, 0, 75),
    ],
)
def test_cpn_drift( class_ratio, replace_all_labels, expected_label_changes, expected_augmented, ):
    def augmentation(x, parameter):
        assert x.shape == (50, 50)

        x = x.copy()
        x[10, 10] = parameter
        return x

    X = pd.Series(
        [np.ones((50, 50)) for _ in range(50)]
        + [np.zeros((50, 50)) for _ in range(50)]
    )

    y = pd.Series([1] * 50 + [0] * 50)

    d = CPNDriftFunc(
        seed=0,
        class_ratio=class_ratio,
        augmentation_func=augmentation,
        augmentation_kwargs={},
        parameter_names="parameter",
        target_replacement=1,
        replace_all_labels=replace_all_labels,
        param_values=[2,3],
        param_cps=[0.3, 0.7],
    )

    X_out, y_out, params = d._transform(X, y)

    # Verify labels
    label_changes = (y_out != y).sum()
    assert label_changes == expected_label_changes

    # Verify number of augmented samples
    augmented_samples = sum(
        x[10, 10] != original[10, 10]
        for x, original in zip(X_out, X)
    )

    assert augmented_samples == expected_augmented

    correctly_augmented_samples = 0

    for idx in X.index:

        original_value = X.loc[idx][10, 10]
        transformed_value = X_out.loc[idx][10, 10]
        expected_value = params[0][idx]

        is_augmented = transformed_value == expected_value
        is_unchanged = transformed_value == original_value

        assert is_augmented or is_unchanged

        if is_augmented:
            correctly_augmented_samples += 1

    assert correctly_augmented_samples == expected_augmented