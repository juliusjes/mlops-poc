from warnings import warn

import cv2 as cv
import numpy as np
import pandas as pd
import pytest
from ml_platform.shared.data_processing.augmentation_funcs import blur, draw_scratch, f


class ScratchVisibilityWarning(UserWarning):
    pass


def test_f_valid():
    x = np.ones((2,))
    constant = 1
    offset = 1
    expected = x*constant + 5*offset
    np.testing.assert_array_equal(f(x, constant=constant, offset=offset), expected)


def test_blur_valid():
    test_grid = np.ones((100, 100), dtype=np.float32)
    expected = cv.GaussianBlur(test_grid, (7, 7), 1, 1)

    # test 2D grid
    np.testing.assert_array_equal(blur(test_grid, sigma=1), expected)

    # test 3D grid
    np.testing.assert_array_equal(blur(test_grid.reshape(1,100,100), sigma=1), expected.reshape(1,100,100))



@pytest.mark.parametrize("seed", [
    (seed,) for seed in range(20)
    ])
def test_draw_scratch(seed):

    test_grid = np.ones((100, 100))

    grid_out = draw_scratch(
            test_grid,
            n_scratches_range=(1, 1),
            length_range = (0.5, 0.5),
            width_range = (0.03, 0.03),
            curvature_range= (0.0, 0.0),
            angle_range = (0, 0),
            circularity_range = (0, 1),
            degree = 4,
            verbose = True,
            rng=np.random.default_rng(seed),
        )

    width = 0.03 * 100 # disk diameter
    length = 0.5 * 100 # line length

    endpoints = np.ceil(np.pi * (width/2)**2) # 2 disk endpoints that go beyond line start and end
    n_expected_placements = width * length + endpoints

    n_affected_out = np.count_nonzero(grid_out == 2)
    
    tolerance = 0.01 * n_expected_placements 

    lpc = int(n_expected_placements - tolerance)
    upc = int(n_expected_placements + tolerance)

    within_tolerance = lpc <= n_affected_out <= upc

    if not within_tolerance:
        min_visible = int(0.5 * n_expected_placements) - endpoints

        assert n_affected_out >= min_visible, (
            f"Only {n_affected_out} pixels affected, below 50% "
            f"of expected {n_expected_placements} ({min_visible})"
        )

        warn(
            f"[test_draw_scratch]: seed={seed}, "
            f"affected pixels={n_affected_out} outside expected range "
            f"[{lpc}, {upc}], but visibility="
            f"{100 * n_affected_out / n_expected_placements:.2f}% > 50%",
            ScratchVisibilityWarning
        )

@pytest.mark.parametrize("seed", [
    (seed,) for seed in range(5)
    ])
def test_draw_scratch_determinisim(seed):
    
    test_grid1 = np.ones((100, 100))

    grid_out1 = draw_scratch(
            test_grid1,
            n_scratches_range=(1, 2),
            length_range = (0.5, 0.5),
            width_range = (0.03, 0.03),
            curvature_range= (0.0, 0.5),
            angle_range = (0, 2),
            circularity_range = (0, 1),
            degree = 4,
            verbose = True,
            rng=np.random.default_rng(seed),
        )

    test_grid2 = np.ones((100, 100))
    
    grid_out2 = draw_scratch(
            test_grid1,
            n_scratches_range=(1, 2),
            length_range = (0.5, 0.5),
            width_range = (0.03, 0.03),
            curvature_range= (0.0, 0.5),
            angle_range = (0, 2),
            circularity_range = (0, 1),
            degree = 4,
            verbose = True,
            rng=np.random.default_rng(seed),
        )

    assert np.array_equal(grid_out1, grid_out2)