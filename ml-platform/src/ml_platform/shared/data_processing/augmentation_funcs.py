from math import comb
from typing import Callable

import cv2
import numpy as np
from skimage.draw import disk

from ml_platform.shared.exceptions import ParameterException
from ml_platform.shared.utils import info, warning


def bezier_curve(points, n=100):
    """
    Vectorized Bezier curve of arbitrary degree.
    points:
        shape (degree + 1, dims)
    returns:
        shape (n, dims)
    """
    points = np.asarray(points, dtype=float)
    degree = len(points) - 1
    t = np.linspace(0, 1, n)
    i = np.arange(degree + 1)

    # Bernstein basis
    bernstein = (
        np.array([comb(degree, k) for k in i])
        * (1 - t[:, None]) ** (degree - i)
        * t[:, None] ** i
    )
    return bernstein @ points, points


def interpret_parameter_range(param_spec, rng) -> Callable:
    """
    Return a sampling function from parameter specification.
    Supported formats: 5 -> Constant value,
    (low, high) ->
        Uniform distribution,
    ("normal", mean, std) ->
        Normal distribution
    """
    if isinstance(param_spec, (int, float)):
        return lambda: param_spec

    # Uniform distribution
    if (
        isinstance(param_spec, (tuple, list))
        and len(param_spec) == 2
        and all(isinstance(x, (int, float)) for x in param_spec)
    ):
        low, high = sorted(param_spec)
        return lambda: rng.uniform(low, high)

    # Named distributions
    if (
        isinstance(param_spec, (tuple, list))
        and len(param_spec) >= 2
        and isinstance(param_spec[0], str)
    ):
        dist_name = param_spec[0]

        if (
            dist_name == "normal"
            and len(param_spec) == 3
            and all(isinstance(x, (int, float)) for x in param_spec[1:])
            and param_spec[2] > 0
        ):
            mean, std = param_spec[1:]
            return lambda: rng.normal(mean, std)

        raise ParameterException(f"Unknown or invalid distribution: {param_spec}")

    raise ParameterException(f"Invalid parameter specification: {param_spec}")


def draw_scratch(
    grid: np.ndarray,
    n_scratches_range: tuple[int] = (1, 2),
    length_range: tuple = (0.5, 0.5),
    width_range: tuple = (0.03, 0.03),
    curvature_range: tuple = (0.0, 0.0),
    angle_range: tuple = (0, 2 * np.pi),
    circularity_range: tuple = (0, 1),
    degree: int = 2,
    verbose: bool = True,
    rng=np.random.default_rng(0),
):
    """
    Draw random Bezier scratches on a 2D or 3D numpy array. if the drawn scratch does not affect atleast 25% of the lenght a new curve is drawn up to 5 times.
    When the grid is a 3D array, the first dimension is used.

    Parameters
    ----------
    n_scratches_range :
        Sampler/range for number of scratches.

    length_range :
        Relative scratch length w.r.t. image height.

    width_range :
        Relative scratch width w.r.t. image height.

    curvature_range :
        Relative perpendicular Bezier offset w.r.t. image height.

    angle_range :
        Angular sampling range for scratch direction.

    circularity_range :
        Determines curvature side.
        < 0.5 -> left
        >= 0.5 -> right

    degree :
        Bezier degree.
    """
    if degree < 1:
        raise ParameterException(f"Bezier degree ({degree}) must be >= 1")

    shape = grid.shape
    assert len(shape) <= 3, f"Grid should have no more than 3 dimensions, has {shape}"
    if len(shape) == 3:
        grid = grid[0, :, :]

    h, w = grid.shape

    new_grid = grid.copy()
    wafer_mask = (grid == 1) | (grid == 2)

    if not np.any(wafer_mask):
        warning("No scratches applied")
        return new_grid

    sample_n_scratches = interpret_parameter_range(n_scratches_range, rng)
    sample_length = interpret_parameter_range(length_range, rng)
    sample_width = interpret_parameter_range(width_range, rng)
    sample_curvature = interpret_parameter_range(curvature_range, rng)
    sample_angle = interpret_parameter_range(angle_range, rng)
    sample_circularity = interpret_parameter_range(circularity_range, rng)

    MAX_TRIES = 5
    n = 0

    while n < MAX_TRIES:
        ys, xs = np.where(wafer_mask)
        n_scratches = max(1, int(sample_n_scratches()))

        for _ in range(n_scratches):
            idx = rng.integers(len(xs))

            p0 = np.array([xs[idx], ys[idx]], dtype=float)

            angle = sample_angle()
            direction = np.array([np.cos(angle), np.sin(angle)])
            perp = np.array([-direction[1], direction[0]])

            length = sample_length() * h
            p_end = p0 + direction * length

            control_points = [p0]

            for i in range(1, degree):
                alpha = i / degree
                base = p0 + direction * (length * alpha)

                sign = -1 if sample_circularity() < 0.5 else 1
                curvature = sample_curvature() * h * sign

                control_points.append(base + perp * curvature)

            control_points.append(p_end)

            curve, _ = bezier_curve(control_points, n=max(50, 2 * h))
            scratch_radius = max(1.0, sample_width() * h / 2.0)

            for x, y in curve:
                xi = int(round(x))
                yi = int(round(y))

                if not (0 <= xi < w and 0 <= yi < h):
                    continue

                rr, cc = disk((yi, xi), scratch_radius, shape=grid.shape)
                mask = wafer_mask[rr, cc]

                new_grid[rr[mask], cc[mask]] = 3

        pixels_affected = np.count_nonzero(new_grid == 3)
        if pixels_affected < length * 2 *scratch_radius // 2:
            if verbose:
                warning(
                    f"{n}: Under 25% of the simulated scratch length is visible ({pixels_affected // length:.1f}%)"
                )
            n += 1
            continue
        else:
            new_grid[new_grid == 3] = 2  # return to defect
            return new_grid.reshape(shape)

    if verbose and n == MAX_TRIES:
        warning(
            f"Tried {n} times without simulated scratch being atleast 25% visible, returning ({pixels_affected // length:.1f}%) coverage"
        )
        # TODO: return false when fail

    new_grid[new_grid == 3] = 2  # return to defect
    return new_grid.reshape(shape)


def f(X, offset=0, constant=1):
    return constant*X + offset*5



def blur(grid, sigma):
    k = int(np.ceil(6 * sigma))
    k += (k + 1) % 2

    grid = grid.astype(np.float32)
    print("Grid has ndim:", grid.ndim)
    if grid.ndim < 2:
        raise ValueError("grid must have at least 2 dimensions")

    if grid.ndim == 2:
        return cv2.GaussianBlur(
            grid, (k, k), sigma, sigma, cv2.BORDER_REPLICATE
        )

    out = np.empty_like(grid)

    for idx in np.ndindex(grid.shape[:-2]):
        out[idx] = cv2.GaussianBlur(
            grid[idx],
            (k, k),
            sigma,
            sigma,
            cv2.BORDER_REPLICATE
        )

    return out


def scratch_and_blur(grid, sigma, **kwargs):
    kwargs["curvature_range"] = 1.5*sigma-1.2 # gimmick
    grid = draw_scratch(grid, **kwargs)
    grid = blur(grid, sigma)
    return grid


if __name__ == "__main__":

    grid = np.ones((256,256))
    import time
    t0 = time.time()
    draw_scratch(grid, verbose=True)
    print(time.time() - t0)