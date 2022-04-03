# -*- coding: utf-8 -*-

"""interval.py

Helper classes from https://github.com/eth-sri/3dcertify

"""

from typing import Iterable, Union

import numpy as np

PI_HALF = np.pi / 2.0
TWO_PI = 2.0 * np.pi


class Interval:
    def __init__(self, lower_bound, upper_bound):
        assert np.all(
            lower_bound <= upper_bound
        ), "lower bound has to be smaller or equal to upper bound"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __neg__(self):
        return Interval(-self.upper_bound, -self.lower_bound)

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(
                self.lower_bound + other.lower_bound,
                self.upper_bound + other.upper_bound,
            )
        else:
            return Interval(self.lower_bound + other, self.upper_bound + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(
                self.lower_bound - other.upper_bound,
                self.upper_bound - other.lower_bound,
            )
        else:
            return Interval(self.lower_bound - other, self.upper_bound - other)

    def __rsub__(self, other):
        return Interval(other - self.upper_bound, other - self.lower_bound)

    def __mul__(self, other):
        if isinstance(other, Interval):
            return Interval(
                np.min(
                    [
                        self.lower_bound * other.lower_bound,
                        self.lower_bound * other.upper_bound,
                        self.upper_bound * other.lower_bound,
                        self.upper_bound * other.upper_bound,
                    ],
                    axis=0,
                ),
                np.max(
                    [
                        self.lower_bound * other.lower_bound,
                        self.lower_bound * other.upper_bound,
                        self.upper_bound * other.lower_bound,
                        self.upper_bound * other.upper_bound,
                    ],
                    axis=0,
                ),
            )
        else:
            return Interval(
                np.minimum(self.lower_bound * other, self.upper_bound * other),
                np.maximum(self.lower_bound * other, self.upper_bound * other),
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if isinstance(other, Interval):
            return (
                self.lower_bound == other.lower_bound
                and self.upper_bound == other.upper_bound
            )
        return False

    def __repr__(self):
        return f"Interval(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"

    def center(self) -> Union[float, np.ndarray]:
        return (self.lower_bound + self.upper_bound) / 2

    def __getitem__(self, item):
        return Interval(
            self.lower_bound.__getitem__(item),
            self.upper_bound.__getitem__(item),
        )

    def __len__(self):
        return self.lower_bound.__len__()

    # Make sure __radd__ etc work correctly with numpy ndarray
    __numpy_ufunc__ = None  # Numpy up to 13.0
    __array_ufunc__ = None  # Numpy 13.0 and above


def square(x: Union[Interval, float, np.ndarray]):
    if isinstance(x, Interval):
        # Default case with positive and negative ranges
        lower_bound = np.zeros_like(x.lower_bound)
        upper_bound = np.maximum(
            np.square(x.lower_bound), np.square(x.upper_bound)
        )
        # Special case where all values are positive
        lower_bound = np.where(
            x.lower_bound >= 0, np.square(x.lower_bound), lower_bound
        )
        upper_bound = np.where(
            x.lower_bound >= 0, np.square(x.upper_bound), upper_bound
        )
        # Special case where all values are negative
        lower_bound = np.where(
            x.upper_bound <= 0, np.square(x.upper_bound), lower_bound
        )
        upper_bound = np.where(
            x.upper_bound <= 0, np.square(x.lower_bound), upper_bound
        )
        return Interval(lower_bound, upper_bound)
    else:
        return np.square(x)


def cos(theta: Union[Interval, float, np.ndarray]):
    return sin(theta + PI_HALF)


def sin(theta: Union[Interval, float, np.ndarray]):
    if isinstance(theta, Interval):
        offset = np.floor(theta.lower_bound / TWO_PI) * TWO_PI
        theta_lower = theta.lower_bound - offset
        theta_upper = theta.upper_bound - offset
        lower = np.minimum(np.sin(theta_lower), np.sin(theta_upper))
        upper = np.maximum(np.sin(theta_lower), np.sin(theta_upper))
        lower = np.where(
            np.logical_and(
                theta_lower <= 3 * PI_HALF, 3 * PI_HALF <= theta_upper
            ),
            -1,
            lower,
        )
        upper = np.where(
            np.logical_and(theta_lower <= PI_HALF, PI_HALF <= theta_upper),
            1,
            upper,
        )
        lower = np.where(
            np.logical_and(
                theta_lower <= 7 * PI_HALF, 7 * PI_HALF <= theta_upper
            ),
            -1,
            lower,
        )
        upper = np.where(
            np.logical_and(
                theta_lower <= 5 * PI_HALF, 5 * PI_HALF <= theta_upper
            ),
            1,
            upper,
        )
        return Interval(lower, upper)
    else:
        return np.sin(theta)


def stack(
    elements: Iterable[Union[Interval, np.ndarray]], axis: int, convert=False
):
    if convert or all([isinstance(it, Interval) for it in elements]):
        return Interval(
            lower_bound=np.stack(
                [as_interval(element).lower_bound for element in elements],
                axis,
            ),
            upper_bound=np.stack(
                [as_interval(element).upper_bound for element in elements],
                axis,
            ),
        )
    else:
        return np.stack(elements, axis)


def as_interval(element: Union[np.ndarray, float, Interval]) -> Interval:
    if isinstance(element, Interval):
        return element
    else:
        return Interval(element, element)
