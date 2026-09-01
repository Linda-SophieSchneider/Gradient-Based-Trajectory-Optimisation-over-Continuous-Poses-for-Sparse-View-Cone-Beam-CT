# FRAUNHOFER IIS CONFIDENTIAL
# __________________
#
# Fraunhofer IIS
# Copyright (c) 2016-2021
# All Rights Reserved.
#
# This file is part of the PythonTools project.
#
# NOTICE:  All information contained herein is, and remains the property of Fraunhofer IIS and its suppliers, if any.
# The intellectual and technical concepts contained herein are proprietary to Fraunhofer IIS and its suppliers and may
# be covered by German and Foreign Patents, patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is strictly forbidden unless prior written
# permission is obtained from Fraunhofer IIS.


from __future__ import annotations

from enum import IntEnum
from math import isclose


class USERLEVEL(IntEnum):
    """User level for UI applications."""
    OPERATOR = 0
    MAINTAINER = 1
    ADMIN = 2


class SEVERITY(IntEnum):
    """Severity classes."""
    INFO = 0
    WARNING = 1
    ERROR = 2


class CTIMAGETYPE(IntEnum):
    """Typical CT image types."""
    DARK = 0
    FLAT = 1
    BPM = 2
    PROJECTION = 3


class IMAGEFORMATS(IntEnum):
    """
    Supported image formats.
    AUTO mode tries to guess the format (if .raw the EZRT format is assumed)
    """
    RAW_BINARY = 0
    RAW_EZRT = 1
    PNG = 2
    TIFF = 3
    BMP = 4
    AUTO = 5  # only for loading


class ACQUISITIONMODE(IntEnum):
    """Acquisition mode enum. Contains options to control acquisition behavior."""
    SINGLE_SHOT = 0
    MULTI_SHOTS = 1
    CONTINUOUS = 2


class TRIGGERMODE(IntEnum):
    """Trigger mode enum. Contains options to control trigger behavior."""
    INTERNAL = 0
    SOFTWARE = 1
    POSITIVE_EDGE = 2
    NEGATIVE_EDGE = 3
    POSITIVE_GATE = 4
    NEGATIVE_GATE = 5


class AXISTYPE(IntEnum):
    """Axis type enum."""
    LINEAR = 0
    LINEAR_UM = 1
    ROTATIONAL = 2
    GONIOMETER = 3


class REFERENCEMODE(IntEnum):
    """Reference mode enum. Contains options for axis reference."""
    MOVE_TO_ZERO = 0
    GOTO_NEGATIVE_LIMIT = 1
    GOTO_POSITIVE_LIMIT = 2
    GOTO_SWITCH_LIMIT = 3
    CUSTOM = 4
    INITIALISE = 5
    SET_POSITION_TO_ZERO = 6
    GOTO_ENCODER_INDEX_IMPULSE = 7


class EZRT_HEADER_DTYPES(IntEnum):
    # integer numbers
    INT8 = 7
    UINT8 = 8
    INT16 = 15
    UINT16 = 16
    INT32 = 31
    UINT32 = 32
    INT64 = 63
    UINT64 = 64
    # floating point numbers
    FLOAT = 24
    DOUBLE = 53


class ACQUISITIONGEOMETRY(IntEnum):
    CONVENTIONAL_3DCT = 0
    SWING_LAMINOGRAPHY = 1
    HELIX_CT = 2
    PARALLEL_CONE_CT = 3
    PARALLEL_CONE_SWING_LAMINOGRAPHY = 4
    STACKED_FANBEAM = 5
    STACKED_FANBEAM_SWING_LAMINOGRAPHY = 6
    OBJ_Z_SHIFT = 100
    ARBITRARY = 0x7fffffff


class CTALGORITHM(IntEnum):
    CYLINDRICAL_FBP = 0
    FBP = 1
    ART = 2
    ART2 = 3
    TOMOSYNTH = 100


class CTFILTER(IntEnum):
    NONE = -1
    SHEPP_LOGAN = 0
    LAMBDA = 1
    RAM_LAK = 2
    SHEPP_LOGAN_MODIFIED = 10
    HAMMING_GENERALIZED = 11
    PARABOLIC = 12
    HANN_VARIABLE_CUTOFF = 13


class CTALGORITHMPLATFORM(IntEnum):
    CPU = 0
    GPUCUDA = 1
    DYNAMIC = 2
    GPUOPENCL = 3
    CPUOPENCL = 4


class CTPROJECTIONPADDING(IntEnum):
    NONE = 0
    CIRCLE = 1
    COSINE = 2
    SIMPLE_MULTISCAN = 100


class Range:
    """
    A class to represent a value range in the form minimum <= value <= maximum.
    (Can be used with any Python object implementing __le__ & __ge__)
    """

    def __init__(self, minimum: int | float, maximum: int | float):
        """
        Construct new Range with given limits (int / float supported).

        :param minimum: minimum value to be in range
        :param maximum: maximum value to be in range

        :raise ValueError: raised if minimum bigger than maximum or maximum is smaller than minimum
        """
        self.set_range(minimum, maximum)

    def __eq__(self, compared_range):
        return isclose(self.minimum, compared_range.minimum, abs_tol=1e-6) and\
            isclose(self.maximum, compared_range.maximum, abs_tol=1e-6)

    def __iter__(self):
        """
        Iterator for whole range (including maximum value).
        For floating point numbers, the hardcoded step size is 0.1
        """
        step = 0.1
        if type(self.minimum) is int and type(self.maximum) is int:
            step = 1

        current_value = self.minimum
        while current_value <= self.maximum or isclose(current_value, self.maximum, abs_tol=1e-3):
            yield current_value
            current_value += step

    @staticmethod
    def value_between_limits(value: int | float, minimum: int | float, maximum: int | float) -> bool:
        """
        Checks if value is within or equal to limits.

        :param value: value to be checked against range
        :param minimum: minimum value to be in range
        :param maximum: maximum value to be in range
        """
        if minimum > maximum:
            raise ValueError('minimum must be smaller than maximum or equal')

        return minimum < value < maximum or isclose(value, minimum, abs_tol=1e-6) or\
            isclose(value, maximum, abs_tol=1e-6)

    @property
    def minimum(self) -> int | float:
        """
        Minimum value to be in range. When using this property the range is not checked for consistency
        so use set_range if possible.
        """
        return self._minimum

    @minimum.setter
    def minimum(self, value: int | float):
        self._minimum = value

    @property
    def maximum(self) -> int | float:
        """
        Maximum value to be in range. When using this property the range is not checked for consistency
        so use set_range if possible.
        """
        return self._maximum

    @maximum.setter
    def maximum(self, value: int | float):
        self._maximum = value

    def set_range(self, minimum: int | float, maximum: int | float):
        """
        Sets the valid range to check against.

        :param minimum: minimum value to be in range
        :param maximum: maximum value to be in range
        """
        if minimum > maximum:
            raise ValueError('minimum must be smaller than maximum or equal')

        self._minimum = minimum
        self._maximum = maximum

    def in_range(self, value: int | float) -> bool:
        """
        Checks if given value is in range.

        :param value: value to be checked against range
        :return: True if in range
        """
        return self.value_between_limits(value, self._minimum, self._maximum)

    @property
    def range(self) -> tuple[int | float, int | float]:
        """
        Returns the current range as tuple.

        :return: tuple (min, max)
        """
        return self._minimum, self._maximum


class ROI:
    """
    A class to represent ROI (region of interests; sometimes called AOIs). Mostly used for detector image size handling.
    """

    def __init__(self, x: Range = Range(0, 1), y: Range = Range(0, 1)):
        """
        Construct new ROI from Ranges (default: 0->1 / 0->1)

        :param x: range of values in x
        :param y: range of values in y
        """
        self._x = x
        self._y = y

    def __str__(self):
        return f'x: {self._x.minimum} -> {self._x.maximum}; y: {self._y.minimum} -> {self._y.maximum}'

    def __eq__(self, compared_roi):
        return self.x == compared_roi.x and self.y == compared_roi.y

    @classmethod
    def from_values(cls, x0: int | float, x1: int | float, y0: int | float, y1: int | float):
        """
        Construct new ROI from single values (int / float supported).

        :param x0: start of x range
        :param x1: end of x range
        :param y0: start of y range
        :param y1: end of y range
        """
        return cls(x=Range(x0, x1), y=Range(y0, y1))

    @property
    def values(self) -> tuple[int | float, int | float, int | float, int | float]:
        """ROI values: x_min, x_max, y_min, y_max"""
        return self._x.minimum, self._x.maximum, self._y.minimum, self._y.maximum

    @property
    def x(self) -> Range:
        """x range of ROI (e.g. range of rows)"""
        return self._x

    @x.setter
    def x(self, x: Range):
        if not isinstance(x, Range):
            raise ValueError('x argument must be of type Range')
        self._x = x

    @property
    def y(self) -> Range:
        """y range of ROI (e.g. range of columns)"""
        return self._y

    @y.setter
    def y(self, y: Range):
        if not isinstance(y, Range):
            raise ValueError('y argument must be of type Range')
        self._y = y

    def shape(self, numpy_order: bool = False) -> tuple[int | float, int | float]:
        """
        Returns shape of ROI (x size (row), y size (column)). Optionally in numpy compliant order
        (y size (column), x size (row)).

        :param numpy_order: Flag to enable numpy compliant return order for images
        """
        if numpy_order:
            return self.y.maximum - self.y.minimum, self.x.maximum - self.x.minimum

        return self.x.maximum - self.x.minimum, self.y.maximum - self.y.minimum
