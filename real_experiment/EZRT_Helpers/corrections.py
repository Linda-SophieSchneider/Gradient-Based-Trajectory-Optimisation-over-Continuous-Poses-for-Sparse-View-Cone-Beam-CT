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

from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    pass
try:
    from ctmt_algorithms.ct.corrections import flat_correction, beam_hardening_correction
except ModuleNotFoundError:
    pass

from PythonTools.ezrt_header import EzrtHeader

from .raw2py import raw2py
from .py2raw import py2raw


# CORRECTIONS BY PATH

def dark_and_flat_correct_by_path(image_path: Path | str, dark_image_path: Path | str, flat_image_path: Path | str,
                                  output_path: Path | str, maximum: int = 65535, mean: int | None = None,
                                  use_numba: bool = True):
    """
    Load, dark/flat correct and save projections. It is assumed that no dark/flat corrections where applied to the
    input images beforehand.

    :param image_path: path to projection to correct
    :param dark_image_path: path to dark / offset image
    :param flat_image_path: path to dark / gain image (non-dark corrected)
    :param output_path: path to save corrected projection to
    :param maximum: maximum, every value exceeding the maximum is clipped
    :param mean: bright value that will be reached /w linear scaling @ maximum (e.g. mean of the flat correction image).
                 if None the mean of the flat correction image is used
    :param use_numba: use numba JIT (improves performance massively). Numba must be installed
    """
    header, image = raw2py(image_path)
    _, dark_image = raw2py(dark_image_path)
    _, flat_image = raw2py(flat_image_path)
    try:
        corrected_image = flat_correction(image, flat_image, dark_correction_image=dark_image, maximum=maximum,
                                          mean=mean, use_numba=use_numba)
    except NameError:
        raise ModuleNotFoundError('to use the beam hardening correction the ctmt algorithms module must be installed')
    py2raw(corrected_image, output_path, header)


def beam_hardening_correct_by_path(image_path: Path | str, correction_value: int, output_path: Path | str):
    """
    Load, correct beam hardening and save projections with adjusted header.

    :param image_path: path to projection to correct
    :param correction_value: value used for correction
    :param output_path: path to save corrected projection to
    """
    header, image = raw2py(image_path)
    header, corrected_image = apply_beam_hardening_correction(header, image, correction_value)
    py2raw(corrected_image, output_path, header)


# CORRECTIONS BY HEADER / IMAGE (CAN BE USED WITH BatchProjectionManipulator)


def apply_beam_hardening_correction(header: EzrtHeader, image: np.ndarray,
                                    correction_value: int) -> tuple[EzrtHeader, np.ndarray]:
    """
    Correct image greyvalues to mitigate beam hardening and change EzrtHeader accordingly.

    :param header: header to correct
    :param image: projection to correct
    :param correction_value: value used for correction
    """
    try:
        corrected_image = beam_hardening_correction(image, correction_value)
    except NameError:
        raise ModuleNotFoundError('to use the beam hardening correction the ctmt algorithms module must be installed')
    header.inull_value = max(0, header.inull_value - correction_value)
    return header, corrected_image


def scale_projection(header: EzrtHeader, image: np.ndarray, zoom_factor: float) -> tuple[EzrtHeader, np.ndarray]:
    """
    Scale the projections to certain zoom factor in both dimensions and adjust headers accordingly.

    :param header: header of projection to zoom
    :param image: projection to zoom
    :param zoom_factor: zoom factor to use
    """
    image_width = int(image.shape[0] * zoom_factor)
    image_height = int(image.shape[1] * zoom_factor)

    try:
        zoomed_image = cv2.resize(image, (image_width, image_height))
    except NameError:
        raise ModuleNotFoundError('to use projection scaling, the cv2 and scipy modules must be installed via \
            "pip install opencv-contrib-python-headless"')

    header.pixel_width_in_um = header.detector_width_in_um / image_width
    header.number_horizontal_pixels = image_width
    header.number_vertical_pixels = image_height
    header.image_width = image_width
    header.image_height = image_height

    return header, zoomed_image
