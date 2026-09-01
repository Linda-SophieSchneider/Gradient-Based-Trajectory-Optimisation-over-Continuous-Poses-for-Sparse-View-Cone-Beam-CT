# FRAUNHOFER IIS CONFIDENTIAL
# __________________
#
# Fraunhofer IIS
# Copyright (c) 2015-2021
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

import numpy as np
from pathlib import Path

from .ezrt_header import EzrtHeader


def raw2py(filepath: str | Path, switch_order: bool = False) -> tuple[EzrtHeader, np.ndarray]:
    """
    Read a .raw projection file into an internal Python representation (2-dim numpy array) and its header.

    :param filepath: file path to .rek file
    :param switch_order: toggle order of numpy array shape (True: (H, W); False (W, H))
    :return: tuple with EzrtHeader object and 2D numpy array representation of .raw file
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f'given path is not a file [{filepath}]')

    with open(filepath, 'rb') as fd:
        raw_file_data = fd.read()

    ezrt_header = EzrtHeader.frombuffer(raw_file_data[:2048])
    if ezrt_header.number_of_images > 1:
        raise ValueError('input file is a .rek volume (number of images > 1) - use rek2py instead')

    dtype = EzrtHeader.convert_to_numpy_bitdepth(ezrt_header.bit_depth)
    # import image payload data to numpy array (excluding header)
    image = np.frombuffer(raw_file_data[2048:], dtype=dtype)
    if switch_order:
        shape = ezrt_header.image_height, ezrt_header.image_width
    else:
        shape = ezrt_header.image_width, ezrt_header.image_height

    return ezrt_header, image.reshape(shape)
