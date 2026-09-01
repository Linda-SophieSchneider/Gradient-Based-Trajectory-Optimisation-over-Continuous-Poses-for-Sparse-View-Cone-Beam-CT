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

from pathlib import Path

import numpy as np

from .ezrt_header import EzrtHeader


def py2raw(image_array: np.ndarray, savepath: Path | str, input_header: EzrtHeader | bytes | None = None,
           switch_order: bool = False):
    """
    Save an internal Python projection representation (2-dim np array) as a .raw projection file with an specified
    header.

    :param image_array: 2D numpy input array to save
    :param filepath: file path where new .raw file is saved
    :param input_header: EzrtHeader to use (if None, a default header is generated)
    :param switch_order: toggle order of numpy array shape (True: (H, W); False (W, H))
    :param compression: compression algorithm to use (currently supported: None (=uncompressed), lz4, gzip)
    :param kwargs: further arguments for compression algorithm
    """
    if len(image_array.shape) != 2:
        raise ValueError(f'input image array has wrong dimensions ({len(image_array.shape)})')

    if switch_order:
        image_array = image_array.reshape(tuple(reversed(image_array.shape)))

    if input_header is not None:
        if isinstance(input_header, EzrtHeader):
            header = input_header
        elif isinstance(input_header, bytes) or isinstance(input_header, bytearray):
            header = EzrtHeader.frombuffer(input_header)
        else:
            raise ValueError('input header data type not supported')
    else:
        converted_bitdepth = EzrtHeader.convert_to_ezrt_bitdepth(image_array.dtype)
        header = EzrtHeader(image_array.shape[0], image_array.shape[1], converted_bitdepth, 1)

    data = header.tobytes() + image_array.ravel().tobytes()
    with open(savepath, 'wb') as outfile:
        outfile.write(data)
