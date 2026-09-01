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

import gzip
from pathlib import Path

import numpy as np
try:
    import lz4.frame
except ModuleNotFoundError:
    pass

from .ezrt_header import EzrtHeader


def py2rek(volume_array: np.ndarray, savepath: Path | str, input_header: EzrtHeader | bytes | None = None,
           switch_order: bool = True, compression: str | None = None, **kwargs):
    """
    Save an internal Python volume representation (3-dim np array) as a .rek volume file with an specified
    header.

    :param volume_array: 3D numpy input array to save
    :param filepath: file path where new .rek file is saved
    :param input_header: EzrtHeader to use (if None, a default header is generated)
    :param switch_order: toggle order of numpy array shape (True: (#images, H, W); False (W, H, #images))
    :param compression: compression algorithm to use (currently supported: None (=uncompressed), lz4, gzip)
    :param kwargs: further arguments for compression algorithm
    """
    if len(volume_array.shape) != 3:
        raise ValueError(f'input volume array has wrong dimensions ({len(volume_array.shape)} vs. 3)')

    if switch_order:
        volume_array = volume_array.reshape(tuple(reversed(volume_array.shape)))

    if input_header is not None:
        # check if input is already a RawHeader or if it is a raw buffer
        if isinstance(input_header, EzrtHeader):
            header = input_header
        elif isinstance(input_header, bytes) or isinstance(input_header, bytearray):
            header = EzrtHeader.frombuffer(input_header)
        else:
            raise ValueError('input header data type not supported')
    else:
        converted_bitdepth = EzrtHeader.convert_to_ezrt_bitdepth(volume_array.dtype)
        image_width = volume_array.shape[0]
        image_height = volume_array.shape[1]
        number_of_images = volume_array.shape[2]
        header = EzrtHeader(image_width, image_height, converted_bitdepth, number_of_images)

    data = header.tobytes() + volume_array.ravel().tobytes()

    if compression is None:
        with open(savepath, 'wb') as f:
            f.write(data)
    elif compression == 'lz4':
        try:
            with lz4.frame.open(savepath, 'wb', **kwargs) as f:
                f.write(data)  # type: ignore
        except NameError:
            raise ModuleNotFoundError('to use lz4 compression, the lz4 module must be installed via "pip install lz4"')
    elif compression == 'gzip':
        with gzip.open(savepath, 'wb', **kwargs) as f:
            f.write(data)
    else:
        raise ValueError('compression algorithm not supported')
