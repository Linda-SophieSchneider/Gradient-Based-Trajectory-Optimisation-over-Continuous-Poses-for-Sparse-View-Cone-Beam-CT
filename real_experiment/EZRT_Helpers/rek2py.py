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


def rek2py(filepath: Path | str, switch_order: bool = False,
           compression: str | None = None, **kwargs) -> tuple[EzrtHeader, np.ndarray]:
    """
    Read a .rek volume file into an internal Python representation (3-dim numpy array) and its header.

    :param filepath: file path to .rek file
    :param switch_order: toggle order of numpy array shape (True: (#images, H, W); False (W, H, #images))
    :param compression: compression algorithm to use (currently supported: None (=uncompressed), lz4, gzip)
    :param kwargs: further arguments for compression algorithm
    :return: tuple with EzrtHeader object and 3D numpy array representation of .rek file
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f'given path is not a file [{filepath}]')

    if compression is None:
        with open(filepath, 'rb') as f:
            raw_file_data = f.read()
    elif compression == 'lz4':
        try:
            with lz4.frame.open(filepath, mode='rb', **kwargs) as f:
                raw_file_data = f.read()
        except NameError:
            raise NameError('to use lz4 compression, the lz4 module must be installed via "pip install lz4"')
    elif compression == 'gzip':
        with gzip.open(filepath, 'rb', **kwargs) as f:
            raw_file_data = f.read()
    else:
        raise ValueError('compression algorithm not supported')

    ezrt_header = EzrtHeader.frombuffer(raw_file_data[:2048])  # type: ignore
    dtype = EzrtHeader.convert_to_numpy_bitdepth(ezrt_header.bit_depth)
    # import image payload data to numpy array (excluding header)
    image = np.frombuffer(raw_file_data[2048:], dtype=dtype)
    if switch_order:
        shape = ezrt_header.number_of_images, ezrt_header.image_height, ezrt_header.image_width
    else:
        shape = ezrt_header.image_width, ezrt_header.image_height, ezrt_header.number_of_images
    return ezrt_header, image.reshape(shape)
