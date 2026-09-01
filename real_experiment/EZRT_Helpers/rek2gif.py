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

from PIL import Image
import numpy as np

from .rek2py import rek2py


def rek2gif(rek_volume_path: Path | str, output_filename: Path | str, skip_projections: int = 5,
            downsample_factor: int = 4, scaling_value: float | None = None, resize_to: tuple[int, int] | None = None,
            frame_duration_in_ms: int = 200, loop_count: int = 0):
    """Generate a GIF from a EZRT REK volume.

    :param projection_folder: path to folder with projections
    :param projection_prefix: projection file prefix (e.g. image for image_0000.raw)
    :param output_filename: path and filename of generated GIF
    :param skip_projections: number of projection to skip during GIF creation (to save memory / disk space)
    :param downsample_factor: downsample factor to make more sparse projections (to save memory / disk space)
    :param scaling_value: greyscale truncating / scaling factor (if None GIF is not scaled)
    :param resize_to: resize image to new width / height (if None GIF is not resized)
    :param frame_duration_in_ms: duration of one frame in ms
    :param loop_count: number of loops until it stops (0: forever)
    """
    rek_volume_path = Path(rek_volume_path)

    if not rek_volume_path.is_file() and not rek_volume_path.suffix == '.rek':
        raise FileNotFoundError('rek volume path invalid')
    if skip_projections < 1:
        raise ValueError('value for skip projections must be > 0')
    if downsample_factor < 1:
        raise ValueError('downsampling factor must be > 0')
    if scaling_value is not None and not 0.0 <= scaling_value <= 1.0:
        raise ValueError('scaling factor must be between 0.0 and 1.0')
    if loop_count < 0:
        raise ValueError('loop count must be >= 0')
    if frame_duration_in_ms < 1:
        raise ValueError('frame duration must be > 0')

    max_greyvalue = 0
    min_greyvalue = 65536

    if scaling_value is not None:
        header, volume = rek2py(rek_volume_path, switch_order=True)
        for i in range(0, header.number_of_images)[::skip_projections]:
            image = volume[i, ::downsample_factor, ::downsample_factor]
            current_min = image.min()
            current_max = image.max()
            if current_min < min_greyvalue:
                min_greyvalue = current_min
            if current_max > max_greyvalue:
                max_greyvalue = current_max

        max_greyvalue = max_greyvalue - int(np.floor(scaling_value * max_greyvalue))

        if min_greyvalue > 255:
            min_greyvalue = int(min_greyvalue / 255)
        if max_greyvalue > 255:
            max_greyvalue = int(max_greyvalue / 255)

    # use generator to avoid needing an image array
    def projection_generator(rek_volume_path, min_greyvalue, max_greyvalue):
        header, volume = rek2py(rek_volume_path, switch_order=True)

        for i in range(0, header.number_of_images)[::skip_projections]:
            image = volume[i, ::downsample_factor, ::downsample_factor]

            if scaling_value is not None:
                image = np.copy(image)
                # truncate
                image[image > max_greyvalue] = max_greyvalue

            if image.max() <= 255:
                image = image.astype(np.uint8)
            else:
                image = (image // 256).astype(np.uint8)

            if scaling_value is not None:
                # scale
                image = ((image - min_greyvalue) / (max_greyvalue - min_greyvalue)) * 255
                image = np.floor(image).astype(np.uint8)

            pil_image = Image.fromarray(image)
            if resize_to is not None:
                pil_image.resize(resize_to, Image.BILINEAR)

            yield pil_image

    images = projection_generator(rek_volume_path, min_greyvalue, max_greyvalue)
    next(images).save(fp=output_filename, format='GIF', append_images=images, save_all=True,
                      duration=frame_duration_in_ms, loop=loop_count)

    return min_greyvalue, max_greyvalue
