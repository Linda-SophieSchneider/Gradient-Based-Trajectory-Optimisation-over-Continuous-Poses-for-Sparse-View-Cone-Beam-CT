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


import warnings
from pathlib import Path
from copy import deepcopy
from json import loads, dumps, JSONDecodeError

import numpy as np

from .common_types import EZRT_HEADER_DTYPES, ACQUISITIONGEOMETRY, CTALGORITHM, CTFILTER
from .common_types import CTALGORITHMPLATFORM, CTPROJECTIONPADDING


def custom_deep_copy(dict_to_copy: dict) -> dict:
    """Faster than deepcopy, for a small dict of simple python types."""
    copied_dict = dict().fromkeys(dict_to_copy)
    for key, value in dict_to_copy.items():
        if isinstance(value, (dict, set)):
            copied_dict[key] = value.copy()
        elif isinstance(value, (list, tuple, str, bytes)):
            copied_dict[key] = value[:]
        else:
            copied_dict[key] = value

    return copied_dict


class EzrtHeader:
    """EZRT projection header class"""
    # explicitely state __slots__ to reduce memory footprint and to speed up instantiation and attribute access.
    # deactivates dynamic attribute creation, which should not be needed - in case it is needed, just remove __slots__
    __slots__ = ['image_width', 'image_height', 'bit_depth', 'number_of_images', 'header_length', 'major_version',
                 'minor_version', 'revision', 'measurement_id', '_focus_detector_distance_in_um',
                 '_focus_object_distance_in_um', 'number_projection_angles', 'detector_width_in_um',
                 'detector_height_in_um', 'number_horizontal_pixels', 'number_vertical_pixels', 'measuring_range_start',
                 'reconstruction_start_line', 'reconstruction_end_line', 'pixel_width_in_um', 'acquisition_geometry',
                 'unused', 'horizontal_shift_in_px', 'vertical_shift_in_px', 'detector_slant_in_deg', '_voltage_in_v',
                 '_current_in_ua', '_exposure_time_in_ms', 'number_averages', 'number_skip_images',
                 'z_axis_position_in_100nm', 'y_detector_position_in_100nm', 'y_sample_position_in_100nm',
                 '_measurement_name', '_date', '_time', 'n_beam_hardening', 'f_beam_hardening_polynomial',
                 'keep_data_or_z_step_in_100nm', 'num_voxel_x', 'num_voxel_y', 'num_voxel_z', 'reco_norm_factor',
                 'voxel_size_x_in_um', 'voxel_size_z_in_um', 'inull_value', 'inull_x_position', 'inull_y_position',
                 'inull_delta_x', 'inull_delta_y', 'inull_align', 'swing_start_angle_in_rad', 'swing_angular_step',
                 '_prefilter', 'file_time', 'header_content', 'reserved', 'reco_vol_min', 'reco_vol_max', 'reco_offset',
                 'reco_max_z_slices', 'reco_first_z_slice', 'reco_last_z_slice', 'ct_algorithm', 'ct_filter',
                 'algorithm_platform', 'projection_padding', 'padding_object_radius', 'filter_param_primary',
                 'recoparamex_reserved', 'z_shift_per_projection_in_um', 'scan_range_in_rad', 'projections_per_z_shift',
                 '_agv_source_position', '_agv_source_direction', '_agv_detector_center_position', 'helix_align',
                 '_agv_detector_line_direction', '_agv_detector_col_direction', '_agv_reco_reference',
                 '_agv_axis_angle', 'range_ext_size_row', 'range_ext_size_col', 'multiscan', 'range_ext_overlap_row',
                 'range_ext_overlap_col', 'detector_skew', 'detector_tilt', '_empty_field', 'endian', 'reserved_field',
                 'endian64', '_user_string']

    def __init__(self, image_width: int = 0, image_height: int = 0, bit_depth: int = 8, number_of_images: int = 1,
                 number_voxels: tuple[int, int, int] = (0, 0, 0), metadata: dict | None = None,
                 use_custom_deepcopy: bool = True):
        """
        Construct new EzrtHeader with given parameters.

        :param image_width: image width
        :param image_height: image height
        :param bit_depth: bit depth
        :param number_of_images: number of images stacked in the file (for projections: 1)
        :param number_voxels: tuple with number of reco voxels in x, y and z (default: 0, 0, 0)
        :param metadata: dict with metadata info
        :param use_custom_deepcopy: use custom deepcopy for metadata - faster, but only viable for simply Python types
        """
        if image_width < 0 or image_height < 0:
            raise ValueError('image dimensions must be > 0')
        if bit_depth <= 0:
            raise ValueError('bit depth must be > 0')
        if number_of_images <= 0:
            raise ValueError('number of images must be > 0')
        if len(number_voxels) != 3:
            raise ValueError('wrong size of number of voxels (must be tuple of length 3)')

        self.image_width: int = image_width
        self.image_height: int = image_height
        self.bit_depth: int = bit_depth
        self.number_of_images: int = number_of_images
        self.header_length: int = 2048
        self.major_version: int = 2
        self.minor_version: int = 6
        self.revision: int = 0

        # --------------------------------measurement---------------------------
        self.measurement_id: int = 0
        self._focus_detector_distance_in_um = 0
        self._focus_object_distance_in_um = 0
        self.number_projection_angles: int = 0
        self.detector_width_in_um: int | float = 0
        self.detector_height_in_um: int | float = 0
        self.number_horizontal_pixels: int = image_width
        self.number_vertical_pixels: int = image_height
        self.measuring_range_start: int = 0
        self.reconstruction_start_line: int = 0
        self.reconstruction_end_line: int = 0
        self.pixel_width_in_um: float = 0.0
        self.acquisition_geometry: ACQUISITIONGEOMETRY | int = ACQUISITIONGEOMETRY.CONVENTIONAL_3DCT
        self.unused: int = 0
        self.horizontal_shift_in_px: float = 0.0
        self.vertical_shift_in_px: float = 0.0
        self.detector_slant_in_deg: float = 0.0

        # -----------------------------------docu-------------------------------
        self._voltage_in_v = 0
        self._current_in_ua = 0
        self._exposure_time_in_ms: int = 0
        self.number_averages: int = 0
        self.number_skip_images: int = 0
        self.z_axis_position_in_100nm: int = 0
        self.y_detector_position_in_100nm: int = 0
        self.y_sample_position_in_100nm: int = 0
        self._measurement_name = '\00' * 288
        self._date = '\00' * 12
        self._time = '\00' * 8
        self.n_beam_hardening: int = 0
        self.f_beam_hardening_polynomial: tuple[float, ...] = (0.0,) * 31
        self.keep_data_or_z_step_in_100nm: int = 0

        # -----------------------------------reco-------------------------------
        self.num_voxel_x: int = number_voxels[0]
        self.num_voxel_y: int = number_voxels[1]
        self.num_voxel_z: int = number_voxels[2]
        self.reco_norm_factor: float = 0.0
        self.voxel_size_x_in_um: float = 0.0
        self.voxel_size_z_in_um: float = 0.0

        # -----------------------------------inull------------------------------
        self.inull_value: int = 0
        self.inull_x_position: int = 0
        self.inull_y_position: int = 0
        self.inull_delta_x: int = 0
        self.inull_delta_y: int = 0
        self.inull_align: int = 0

        # -----------------------------------swinglam---------------------------
        self.swing_start_angle_in_rad: float = 0.0
        self.swing_angular_step: float = 0.0

        # -----------------------------------docuex-----------------------------
        self._prefilter = '\00' * 32
        self.file_time: int = 0
        self.header_content: int = 0
        self.reserved: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)

        # ------------------------------- reco param ex-------------------------
        self.reco_vol_min: float = 0.0
        self.reco_vol_max: float = 0.0
        self.reco_offset: float = 0.0
        self.reco_max_z_slices: int = 0
        self.reco_first_z_slice: int = 0
        self.reco_last_z_slice: int = 0
        self.ct_algorithm: CTALGORITHM | int = CTALGORITHM.CYLINDRICAL_FBP
        self.ct_filter: CTFILTER | int = CTFILTER.SHEPP_LOGAN
        self.algorithm_platform: CTALGORITHMPLATFORM | int = CTALGORITHMPLATFORM.DYNAMIC
        self.projection_padding: CTPROJECTIONPADDING | int = CTPROJECTIONPADDING.NONE
        self.padding_object_radius: float = 0.0
        self.filter_param_primary: float = 0.0
        self.recoparamex_reserved: tuple[int, int, int, int] = (0, 0, 0, 0)

        # ---------------------------------- helix ------------------------------
        self.z_shift_per_projection_in_um: float = 0.0
        self.scan_range_in_rad: float = 0.0
        self.projections_per_z_shift = 0
        self.helix_align = 0

        # ---------------------------- arbitrary geometry -----------------------
        self._agv_source_position = np.zeros(3, dtype=np.float64)
        self._agv_source_direction = np.zeros(3, dtype=np.float64)
        self._agv_detector_center_position = np.zeros(3, dtype=np.float64)
        self._agv_detector_line_direction = np.zeros(3, dtype=np.float64)
        self._agv_detector_col_direction = np.zeros(3, dtype=np.float64)
        self._agv_reco_reference = np.zeros(3, dtype=np.float64)
        self._agv_axis_angle = np.zeros(4, dtype=np.float64)

        # ----------------------------- range extension ------------------------
        self.range_ext_size_row: int = 0
        self.range_ext_size_col: int = 0
        self.multiscan: int = 0
        self.range_ext_overlap_row: int = 0
        self.range_ext_overlap_col: int = 0

        # --------------------------------- meas ex ----------------------------
        self.detector_skew: float = 0.0
        self.detector_tilt: float = 0.0

        # ---------------------------------- empty ------------------------------
        self._empty_field = '\00' * 24

        # -----------------------------------platform---------------------------
        self.endian: int = 0x01020304
        self.reserved_field: int = 0
        self.endian64: int = 0x0102030405060708

        # ---------------------------------- user ------------------------------
        self._user_string = '\00' * 1024

        # -------------------------------- metadata ----------------------------
        if metadata is None:
            return

        self.parse_metadata(metadata, use_custom_deepcopy)

    def __len__(self):
        return self.header_length

    def __str__(self):
        return f'### EZRT HEADER {self.major_version}.{self.minor_version}.{self.revision} ###\n'\
            + '# IMAGE\n'\
            + f' - size (W x H x D) @ bitdepth: {self.image_width} x {self.image_height} x {self.number_of_images}'\
            + f' @ {self.bit_depth} bit\n'\
            + f' - header length: {self.header_length}\n'\
            + '# MEASUREMENT\n'\
            + f' - measurement id: {self.measurement_id}\n'\
            + f' - FDD / FOD [µm]: {self._focus_detector_distance_in_um:.4f}'\
            + f' / {self._focus_object_distance_in_um:.4f}\n'\
            + f' - number of projection angles: {self.number_projection_angles}\n'\
            + f' - detector width / height [µm]: {self.detector_width_in_um} / {self.detector_height_in_um} \n'\
            + f' - detector pixel horizontal x vertical: {self.number_horizontal_pixels}'\
            + ' x {self.number_vertical_pixels}\n'\
            + f' - start measuring range: {self.measuring_range_start}\n'\
            + f' - line reco start -> end: {self.reconstruction_start_line} -> {self.reconstruction_end_line}\n'\
            + f' - pixel width [µm]: {self.pixel_width_in_um}\n'\
            + f' - acquisition geometry: {self.acquisition_geometry}\n'\
            + f' - horizontal / vertical detector shift: {self.horizontal_shift_in_px} / {self.vertical_shift_in_px}\n'\
            + f' - tilt angle: {self.detector_slant_in_deg}\n'\
            + '# DOCUMENTATION\n'\
            + f' - voltage [kV]: {self.voltage_in_kv}\n'\
            + f' - current [µA]: {self.current_in_ua}\n'\
            + f' - exposure time [ms]: {self._exposure_time_in_ms}\n'\
            + f' - frame average / skips: {self.number_averages} / {self.number_skip_images}\n'\
            + f' - z position [µm]: {self.z_axis_position_in_100nm / 10}\n'\
            + f' - x position detector [µm]: {self.y_detector_position_in_100nm / 10}\n'\
            + f' - x position object [µm]: {self.y_sample_position_in_100nm / 10}\n'\
            + f' - measurement name: {self.measurement_name}\n'\
            + f' - date and time: {self.date} {self.time}\n'\
            + f' - ray: {self.n_beam_hardening}\n'\
            + f' - keep data or z step [µm]: {self.keep_data_or_z_step_in_100nm / 10}\n'\
            + '# RECONSTRUCTION\n'\
            + f' - volume (WxHxD): {self.num_voxel_x}x{self.num_voxel_y}x{self.num_voxel_z}\n'\
            + f' - normalization factor: {self.reco_norm_factor}\n'\
            + f' - voxelsize x / z [µm]: {self.voxel_size_x_in_um} / {self.voxel_size_z_in_um}\n'\
            + '# INULL\n'\
            + f' - i0 value: {self.inull_value}\n'\
            + f' - ROI x origin / size x: {self.inull_x_position} / {self.inull_delta_x}\n'\
            + f' - ROI y origin / size y: {self.inull_y_position} / {self.inull_delta_y}\n'\
            + f' - align: {self.inull_align}\n'\
            + '# LIMITED ANGLE\n'\
            + f' - start angle: {self.swing_start_angle_in_rad}\n'\
            + f' - angle step: {self.swing_angular_step}\n'\
            + '# DOCUMENTATION EX\n'\
            + f' - prefilter: {self.prefilter}\n'\
            + '# AGT\n'\
            + f' - AG source: {self.agv_source_position}\n'\
            + f' - AG source direction: {self.agv_source_direction}\n'\
            + f' - AG detector center: {self.agv_detector_center_position}\n'\
            + f' - AG detector line direction: {self.agv_detector_line_direction}\n'\
            + f' - AG detector col direction: {self.agv_detector_col_direction}\n'\
            + f' - AG reco ref: {self.agv_reco_reference}\n'\
            + f' - AG axis angle: {self.agv_axis_angle}\n'\
            + '# RANGE EXT\n'\
            + f' - RE size row / col: {self.range_ext_size_row} / {self.range_ext_size_col}\n'\
            + f' - RE multiscan: {self.multiscan}\n'\
            + f' - RE row / col overlap: {self.range_ext_overlap_row} / {self.range_ext_overlap_col}\n'\
            + '# MEASUREMENT EXT\n'\
            + f' - detector skew / tilt: {self.detector_skew} / {self.detector_tilt}\n'\
            + '# PLATFORM\n'\
            + f' - reserved: {self.reserved_field}\n'\
            + f' - endian / endian 64-bit: {self.endian} / {self.endian64}\n'\
            + '# USER\n'\
            + f' - user string: {self.user_string}'

    @classmethod
    def frombuffer(cls, buffer: bytes):
        """
        Create new header instance from buffer (byte array).

        :param buffer: buffer / bytearray with header information
        """
        if len(buffer) < 2048:
            raise ValueError('wrong input byte buffer size (must be >= 2048)')

        header = cls()
        offset = 0

        # -----------------------------------image------------------------------
        header.image_width = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.image_height = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.bit_depth = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.number_of_images = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.header_length = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.major_version = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        offset += 2
        header.revision = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2

        # --------------------------------measurement---------------------------
        header.measurement_id = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header._focus_detector_distance_in_um = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header._focus_object_distance_in_um = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.number_projection_angles = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.detector_width_in_um = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.detector_height_in_um = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.number_horizontal_pixels = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.number_vertical_pixels = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.measuring_range_start = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.reconstruction_start_line = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.reconstruction_end_line = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.pixel_width_in_um = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.acquisition_geometry = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += 4
        header.unused = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.horizontal_shift_in_px = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8
        header.vertical_shift_in_px = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8
        header.detector_slant_in_deg = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8

        # -----------------------------------docu-------------------------------
        header._voltage_in_v = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header._current_in_ua = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header._exposure_time_in_ms = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.number_averages = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.number_skip_images = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.z_axis_position_in_100nm = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.y_detector_position_in_100nm = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.y_sample_position_in_100nm = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        try:
            header._measurement_name = np.frombuffer(buffer, dtype='|S1', count=288, offset=offset).tobytes().decode()
        except UnicodeDecodeError:
            warnings.warn('could not decode measurement name - field is left blank')
            header._measurement_name = '\00' * 288
        offset += 288
        try:
            header._date = np.frombuffer(buffer, dtype='|S1', count=12, offset=offset).tobytes().decode()
        except UnicodeDecodeError:
            warnings.warn('could not decode date - field is left blank')
            header._date = '\00' * 12
        offset += 12
        try:
            header._time = np.frombuffer(buffer, dtype='|S1', count=8, offset=offset).tobytes().decode()
        except UnicodeDecodeError:
            warnings.warn('could not decode time - field is left blank')
            header._time = '\00' * 8
        offset += 8
        header.n_beam_hardening = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.f_beam_hardening_polynomial = tuple(np.frombuffer(buffer, dtype=np.float32,
                                                                 count=31, offset=offset).tolist())
        offset += 31 * 4
        header.keep_data_or_z_step_in_100nm = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4

        # -----------------------------------reko-------------------------------
        header.num_voxel_x = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.num_voxel_y = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.num_voxel_z = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.reco_norm_factor = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.voxel_size_x_in_um = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.voxel_size_z_in_um = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4

        # -----------------------------------inull------------------------------
        header.inull_value = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += 4
        header.inull_x_position = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.inull_y_position = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.inull_delta_x = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.inull_delta_y = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.inull_align = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4

        # -----------------------------------swinglam---------------------------
        header.swing_start_angle_in_rad = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8
        header.swing_angular_step = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8

        # -----------------------------------dokuex-----------------------------
        try:
            header._prefilter = np.frombuffer(buffer, dtype='|S1', count=32, offset=offset).tobytes().decode()
        except UnicodeDecodeError:
            warnings.warn('could not decode prefilter - field is left blank')
            header._prefilter = '\00' * 32
        offset += 1 * 32
        header.file_time = np.frombuffer(buffer, dtype=np.uint64, count=1, offset=offset)[0]
        offset += 8
        header.header_content = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.reserved = tuple(np.frombuffer(buffer, dtype=np.int32, count=5, offset=offset).tolist())  # type: ignore
        offset += 5 * 4

        # -------------------------------- rekoparamex--------------------------
        header.reco_vol_min = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.reco_vol_max = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.reco_offset = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.reco_max_z_slices = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.reco_first_z_slice = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.reco_last_z_slice = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.ct_algorithm = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += 4
        header.ct_filter = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.algorithm_platform = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.projection_padding = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.padding_object_radius = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.filter_param_primary = np.frombuffer(buffer, dtype=np.float32, count=1, offset=offset)[0]
        offset += 4
        header.recoparamex_reserved = tuple(np.frombuffer(buffer, dtype=np.int32,   # type: ignore
                                                          count=4, offset=offset).tolist())
        offset += 4 * 4

        # ---------------------------------- helix ------------------------------
        header.z_shift_per_projection_in_um = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8
        header.scan_range_in_rad = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8
        header.projections_per_z_shift = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.helix_align = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4

        # ---------------------------- arbitrary geometry -----------------------
        header._agv_source_position = np.frombuffer(buffer, dtype=np.float64, count=3, offset=offset)
        offset += 3 * 8
        header._agv_source_direction = np.frombuffer(buffer, dtype=np.float64, count=3, offset=offset)
        offset += 3 * 8
        header._agv_detector_center_position = np.frombuffer(buffer, dtype=np.float64, count=3, offset=offset)
        offset += 3 * 8
        header._agv_detector_line_direction = np.frombuffer(buffer, dtype=np.float64, count=3, offset=offset)
        offset += 3 * 8
        header._agv_detector_col_direction = np.frombuffer(buffer, dtype=np.float64, count=3, offset=offset)
        offset += 3 * 8
        header._agv_reco_reference = np.frombuffer(buffer, dtype=np.float64, count=3, offset=offset)
        offset += 3 * 8
        header._agv_axis_angle = np.frombuffer(buffer, dtype=np.float64, count=4, offset=offset)
        offset += 4 * 8

        # ----------------------------- range extension ------------------------
        header.range_ext_size_row = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.range_ext_size_col = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.multiscan = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.range_ext_overlap_row = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2
        header.range_ext_overlap_col = np.frombuffer(buffer, dtype=np.int16, count=1, offset=offset)[0]
        offset += 2

        # --------------------------------- mess ex ----------------------------
        header.detector_skew = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8
        header.detector_tilt = np.frombuffer(buffer, dtype=np.float64, count=1, offset=offset)[0]
        offset += 8

        # -----------------------------------leer-------------------------------
        try:
            header._empty_field = np.frombuffer(buffer, dtype='|S1', count=24, offset=offset).tobytes().decode()
        except UnicodeDecodeError:
            warnings.warn('could not decode empty slot - field is left blank')
            header._empty_field = 24 * '\00'
        offset += 24
        # -----------------------------------platform---------------------------
        header.endian = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += 4
        header.reserved_field = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4
        header.endian64 = np.frombuffer(buffer, dtype=np.uint64, count=1, offset=offset)[0]
        offset += 8
        # -----------------------------------user-------------------------------
        try:
            header._user_string = np.frombuffer(buffer, '|S1', count=1024, offset=offset).tobytes().decode()
        except UnicodeDecodeError:
            warnings.warn('could not decode user string - field is left blank')
            header._user_string = 1024 * '\00'
        offset += 1024

        if offset != header.header_length:
            raise ValueError('offset does not match header length - file may be corrupted')
        return header

    @classmethod
    def fromfile(cls, filename: Path | str) -> EzrtHeader:
        """
        Create new header instance from file.

        :param filename: path to file
        """
        buffer = b''
        with open(filename, 'rb') as fd:
            buffer = fd.read(2048)

        return cls.frombuffer(buffer)

    @classmethod
    def create_for_conventional_3D_ct(cls, image_width_in_px: int, image_height_in_px: int, bit_depth: int,
                                      number_projection_angles: int, fdd_in_mm: float, fod_in_mm: float,
                                      detector_width_in_um: int, detector_height_in_um: int) -> EzrtHeader:
        """
        Create a minimal EZRT header as required for a conventional 3DCT.

        :param image_width_in_px: image width in pixels
        :param image_height_in_px: image height in pixels
        :param bit_depth: image bit depth
        :param number_projection_angles: number of angles of CT
        :param fdd_in_mm: focus-detector distance in millimeters
        :param fod_in_mm: focus-object distance in millimeters
        :param detector_width_in_um: detector width in micrometers
        :param detector_height_in_um: detector height in micrometers

        :return EzrtHeader instance for conventional 3D CT

        """
        header = cls(image_width_in_px, image_height_in_px)
        header.bit_depth = bit_depth
        header.number_projection_angles = number_projection_angles
        header.acquisition_geometry = ACQUISITIONGEOMETRY.CONVENTIONAL_3DCT
        header.focus_detector_distance_in_mm = fdd_in_mm
        header.focus_object_distance_in_mm = fod_in_mm
        header.detector_width_in_um = detector_width_in_um
        header.detector_height_in_um = detector_height_in_um
        return header

    @property
    def exposure_time_in_ms(self) -> float:
        return float(self._exposure_time_in_ms)

    @exposure_time_in_ms.setter
    def exposure_time_in_ms(self, value: float):
        if value < 0.0:
            raise ValueError('exposure time in ms must be >= 0')
        self._exposure_time_in_ms = int(value)

    @property
    def voltage_in_kv(self) -> float:
        return self._voltage_in_v / 1000.0

    @voltage_in_kv.setter
    def voltage_in_kv(self, value: float):
        if value < 0.0:
            raise ValueError('voltage in kV must be >= 0')
        self._voltage_in_v = int(value) * 1000

    @property
    def current_in_ua(self) -> float:
        return float(self._current_in_ua)

    @current_in_ua.setter
    def current_in_ua(self, value: float):
        if value < 0.0:
            raise ValueError('current in µA must be >= 0')
        self._current_in_ua = int(value)

    @property
    def focus_object_distance_in_mm(self) -> float:
        return self._focus_object_distance_in_um / 1000.0

    @focus_object_distance_in_mm.setter
    def focus_object_distance_in_mm(self, fod: float):
        if fod < 0.0:
            raise ValueError('fod value must be >= 0')
        self._focus_object_distance_in_um = int(fod * 1000.0)

    @property
    def focus_detector_distance_in_mm(self) -> float:
        return self._focus_detector_distance_in_um / 1000.0

    @focus_detector_distance_in_mm.setter
    def focus_detector_distance_in_mm(self, fdd: float):
        if fdd < 0.0:
            raise ValueError('fdd value must be >= 0')
        self._focus_detector_distance_in_um = int(fdd * 1000.0)

    @property
    def magnification(self) -> float:
        """
        Magnification of measurement (convenience field - not actually part of header).

        :return: magnification factor (or -1.0 if magnification cannot be calculated)
        """
        if self.focus_object_distance_in_mm <= 0.0 or self.focus_object_distance_in_mm <= 0.0:
            warnings.warn('FOD and/or FDD not set - magnification cannot be calculated')
            return -1.0
        return self._focus_detector_distance_in_um / self._focus_object_distance_in_um

    @property
    def voxel_size_in_um(self) -> float:
        """
        Resulting voxel size of measurement geometry (convenience field - not actually part of header).

        :return: voxel size in µm (or 0.0 if voxel size cannot be calculated)
        """
        if self.pixel_width_in_um > 0.0:
            pixel_width_in_um = self.pixel_width_in_um
        elif self.image_width > 0 and self.detector_width_in_um > 0.0:
            pixel_width_in_um = self.detector_width_in_um / self.image_width
        else:
            warnings.warn('voxel size cannot be calculated (pixel width or detector_width missing)')
            return 0.0

        voxel_size_in_um = pixel_width_in_um / self.magnification
        return voxel_size_in_um if voxel_size_in_um >= 0.0 else 0.0

    @property
    def agv_source_position(self) -> tuple[float, float, float]:
        return tuple(self._agv_source_position)   # type: ignore

    @agv_source_position.setter
    def agv_source_position(self, vector: tuple[float, float, float]):
        if len(vector) != 3:
            raise ValueError('wrong input vector length')
        self._agv_source_position = np.array(vector, dtype=np.float64)

    @property
    def agv_source_direction(self) -> tuple[float, float, float]:
        return tuple(self._agv_source_direction)   # type: ignore

    @agv_source_direction.setter
    def agv_source_direction(self, vector: tuple[float, float, float]):
        if len(vector) != 3:
            raise ValueError('wrong input vector length')
        self._agv_source_direction = np.array(vector, dtype=np.float64)

    @property
    def agv_detector_center_position(self) -> tuple[float, float, float]:
        return tuple(self._agv_detector_center_position)   # type: ignore

    @agv_detector_center_position.setter
    def agv_detector_center_position(self, vector: tuple[float, float, float]):
        if len(vector) != 3:
            raise ValueError('wrong input vector length')
        self._agv_detector_center_position = np.array(vector, dtype=np.float64)

    @property
    def agv_detector_line_direction(self) -> tuple[float, float, float]:
        return tuple(self._agv_detector_line_direction)   # type: ignore

    @agv_detector_line_direction.setter
    def agv_detector_line_direction(self, vector: tuple[float, float, float]):
        if len(vector) != 3:
            raise ValueError('wrong input vector length')
        self._agv_detector_line_direction = np.array(vector, dtype=np.float64)

    @property
    def agv_detector_col_direction(self) -> tuple[float, float, float]:
        return tuple(self._agv_detector_col_direction)   # type: ignore

    @agv_detector_col_direction.setter
    def agv_detector_col_direction(self, vector: tuple[float, float, float]):
        if len(vector) != 3:
            raise ValueError('wrong input vector length')
        self._agv_detector_col_direction = np.array(vector, dtype=np.float64)

    @property
    def agv_reco_reference(self) -> tuple[float, float, float]:
        return tuple(self._agv_reco_reference)   # type: ignore

    @agv_reco_reference.setter
    def agv_reco_reference(self, vector: tuple[float, float, float]):
        if len(vector) != 3:
            raise ValueError('wrong input vector length')
        self._agv_reco_reference = np.array(vector, dtype=np.float64)

    @property
    def agv_axis_angle(self) -> tuple[float, float, float, float]:
        return tuple(self._agv_axis_angle)   # type: ignore

    @agv_axis_angle.setter
    def agv_axis_angle(self, vector: tuple[float, float, float, float]):
        if len(vector) != 4:
            raise ValueError('wrong input vector length')
        self._agv_axis_angle = np.array(vector, dtype=np.float64)

    @property
    def measurement_name(self) -> str:
        return str(self._measurement_name).replace('\x00', '').strip()

    @measurement_name.setter
    def measurement_name(self, name: str):
        if len(name) > 287:
            raise ValueError('measurement_name string must be < 288 characters')
        self._measurement_name = name + (288 - len(name)) * '\00'

    @property
    def date(self) -> str:
        return str(self._date).replace('\x00', '').strip()

    @date.setter
    def date(self, date: str):
        if len(date) > 11:
            raise ValueError('date string must be < 12 characters')
        self._date = date + (12 - len(date)) * '\00'

    @property
    def time(self) -> str:
        return str(self._time).replace('\x00', '').strip()

    @time.setter
    def time(self, time: str):
        if len(time) > 7:
            raise ValueError('time string must be < 8 characters')
        self._time = time + (8 - len(time)) * '\00'

    @property
    def prefilter(self) -> str:
        return str(self._prefilter).replace('\x00', '').strip()

    @prefilter.setter
    def prefilter(self, prefilter: str):
        if len(prefilter) > 31:
            raise ValueError('prefilter string must be < 32 characters')
        self._prefilter = prefilter + (32 - len(prefilter)) * '\00'

    @property
    def user_string(self) -> str:
        return str(self._user_string).replace('\x00', '')

    @user_string.setter
    def user_string(self, string: str):
        if len(string) > 1023:
            raise ValueError('user_string must be < 1024 characters')
        self._user_string = string + (1024 - len(string)) * '\00'

    @property
    def version(self) -> str:
        return f'{self.major_version}.{self.minor_version}.{self.revision}'

    @staticmethod
    def convert_to_ezrt_bitdepth(numpy_bitdepth) -> EZRT_HEADER_DTYPES:
        if numpy_bitdepth == np.int8:
            converted_bitdepth = EZRT_HEADER_DTYPES.INT8
        elif numpy_bitdepth == np.int16:
            converted_bitdepth = EZRT_HEADER_DTYPES.INT16
        elif numpy_bitdepth == np.int32:
            converted_bitdepth = EZRT_HEADER_DTYPES.INT32
        elif numpy_bitdepth == np.int64:
            converted_bitdepth = EZRT_HEADER_DTYPES.INT64
        elif numpy_bitdepth == np.uint8:
            converted_bitdepth = EZRT_HEADER_DTYPES.UINT8
        elif numpy_bitdepth == np.uint16:
            converted_bitdepth = EZRT_HEADER_DTYPES.UINT16
        elif numpy_bitdepth == np.uint32:
            converted_bitdepth = EZRT_HEADER_DTYPES.UINT32
        elif numpy_bitdepth == np.uint64:
            converted_bitdepth = EZRT_HEADER_DTYPES.UINT64
        elif numpy_bitdepth == np.float32:
            converted_bitdepth = EZRT_HEADER_DTYPES.FLOAT
        elif numpy_bitdepth == np.float64:
            converted_bitdepth = EZRT_HEADER_DTYPES.DOUBLE
        else:
            raise ValueError('bitdepth not supported')

        return converted_bitdepth

    @staticmethod
    def convert_to_numpy_bitdepth(ezrt_bitdepth: EZRT_HEADER_DTYPES | int):
        if ezrt_bitdepth == EZRT_HEADER_DTYPES.INT8:
            converted_bitdepth = np.int8
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.INT16:
            converted_bitdepth = np.int16
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.INT32:
            converted_bitdepth = np.int32
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.INT64:
            converted_bitdepth = np.int64
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.UINT8:
            converted_bitdepth = np.uint8
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.UINT16:
            converted_bitdepth = np.uint16
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.UINT32:
            converted_bitdepth = np.uint32
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.UINT64:
            converted_bitdepth = np.uint64
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.FLOAT:
            converted_bitdepth = np.float32
        elif ezrt_bitdepth == EZRT_HEADER_DTYPES.DOUBLE:
            converted_bitdepth = np.float64
        else:
            raise ValueError('bitdepth not supported')

        return converted_bitdepth

    def tobytes(self) -> bytes:
        """
        Convert header to byte array according to header standard definition.

        :return: byte representation of header
        """
        raw_bytes = bytearray()

        # -----------------------------------image------------------------------
        raw_bytes += np.int16(self.image_width).tobytes()
        raw_bytes += np.int16(self.image_height).tobytes()
        raw_bytes += np.int16(self.bit_depth).tobytes()
        raw_bytes += np.int16(self.number_of_images).tobytes()
        raw_bytes += np.int16(self.header_length).tobytes()
        raw_bytes += np.int16(self.major_version).tobytes()
        raw_bytes += np.int16(self.minor_version).tobytes()
        raw_bytes += np.int16(self.revision).tobytes()

        # --------------------------------measurement---------------------------
        raw_bytes += np.int32(self.measurement_id).tobytes()
        raw_bytes += np.int32(self._focus_detector_distance_in_um).tobytes()
        raw_bytes += np.int32(self._focus_object_distance_in_um).tobytes()
        raw_bytes += np.int32(self.number_projection_angles).tobytes()
        raw_bytes += np.int32(self.detector_width_in_um).tobytes()
        raw_bytes += np.int32(self.detector_height_in_um).tobytes()
        raw_bytes += np.int32(self.number_horizontal_pixels).tobytes()
        raw_bytes += np.int32(self.number_vertical_pixels).tobytes()
        raw_bytes += np.int32(self.measuring_range_start).tobytes()
        raw_bytes += np.int32(self.reconstruction_start_line).tobytes()
        raw_bytes += np.int32(self.reconstruction_end_line).tobytes()
        raw_bytes += np.float32(self.pixel_width_in_um).tobytes()
        raw_bytes += np.uint32(self.acquisition_geometry).tobytes()
        raw_bytes += np.int32(self.unused).tobytes()
        raw_bytes += np.float64(self.horizontal_shift_in_px).tobytes()
        raw_bytes += np.float64(self.vertical_shift_in_px).tobytes()
        raw_bytes += np.float64(self.detector_slant_in_deg).tobytes()

        # -----------------------------------docu-------------------------------
        raw_bytes += np.int32(self._voltage_in_v).tobytes()
        raw_bytes += np.int32(self._current_in_ua).tobytes()
        raw_bytes += np.int32(self._exposure_time_in_ms).tobytes()
        raw_bytes += np.int32(self.number_averages).tobytes()
        raw_bytes += np.int32(self.number_skip_images).tobytes()
        raw_bytes += np.int32(self.z_axis_position_in_100nm).tobytes()
        raw_bytes += np.int32(self.y_detector_position_in_100nm).tobytes()
        raw_bytes += np.int32(self.y_sample_position_in_100nm).tobytes()
        raw_bytes += self._measurement_name.encode()
        raw_bytes += self._date.encode()
        raw_bytes += self._time.encode()

        raw_bytes += np.int32(self.n_beam_hardening).tobytes()
        raw_bytes += np.asarray(self.f_beam_hardening_polynomial, dtype=np.float32).tobytes()
        raw_bytes += np.int32(self.keep_data_or_z_step_in_100nm).tobytes()

        # -----------------------------------reco-------------------------------
        raw_bytes += np.int32(self.num_voxel_x).tobytes()
        raw_bytes += np.int32(self.num_voxel_y).tobytes()
        raw_bytes += np.int32(self.num_voxel_z).tobytes()
        raw_bytes += np.float32(self.reco_norm_factor).tobytes()
        raw_bytes += np.float32(self.voxel_size_x_in_um).tobytes()
        raw_bytes += np.float32(self.voxel_size_z_in_um).tobytes()

        # -----------------------------------inull------------------------------
        raw_bytes += np.uint32(self.inull_value).tobytes()
        raw_bytes += np.int16(self.inull_x_position).tobytes()
        raw_bytes += np.int16(self.inull_y_position).tobytes()
        raw_bytes += np.int16(self.inull_delta_x).tobytes()
        raw_bytes += np.int16(self.inull_delta_y).tobytes()
        raw_bytes += np.int32(self.inull_align).tobytes()

        # -----------------------------------swinglam---------------------------
        raw_bytes += np.float64(self.swing_start_angle_in_rad).tobytes()
        raw_bytes += np.float64(self.swing_angular_step).tobytes()

        # --------------------------------docmentation ex-----------------------
        raw_bytes += self._prefilter.encode()
        raw_bytes += np.uint64(self.file_time).tobytes()
        raw_bytes += np.uint32(self.header_content).tobytes()
        raw_bytes += np.asarray(self.reserved, dtype=np.int32).tobytes()

        # -------------------------------- recoparamex--------------------------
        raw_bytes += np.float32(self.reco_vol_min).tobytes()
        raw_bytes += np.float32(self.reco_vol_max).tobytes()
        raw_bytes += np.float32(self.reco_offset).tobytes()
        raw_bytes += np.int32(self.reco_max_z_slices).tobytes()
        raw_bytes += np.int32(self.reco_first_z_slice).tobytes()
        raw_bytes += np.int32(self.reco_last_z_slice).tobytes()
        raw_bytes += np.uint32(self.ct_algorithm).tobytes()
        raw_bytes += np.int32(self.ct_filter).tobytes()
        raw_bytes += np.int32(self.algorithm_platform).tobytes()
        raw_bytes += np.int32(self.projection_padding).tobytes()
        raw_bytes += np.float32(self.padding_object_radius).tobytes()
        raw_bytes += np.float32(self.filter_param_primary).tobytes()
        raw_bytes += np.asarray(self.recoparamex_reserved, dtype=np.int32).tobytes()

        # ---------------------------------- helix ------------------------------
        raw_bytes += np.float64(self.z_shift_per_projection_in_um).tobytes()
        raw_bytes += np.float64(self.scan_range_in_rad).tobytes()
        raw_bytes += np.int32(self.projections_per_z_shift).tobytes()
        raw_bytes += np.int32(self.helix_align).tobytes()

        # ---------------------------- arbitrary geometry -----------------------
        raw_bytes += self._agv_source_position.tobytes()
        raw_bytes += self._agv_source_direction.tobytes()
        raw_bytes += self._agv_detector_center_position.tobytes()
        raw_bytes += self._agv_detector_line_direction.tobytes()
        raw_bytes += self._agv_detector_col_direction.tobytes()
        raw_bytes += self._agv_reco_reference.tobytes()
        raw_bytes += self._agv_axis_angle.tobytes()

        # ----------------------------- range extension ------------------------
        raw_bytes += np.int32(self.range_ext_size_row).tobytes()
        raw_bytes += np.int32(self.range_ext_size_col).tobytes()
        raw_bytes += np.int32(self.multiscan).tobytes()
        raw_bytes += np.int16(self.range_ext_overlap_row).tobytes()
        raw_bytes += np.int16(self.range_ext_overlap_col).tobytes()

        # --------------------------------- mess ex ----------------------------
        raw_bytes += np.float64(self.detector_skew).tobytes()
        raw_bytes += np.float64(self.detector_tilt).tobytes()

        # ----------------------------------empty-------------------------------
        raw_bytes += self._empty_field.encode()

        # -----------------------------------platform---------------------------
        raw_bytes += np.uint32(self.endian).tobytes()
        raw_bytes += np.int32(self.reserved_field).tobytes()
        raw_bytes += np.uint64(self.endian64).tobytes()

        # -----------------------------------user-------------------------------
        raw_bytes += self._user_string.encode()

        if len(raw_bytes) != self.header_length:
            raise ValueError(f'wrong header length: {len(raw_bytes)} vs. {self.header_length}')

        return bytes(raw_bytes)

    @property
    def metadata(self) -> dict:
        """Current metadata information (official EZRT header fields only)."""
        metadata = {}
        metadata['measurement_name'] = self.measurement_name
        metadata['measurement_id'] = int(self.measurement_id)
        metadata['timestamp'] = f'{self.date} {self.time}'
        metadata['image_width'] = int(self.image_width)
        metadata['image_height'] = int(self.image_height)
        metadata['exposure_time_in_ms'] = self.exposure_time_in_ms
        metadata['voltage_in_kv'] = self.voltage_in_kv
        metadata['current_in_ua'] = self.current_in_ua
        metadata['focus_detector_distance_in_mm'] = float(self._focus_detector_distance_in_um) / 1000.0
        metadata['focus_object_distance_in_mm'] = float(self._focus_object_distance_in_um) / 1000.0
        metadata['frame_average'] = int(self.number_averages)
        metadata['frame_skip'] = int(self.number_skip_images)
        metadata['number_of_projections'] = int(self.number_projection_angles)
        metadata['acquisition_geometry'] = int(self.acquisition_geometry)
        metadata['detector_width_in_mm'] = float(self.detector_width_in_um) / 1000.0
        metadata['detector_height_in_mm'] = float(self.detector_height_in_um) / 1000.0
        metadata['prefilter'] = self.prefilter
        metadata['scan_range_in_rad'] = float(self.scan_range_in_rad)
        metadata['z_shift_per_projection_in_mm'] = float(self.z_shift_per_projection_in_um) / 1000.0
        metadata['projections_per_z_shift'] = int(self.projections_per_z_shift)
        metadata['agv_source_position'] = self.agv_source_position
        metadata['agv_source_direction'] = self.agv_source_direction
        metadata['agv_detector_center_position'] = self.agv_detector_center_position
        metadata['agv_detector_line_direction'] = self.agv_detector_line_direction
        metadata['agv_detector_col_direction'] = self.agv_detector_col_direction
        metadata['agv_reco_reference'] = self.agv_reco_reference
        metadata['agv_axis_angle'] = self.agv_axis_angle

        user_string = self.user_string
        if user_string != '':
            try:
                user_dict = loads(user_string)
                metadata.update(user_dict)
            except JSONDecodeError:
                metadata['additional_info'] = user_string

        return metadata

    def tofile(self, filename: Path | str):
        """Write header to a file in binary representation.

        :param filename: path and name of the file
        """
        with open(filename, 'wb') as fd:
            fd.write(self.tobytes())

    def add_to_buffer(self, input_array: bytes, overwrite_data: bool = False,
                      validate_if_ezrt_header: bool = True) -> bytes:
        """Adds or overwrites EZRT header to / of byte array / buffer.

        :param input_array: buffer to add header to
        :param overwrite_data: overwrite buffer data with header, beginning at byte 0
        :param validate_if_ezrt_header: validate if overwritten data is also a valid EZRT header
        :return: buffer with added header
        """
        if not overwrite_data:
            return self.tobytes() + input_array

        if len(input_array) < len(self):
            raise BufferError('input buffer not large enough to overwrite data')

        # parse buffer to see if the header creation succeeds
        if validate_if_ezrt_header:
            self.frombuffer(input_array)

        return self.tobytes() + input_array[len(self):]

    def add_to_file(self, filename: Path | str, overwrite_data: bool = False, validate_if_ezrt_header: bool = True):
        """Adds or overwrites EZRT header to / of file.

        :param filename: path and name of the file
        :param overwrite_data: overwrite buffer data with header, beginning at byte 0
        :param validate_if_ezrt_header: validate if overwritten data is also a valid EZRT header
        """
        with open(filename, 'rb+') as fd:
            buffer = fd.read()
            buffer = self.add_to_buffer(buffer, overwrite_data, validate_if_ezrt_header)
            fd.seek(0)
            fd.write(buffer)

    def parse_metadata(self, metadata: dict, use_custom_deepcopy: bool = True):
        # deep copy metadata to not change ingoing metadata object
        if use_custom_deepcopy:
            metadata = custom_deep_copy(metadata)
        else:
            metadata = deepcopy(metadata)

        if 'measurement_id' in metadata:
            self.measurement_id = int(metadata.pop('measurement_id'))
        if 'image_width' in metadata:
            self.image_width = self.number_horizontal_pixels = int(metadata.pop('image_width'))
        if 'image_height' in metadata:
            self.image_height = self.number_vertical_pixels = int(metadata.pop('image_height'))
            # ignore 5% of rows for reconstruction
            rows_to_ignore = self.image_height // 100 * 5
            self.reconstruction_start_line = rows_to_ignore
            self.reconstruction_end_line = self.image_height - rows_to_ignore
        if 'measurement_name' in metadata:
            measurement_name = str(metadata.pop('measurement_name'))
            if len(measurement_name) < 288:
                self._measurement_name = measurement_name + (288 - len(measurement_name)) * '\00'
        if 'timestamp' in metadata:
            timestamp = metadata.pop('timestamp').split(' ')
            if len(timestamp) == 2:
                if len(timestamp[0]) < 12:
                    self._date = timestamp[0] + (12 - len(timestamp[0])) * '\00'
                if len(timestamp[1]) < 8:
                    self._time = timestamp[1] + (8 - len(timestamp[1])) * '\00'
        if 'exposure_time_in_ms' in metadata:
            self._exposure_time_in_ms = int(float(metadata.pop('exposure_time_in_ms')))
        if 'voltage_in_kv' in metadata:
            # EZRT header takes voltage in V
            self._voltage_in_v = int(float(metadata.pop('voltage_in_kv'))) * 1000
        if 'current_in_ua' in metadata:
            self._current_in_ua = int(float(metadata.pop('current_in_ua')))
        if 'focus_detector_distance_in_mm' in metadata:
            self._focus_detector_distance_in_um = int(float(metadata.pop('focus_detector_distance_in_mm')) * 1000.0)
        if 'focus_object_distance_in_mm' in metadata:
            self._focus_object_distance_in_um = int(float(metadata.pop('focus_object_distance_in_mm')) * 1000.0)
        if 'frame_average' in metadata:
            self.number_averages = int(metadata.pop('frame_average'))
        if 'frame_skip' in metadata:
            self.number_skip_images = int(metadata.pop('frame_skip'))
        if 'number_of_projections' in metadata:
            self.number_projection_angles = int(metadata.pop('number_of_projections'))
        if 'acquisition_geometry' in metadata:
            self.acquisition_geometry = int(metadata.pop('acquisition_geometry'))
        if 'detector_width_in_mm' in metadata:
            self.detector_width_in_um = int(float(metadata.pop('detector_width_in_mm')) * 1000.0)
        if 'detector_height_in_mm' in metadata:
            self.detector_height_in_um = int(float(metadata.pop('detector_height_in_mm')) * 1000.0)
        if 'projections_per_z_shift' in metadata:
            self.projections_per_z_shift = int(metadata.pop('projections_per_z_shift'))
        if 'scan_range_in_rad' in metadata:
            self.scan_range_in_rad = float(metadata.pop('scan_range_in_rad'))
        if 'z_shift_per_projection_in_mm' in metadata:
            self.z_shift_per_projection_in_um = float(metadata.pop('z_shift_per_projection_in_mm')) * 1000.0
        if 'prefilter' in metadata:
            prefilter = str(metadata.pop('prefilter'))
            if len(prefilter) < 32:
                self._prefilter = prefilter + (32 - len(prefilter)) * '\00'
        if 'agv_source_position' in metadata:
            agv_source_position = tuple(metadata.pop('agv_source_position'))
            if len(agv_source_position) != 3:
                raise ValueError('wrong input vector length')
            self._agv_source_position = np.array(agv_source_position, dtype=np.float64)
        if 'agv_source_direction' in metadata:
            agv_source_direction = tuple(metadata.pop('agv_source_direction'))
            if len(agv_source_direction) != 3:
                raise ValueError('wrong input vector length')
            self._agv_source_direction = np.array(agv_source_direction, dtype=np.float64)
        if 'agv_detector_center_position' in metadata:
            agv_detector_center_position = tuple(metadata.pop('agv_detector_center_position'))
            if len(agv_detector_center_position) != 3:
                raise ValueError('wrong input vector length')
            self._agv_detector_center_position = np.array(agv_detector_center_position, dtype=np.float64)
        if 'agv_detector_line_direction' in metadata:
            agv_detector_line_direction = tuple(metadata.pop('agv_detector_line_direction'))
            if len(agv_detector_line_direction) != 3:
                raise ValueError('wrong input vector length')
            self._agv_detector_line_direction = np.array(agv_detector_line_direction, dtype=np.float64)
        if 'agv_detector_col_direction' in metadata:
            agv_detector_col_direction = tuple(metadata.pop('agv_detector_col_direction'))
            if len(agv_detector_col_direction) != 3:
                raise ValueError('wrong input vector length')
            self._agv_detector_col_direction = np.array(agv_detector_col_direction, dtype=np.float64)
        if 'agv_reco_reference' in metadata:
            agv_reco_reference = tuple(metadata.pop('agv_reco_reference'))
            if len(agv_reco_reference) != 3:
                raise ValueError('wrong input vector length')
            self._agv_reco_reference = np.array(agv_reco_reference, dtype=np.float64)
        if 'agv_axis_angle' in metadata:
            agv_axis_angle = tuple(metadata.pop('agv_axis_angle'))
            if len(agv_axis_angle) != 4:
                raise ValueError('wrong input vector length')
            self._agv_axis_angle = np.array(agv_axis_angle, dtype=np.float64)

        # write rest of metadata in "user_string" field
        user_string = dumps(metadata)
        user_string_length = len(user_string)
        if user_string_length < 1024:
            self._user_string = user_string + (1024 - user_string_length) * '\00'
