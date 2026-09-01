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

import xml.etree.ElementTree as ET

from xml.etree.ElementTree import Element
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DetectorParameter:
    total_scan_time_in_sec: float = 0.0
    number_averages: int = 0
    number_skip_images: int = 0
    do_dark_correction: bool = False
    do_mid_correction: bool = False
    do_bright_correction: bool = False
    do_bpm_correction: bool = False
    label: str = ''
    capacity: int = 0
    horizontal_shift_in_px: float = 0.0
    vertical_shift_in_px: float = 0.0
    number_horizontal_pixels: int = 0
    number_vertical_pixels: int = 0
    detector_tilt: float = 0.0
    pixel_pitch_x_in_um: float = 0.0
    pixel_pitch_y_in_um: float = 0.0
    frame_rate: float = 0.0


@dataclass
class ManipulatorParameter:
    label: str = ''
    focus_detector_distance_in_um: float = 0.0
    focus_object_distance_in_um: float = 0.0
    # TODO: list


@dataclass
class XrayParameter:
    label: str = ''
    voltage_in_kv: float = 0.0
    current_in_ua: float = 0.0
    focus: int = 0
    exposure_time_in_ms: float = 0.0


@dataclass
class MfeParameter:
    mfe_horizontal: int = 0
    overlap_mfe_horizontal: int = 0
    balance_method_mfe_horizontal: int = 0
    mfe_vertical: int = 0
    overlap_mfe_vertical: int = 0
    balance_method_mfe_vertical: int = 0
    inactive_columns: int = 0
    inactive_rows: int = 0
    multi_scan: int = 0


@dataclass
class CtParameter:
    measurement_name: str = ''
    measurement_path: str | Path = ''
    reconstruction_path: str | Path = ''
    number_projection_angles: int = 0
    scan_range_in_deg: float = 0.0
    volume_type: str = ''
    reco_first_z_slice: int = 0
    reco_last_z_slice: int = 0
    ctos_procedure: str = ''
    save_projections: bool = False
    reco_columns: int = 0
    reco_rows: int = 0
    inull_value: int = 0
    inull_x_position: int = 0
    inull_y_position: int = 0
    inull_delta_x: int = 0
    inull_delta_y: int = 0
    inull_align: int = 0
    do_online_reco: bool = False
    do_median_filter: bool = False
    do_artefact_reduction: bool = False
    do_correct_beamhardening: bool = False
    beam_hardening_files: str | Path = ''
    reduced_detector_width: int = 0
    roi_start_x: int = 0
    roi_start_y: int = 0
    roi_count_x: int = 0
    roi_count_y: int = 0
    min_aoi_intensity: int = 0
    max_aoi_diff_intensity: int = 0
    reconstruction_start_line: int = 0
    reconstruction_end_line: int = 0
    swing_start_angle_in_deg: float = 0.0
    swing_end_angle_in_deg: float = 0.0
    scale_unit_min: float = 0.0
    scale_unit_max: float = 0.0
    rebinning: int = 0
    acquisition_mode: str = ''


@dataclass
class ContinuableParameter:
    used_cone_beam: float = 0.0
    reconstruction_start_point: float = 0.0
    reconstruction_end_point: float = 0.0
    short_or_full_scan: bool = False
    projections_per_z_shift: int = 0
    z_shift_per_projection_in_um: float = 0.0
    object_radius: float = 0.0
    ct_algorithm: str = ''
    ct_filter: str = ''
    algorithm_platform: str = ''
    projection_padding: str = ''
    ct_filter_length: int = 0
    ct_parameter: CtParameter = CtParameter()


class VolexMetaData:
    """EZRT Volex XML data / parser class"""

    def __init__(self, image_width: int = 0, image_height: int = 0):
        """
        Construct new VolexMetaData instance with given parameters.

        :param image_width: image width
        :param image_height: image height
        """
        if image_width < 0 or image_height < 0:
            raise ValueError('image dimensions must be > 0')

        self.detector = DetectorParameter(number_horizontal_pixels=image_width, number_vertical_pixels=image_height)
        self.manipulator = ManipulatorParameter()
        self.xray = XrayParameter()
        self.continuable_parameter = ContinuableParameter()
        self.mfe = MfeParameter()

    @classmethod
    def fromfile(cls, filename: Path | str) -> VolexMetaData:
        """
        Create new VolexMetaData instance from file

        :param filename: path to file
        """
        xml_string = ''
        with open(filename, 'r') as fd:
            xml_string = fd.read()

        return cls.fromstring(xml_string)

    @classmethod
    def fromstring(cls, xml_string: str) -> VolexMetaData:
        """
        Create new VolexMetaData instance from XML string.

        :param buffer: string with XML
        """
        header = cls()
        header._parse_xml(xml_string)
        return header

    def _parse_xml(self, xml_string: str):
        root = ET.fromstring(xml_string)
        if len(root) < 1:
            raise Exception('XML invalid - does not contain data')

        procedure_param_container = root[0]
        if procedure_param_container.tag != 'object':
            raise Exception('XML invalid - content does not match the expectations')

        for child in procedure_param_container:
            if child.tag != 'object' or 'propertyname' not in child.attrib:
                continue

            propertyname = child.attrib['propertyname']
            if propertyname == 'Messeinstellungen':
                for sub_child in child:
                    if sub_child.tag != 'object' or 'type' not in sub_child.attrib:
                        continue

                    type_id = sub_child.attrib['type']

                    if 'DetectorParameterClass' in type_id:
                        self._parse_detector_parameter(sub_child)
                    elif 'ManipulatorParameterClass' in type_id:
                        self._parse_manipulator_parameter(sub_child)
                    elif 'MFEParameterClass' in type_id:
                        self._parse_mfe_parameter(sub_child)
                    elif 'XRayParameterClass' in type_id:
                        self._parse_xray_parameter(sub_child)

            elif propertyname == 'Messergebnisse':
                for sub_child in child:
                    if sub_child.tag != 'object' or 'type' not in sub_child.attrib:
                        continue

                    type_id = sub_child.attrib['type']

                    if 'ContinuableParameterClass' in type_id:
                        self._parse_continuable_parameter(sub_child)

    def _parse_xray_parameter(self, xml_element: Element):
        for child in xml_element:
            if 'propertyname' not in child.attrib or 'value' not in child.attrib:
                continue

            property_name = child.attrib['propertyname']
            value = child.attrib['value']

            if property_name == 'kV':
                self.xray.voltage_in_kv = float(value)
            elif property_name == 'uA':
                self.xray.current_in_ua = float(value)
            elif property_name == 'Focus':
                self.xray.focus = int(value)
            elif property_name == 'ExposureTime':
                self.xray.exposure_time_in_ms = float(value)
            elif property_name == 'Label':
                self.xray.label = value

    def _parse_mfe_parameter(self, xml_element: Element):
        for child in xml_element:
            if 'propertyname' not in child.attrib or 'value' not in child.attrib:
                continue

            property_name = child.attrib['propertyname']
            value = child.attrib['value']

            if property_name == 'MFEHorizontal':
                self.mfe.mfe_horizontal = int(value)
            elif property_name == 'OverlapMFEHorizontal':
                self.mfe.overlap_mfe_horizontal = int(value)
            elif property_name == 'BalanceMethodMFEHorizontal':
                self.mfe.balance_method_mfe_horizontal = int(value)
            elif property_name == 'MFEVertical':
                self.mfe.mfe_vertical = int(value)
            elif property_name == 'OverlapMFEVertical':
                self.mfe.overlap_mfe_vertical = int(value)
            elif property_name == 'BalanceMethodMFEVertical':
                self.mfe.balance_method_mfe_vertical = int(value)
            elif property_name == 'InactiveColumns':
                self.mfe.inactive_columns = int(value)
            elif property_name == 'InactiveRows':
                self.mfe.inactive_rows = int(value)
            elif property_name == 'MultiScan':
                self.mfe.multi_scan = int(value)

    def _parse_manipulator_parameter(self, xml_element: Element):
        for child in xml_element:
            if 'propertyname' not in child.attrib or 'value' not in child.attrib:
                continue

            property_name = child.attrib['propertyname']
            value = child.attrib['value']

            if property_name == 'FOD':
                self.manipulator.focus_object_distance_in_um = float(value)
            elif property_name == 'FDD':
                self.manipulator.focus_detector_distance_in_um = float(value)
            elif property_name == 'Label':
                self.manipulator.label = value

    def _parse_detector_parameter(self, xml_element: Element):
        for child in xml_element:
            if 'propertyname' not in child.attrib or 'value' not in child.attrib:
                continue

            property_name = child.attrib['propertyname']
            value = child.attrib['value']

            if property_name == 'TotalScanTime':
                self.detector.total_scan_time_in_sec = float(value)
            elif property_name == 'SkipFrames':
                self.detector.number_skip_images = int(value)
            elif property_name == 'DoDarkCorrection':
                self.detector.do_dark_correction = bool(int(value))
            elif property_name == 'DoMidCorrection':
                self.detector.do_mid_correction = bool(int(value))
            elif property_name == 'DoBrightCorrection':
                self.detector.do_bright_correction = bool(int(value))
            elif property_name == 'DoBadPixelCorrection':
                self.detector.do_bpm_correction = bool(int(value))
            elif property_name == 'AverageCount':
                self.detector.number_averages = int(value)
            elif property_name == 'Label':
                self.detector.label = value
            elif property_name == 'Capacity':
                self.detector.capacity = int(value)
            elif property_name == 'ShiftHorizontal':
                self.detector.horizontal_shift_in_px = float(value)
            elif property_name == 'ShiftVertical':
                self.detector.vertical_shift_in_px = float(value)
            elif property_name == 'TiltAngle':
                self.detector.detector_tilt = float(value)
            elif property_name == 'PixelsX':
                self.detector.number_vertical_pixels = int(value)
            elif property_name == 'PixelsY':
                self.detector.number_horizontal_pixels = int(value)
            elif property_name == 'PitchX':
                self.detector.pixel_pitch_x_in_um = float(value)
            elif property_name == 'PitchY':
                self.detector.pixel_pitch_y_in_um = float(value)
            elif property_name == 'FrameRate':
                self.detector.frame_rate = float(value)

    def _parse_continuable_parameter(self, xml_element: Element):
        for child in xml_element:
            if 'propertyname' not in child.attrib:
                continue

            property_name = child.attrib['propertyname']
            value = ''
            if 'value' in child.attrib:
                value = child.attrib['value']

            if property_name == 'ProjectionsPerZShift':
                self.continuable_parameter.projections_per_z_shift = int(value)
            elif property_name == 'UsedConeBeam':
                self.continuable_parameter.used_cone_beam = float(value)
            elif property_name == 'ReconstructionStartPoint':
                self.continuable_parameter.reconstruction_start_point = int(value)
            elif property_name == 'ReconstructionEndPoint':
                self.continuable_parameter.reconstruction_end_point = int(value)
            elif property_name == 'ShortOrFullScan':
                self.continuable_parameter.short_or_full_scan = bool(int(value))
            elif property_name == 'ZShift':
                self.continuable_parameter.z_shift_per_projection_in_um = float(value)
            elif property_name == 'ObjectRadius':
                self.continuable_parameter.object_radius = float(value)
            elif property_name == 'CTProjectionPadding':
                self.continuable_parameter.projection_padding = value
            elif property_name == 'CTAlgorithm':
                self.continuable_parameter.ct_algorithm = value
            elif property_name == 'CTAlgorithmPlatform':
                self.continuable_parameter.algorithm_platform = value
            elif property_name == 'FilterType':
                self.continuable_parameter.ct_filter = value
            elif property_name == 'FilterLength':
                self.continuable_parameter.ct_filter_length = int(value)
            elif property_name == 'CTParameter':
                for sub_child in child:
                    if 'propertyname' not in sub_child.attrib or 'value' not in sub_child.attrib:
                        continue

                    property_name = sub_child.attrib['propertyname']
                    value = sub_child.attrib['value']

                    if property_name == 'MeasurementName':
                        self.continuable_parameter.ct_parameter.measurement_name = value
                    elif property_name == 'MeasurementPath':
                        self.continuable_parameter.ct_parameter.measurement_path = value
                    elif property_name == 'RekoPath':
                        self.continuable_parameter.ct_parameter.reconstruction_path = value
                    elif property_name == 'NumberOfProjections':
                        self.continuable_parameter.ct_parameter.number_projection_angles = int(value)
                    elif property_name == 'AngularScanRange':
                        self.continuable_parameter.ct_parameter.scan_range_in_deg = float(value)
                    elif property_name == 'VolType':
                        self.continuable_parameter.ct_parameter.volume_type = value
                    elif property_name == 'FirstSlice':
                        self.continuable_parameter.ct_parameter.reco_first_z_slice = int(value)
                    elif property_name == 'LastSlice':
                        self.continuable_parameter.ct_parameter.reco_last_z_slice = int(value)
                    elif property_name == 'CTOSProcedure':
                        self.continuable_parameter.ct_parameter.ctos_procedure = value
                    elif property_name == 'SaveProjections':
                        self.continuable_parameter.ct_parameter.save_projections = bool(int(value))
                    elif property_name == 'RekoColumns':
                        self.continuable_parameter.ct_parameter.reco_columns = int(value)
                    elif property_name == 'RekoRows':
                        self.continuable_parameter.ct_parameter.reco_rows = int(value)
                    elif property_name == 'INullValue':
                        self.continuable_parameter.ct_parameter.inull_value = int(value)
                    elif property_name == 'INullStartPosX':
                        self.continuable_parameter.ct_parameter.inull_x_position = int(value)
                    elif property_name == 'INullStartPosY':
                        self.continuable_parameter.ct_parameter.inull_y_position = int(value)
                    elif property_name == 'INullDeltaX':
                        self.continuable_parameter.ct_parameter.inull_delta_x = int(value)
                    elif property_name == 'INullDeltaY':
                        self.continuable_parameter.ct_parameter.inull_delta_y = int(value)
                    elif property_name == 'DoOnlineReko':
                        self.continuable_parameter.ct_parameter.do_online_reco = bool(int(value))
                    elif property_name == 'DoMedianFilter':
                        self.continuable_parameter.ct_parameter.do_median_filter = bool(int(value))
                    elif property_name == 'DoArtefactReduction':
                        self.continuable_parameter.ct_parameter.do_artefact_reduction = bool(int(value))
                    elif property_name == 'ReducedDetectorWidth':
                        self.continuable_parameter.ct_parameter.reduced_detector_width = int(value)
                    elif property_name == 'ROIStartX':
                        self.continuable_parameter.ct_parameter.roi_start_x = int(value)
                    elif property_name == 'ROIStartY':
                        self.continuable_parameter.ct_parameter.roi_start_y = int(value)
                    elif property_name == 'ROICountX':
                        self.continuable_parameter.ct_parameter.roi_count_x = int(value)
                    elif property_name == 'ROICountY':
                        self.continuable_parameter.ct_parameter.roi_count_y = int(value)
                    elif property_name == 'DoCorrectBeamHardening':
                        self.continuable_parameter.ct_parameter.do_correct_beamhardening = bool(int(value))
                    elif property_name == 'BeamHardeningFiles':
                        self.continuable_parameter.ct_parameter.beam_hardening_files = value
                    elif property_name == 'MinAOIIntensity':
                        self.continuable_parameter.ct_parameter.min_aoi_intensity = int(value)
                    elif property_name == 'MaxAOIDiffIntensity':
                        self.continuable_parameter.ct_parameter.max_aoi_diff_intensity = int(value)
                    elif property_name == 'FirstRow':
                        self.continuable_parameter.ct_parameter.reconstruction_start_line = int(value)
                    elif property_name == 'LastRow':
                        self.continuable_parameter.ct_parameter.reconstruction_end_line = int(value)
                    elif property_name == 'SwingStartAngle':
                        self.continuable_parameter.ct_parameter.swing_start_angle_in_deg = float(value)
                    elif property_name == 'SwingEndAngle':
                        self.continuable_parameter.ct_parameter.swing_end_angle_in_deg = float(value)
                    elif property_name == 'ScaleUnitMin':
                        self.continuable_parameter.ct_parameter.scale_unit_min = float(value)
                    elif property_name == 'ScaleUnitMax':
                        self.continuable_parameter.ct_parameter.scale_unit_max = float(value)
                    elif property_name == 'ReBinning':
                        self.continuable_parameter.ct_parameter.rebinning = int(value)
                    elif property_name == 'AcquisitionMode':
                        self.continuable_parameter.ct_parameter.acquisition_mode = value
