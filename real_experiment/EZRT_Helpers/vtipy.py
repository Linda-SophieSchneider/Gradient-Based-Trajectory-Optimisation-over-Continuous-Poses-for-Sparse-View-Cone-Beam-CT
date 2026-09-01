# FRAUNHOFER IIS CONFIDENTIAL
# __________________
#
# Fraunhofer IIS
# Copyright (c) 2015-2021
# All Rights Reserved.
#
# This file is part of the PythonTools project (used to be part of the FileIO project).
#
# NOTICE:  All information contained herein is, and remains the property of Fraunhofer IIS and its suppliers, if any.
# The intellectual and technical concepts contained herein are proprietary to Fraunhofer IIS and its suppliers and may
# be covered by German and Foreign Patents, patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is strictly forbidden unless prior written
# permission is obtained from Fraunhofer IIS.


from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except ModuleNotFoundError:
    raise ModuleNotFoundError('to use vtipy, the vtk module must be installed via "pip install vtk"')


# mapping from np.dtype to vtk dtype
VTK_TYPE_BY_NUMPY_TYPE = {
    np.uint8: vtk.VTK_UNSIGNED_CHAR,
    np.uint16: vtk.VTK_UNSIGNED_SHORT,
    np.uint32: vtk.VTK_UNSIGNED_INT,
    np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
    np.int8: vtk.VTK_CHAR,
    np.int16: vtk.VTK_SHORT,
    np.int32: vtk.VTK_INT,
    np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
    np.float32: vtk.VTK_FLOAT,
    np.float64: vtk.VTK_DOUBLE
}


def py2vti(volume_array: np.ndarray, filepath: Path | str, image_data_writer_callback: Callable | None = None):
    """
    Save an internal Python volume representation (3-dim np array) as a .vti volume file.

    :param volume_array: 3D numpy input array to save
    :param filepath: file path where new .rek file is saved
    :param image_data_writer_callback: callback to manipulate vtkXMLImageDataWriter options; prototype: func(writer)
    """
    # vtk wants 3d
    if len(volume_array.shape) != 3:
        raise ValueError(f'input volume array has wrong dimensions ({len(volume_array.shape)} vs. 3)')

    # numpy to vtk
    try:
        vtk_array_type = VTK_TYPE_BY_NUMPY_TYPE[volume_array.dtype.type]
    except Exception:
        raise ValueError(f'datatype {volume_array.dtype.type} not supported')

    depthArray = numpy_to_vtk(volume_array.ravel(), deep=False, array_type=vtk_array_type)

    vtk_image_data = vtk.vtkImageData()
    # fill the vtk image data object
    vtk_image_data.SetDimensions(volume_array.shape[::-1])
    vtk_image_data.SetSpacing([1, ] * volume_array.ndim)
    vtk_image_data.SetOrigin([0, ] * volume_array.ndim)
    vtk_image_data.GetPointData().SetScalars(depthArray)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputData(vtk_image_data)

    # callback to manipulate vtkXMLImageDataWriter options
    if image_data_writer_callback is not None:
        image_data_writer_callback(writer)

    writer.Write()


def vti2py(filepath: Path | str) -> np.ndarray:
    """
    Load VTI file as numpy array.

    :param filepath: file path to .vti file
    :return: 2D numpy array representation of .vti file
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f'given path is not a file [{filepath}]')

    # read file
    data_reader = vtk.vtkXMLImageDataReader()
    data_reader.SetFileName(str(filepath))
    data_reader.Update()
    vtk_image = data_reader.GetOutput()

    # convert file to numpy representation
    vtk_scalars = vtk_image.GetPointData().GetScalars()
    shape = tuple(vtk_image.GetDimensions())[::-1]
    volume_array = vtk_to_numpy(vtk_scalars).reshape(shape)

    return volume_array
