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

import re
from pathlib import Path

from .ezrt_header import EzrtHeader


def get_postfix_number(filename: Path | str):
    if isinstance(filename, Path):
        filename = filename.name
    return int(re.findall(r'_[0-9]+', filename)[0][1:])


def get_folder_state(folderpath: Path | str, fileprefix: str, raise_exception: bool = False) -> tuple[int, int]:
    """
    Find how many projections are already acquired and how many are planned by analyzing the measurement folder.

    If no valid folder items are found or if the folder is no valid dir 0, 0 is returned (or an exception is thrown
    if raise_exception=True).

    :param folderpath: path to folder
    :param fileprefix: prefix of projection files
    :param raise_exception: if True, Exceptions are thrown instead of the 0, 0 return value in an error case
    :return: number of projections found, number of planned projections
    """
    folderpath = Path(folderpath)

    if not folderpath.is_dir():
        if raise_exception:
            raise NotADirectoryError('given folderpath is not a valid directory')
        return 0, 0

    filtered_filenames = list(filter(lambda filename: re.match(fileprefix + r'_[0-9]+', filename.name),
                                     folderpath.iterdir()))
    sorted_filtered_filenames = sorted(filtered_filenames, key=get_postfix_number)
    number_of_projections_found = len(sorted_filtered_filenames)

    if number_of_projections_found <= 0:
        return 0, 0

    max_file_index_found = get_postfix_number(sorted_filtered_filenames[-1])
    if number_of_projections_found != max_file_index_found + 1:
        if raise_exception:
            raise Exception('found projections in directory are not numbered consecutively')
        return 0, 0

    header = EzrtHeader.fromfile(sorted_filtered_filenames[-1])
    return number_of_projections_found, header.number_projection_angles
