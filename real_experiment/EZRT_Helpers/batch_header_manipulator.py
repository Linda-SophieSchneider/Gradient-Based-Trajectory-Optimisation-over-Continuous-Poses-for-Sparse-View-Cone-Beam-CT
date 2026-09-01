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
from typing import Callable

import numpy as np

from .common_types import IMAGEFORMATS
from .ezrt_header import EzrtHeader
from .raw2py import raw2py
from .py2raw import py2raw
from .imageio import ImageConverter


class BatchHeaderManipulator:
    """Add, read and manipulate the EZRT header to/of files in a folder."""

    def __init__(self, folderpath: Path | str, file_pattern: str = '*.raw', ezrt_header: EzrtHeader | None = None,
                 verbose: bool = False):
        """
        Construct new BatchHeaderManipulator instance from files with request suffix in given path.

        :param folderpath: folder in which files with suffix are searched
        :param file_pattern: file pattern to search for (see glob)
        :param ezrt_header: header to add or overwrite file with (optional for FhG formats; if given in this case,
                           this input header is used instead of the files own header)
        :param verbose: print debug output
        """
        self.folderpath = Path(folderpath)
        self.headers: list[EzrtHeader] = []
        self.file_pattern: str = file_pattern
        self.verbose: bool = verbose

        self._files_with_suffix_in_path: list[Path] = []
        self._has_ezrt_header = file_pattern.find('.raw') > 0 or file_pattern.find('.rek') > 0

        self.add_files_from_folder(folderpath, ezrt_header)

    def __str__(self):
        return f'--------- Header @ path {self.folderpath}----------\n\n'\
            + f'# files with header : {len(self.headers)}\n\n'\
            + 'Header 0:\n'\
            + f'{self.headers[0]}' if len(self.headers) > 0 else ''

    def add_files_from_folder(self, folderpath: Path | str, ezrt_header: EzrtHeader | None = None):
        """
        Fetch all files with EzrtHeader and with correct suffix in the path folder.

        :param folderpath: folder in which files with EZRT header are searched
        :param ezrt_header: header to add or overwrite file with (optional for FhG formats; if given in this case,
                           this input header is used instead of the files own header)
        """
        self.folderpath = Path(folderpath)

        if not self.folderpath.exists():
            raise NotADirectoryError('directory not found')
        if not self.folderpath.is_dir():
            raise NotADirectoryError('"folderpath" should be a directory')

        self._files_with_suffix_in_path = []

        for item in sorted(self.folderpath.glob(self.file_pattern)):
            self._files_with_suffix_in_path.append(item)

        if len(self._files_with_suffix_in_path) == 0:
            raise Exception(f'no projections with pattern "{self.file_pattern}" in folder {self.folderpath}')

        self.headers = []
        if ezrt_header is None and not self._has_ezrt_header:
            return

        for file_path in self._files_with_suffix_in_path:
            if ezrt_header is not None:
                header = ezrt_header
            else:
                header = EzrtHeader.fromfile(file_path)
            self.headers.append(header)

    def get_attribute_by_index(self, attribute: str, header_index: int):
        """
        Get stated attribute value of the header at the given index.

        :param attribute: name of header attribute
        :return: attribute value
        """
        if header_index >= len(self.headers):
            raise ValueError('no header loaded or header index out of range')

        return getattr(self.headers[header_index], attribute)

    def change_attribute_by_index(self, attribute: str, value: int | float | str, header_index: int,
                                  save_headers: bool = False):
        """
        Change stated attribute value of the header at the given index.

        :param attribute: name of header attribute
        :param header_num: specifies the header in the self.header list
        :param value: the value, which gets assigned.
        :param save_headers: save the headers to hdd.
        """
        if header_index >= len(self.headers):
            raise ValueError('no header loaded or header index out of range')
        if not hasattr(self.headers[header_index], attribute):
            raise AttributeError

        setattr(self.headers[header_index], attribute, value)

        if save_headers:
            self.save()

    def change_attribute_all(self, attribute: str, value: int | float | str, save_headers: bool = False):
        """
        Change stated attribute values of all loaded header.

        :param attribute: name of header attribute
        :param value: the value, which gets assigned.
        :param save_headers: save the headers to hdd.
        """
        if len(self.headers) == 0:
            raise ValueError('no header loaded, thus no attribute can be changed')

        for header in self.headers:
            if not hasattr(header, attribute):
                raise AttributeError
            setattr(header, attribute, value)

        if save_headers:
            self.save()

    def get_attribute_all(self, attribute: str) -> list:
        """
        Get stated attribute values of all loaded header as list.

        :param attribute: name of header attribute
        :return: list of the specified attribute from all headers from th project
        """
        if len(self.headers) == 0:
            raise ValueError('no header loaded, thus no attribute can be retrieved')

        return_array = []
        for header in self.headers:
            return_array.append(getattr(header, attribute))

        return return_array

    def save(self, overwrite: bool = True, new_folder_path: Path | str = 'files_with_header'):
        """
        Save all files with adjusted header (either overwrite existing file or use new folder).

        :param overwrite: if True overwrites loaded files, if False a new folder is created to store manipulated
                          files
        :param new_folder_path: path of new folder, if overwrite is False
        """
        if len(self.headers) != len(self._files_with_suffix_in_path):
            raise ValueError('not enough header loaded to add to files')

        if overwrite:
            for header, filepath in zip(self.headers, self._files_with_suffix_in_path):
                header.add_to_file(filepath, overwrite_data=self._has_ezrt_header)
            return

        if new_folder_path == '':
            raise ValueError('new folder path not specified')

        new_folder_path = Path(new_folder_path)
        new_folder_path.mkdir(parents=True, exist_ok=True)

        for header, filepath in zip(self.headers, self._files_with_suffix_in_path):
            with open(filepath, 'rb') as infile:
                filecontent = header.add_to_buffer(infile.read(), overwrite_data=self._has_ezrt_header)
                with open(new_folder_path / filepath.name, 'wb') as outfile:
                    outfile.write(filecontent)


class BatchProjectionManipulator(BatchHeaderManipulator):
    """Read, manipulate and save projections and the corresponding EZRT header in a folder."""

    def __init__(self, folderpath: Path | str, load_projection_data: bool = False, dtype=np.uint16,
                 verbose: bool = False):
        """
        Construct new BatchProjectionManipulator instance from files with request suffix in given path.

        :param folderpath: specifies the folder in which files with EZRT header are searched
        :param load_projection_data: state if projection data should be loaded to memory (needs a lot more memory) or
                                     read on demand (projection need to be loaded for every operation => more CPU load).
                                     Does only apply for image data, headers are always loaded.
        :param dtype: target data types of projections
        :param verbose: print debug output
        """
        super().__init__(folderpath, '*.raw', verbose=verbose)

        self.dtype = dtype
        self.projection_data = []
        self._projections_loaded = False

        if load_projection_data:
            self.load_projection_data()

        if self.verbose:
            print('Loaded all projections ...\n')

    def __str__(self):
        return f'--------- Projections @ path {self.folderpath}----------\n\n' \
               + f'# projections : {len(self.headers)}\n\n' \
               + 'Projection information of header 0:\n\n' \
               + f'{self.headers[0]}' if len(self.headers) > 0 else ''

    def load_projection_data(self):
        """Load all projection data specified in the projection_path variable."""
        for projection_path in self._files_with_suffix_in_path:
            self.projection_data.append(raw2py(projection_path)[1])
        self._projections_loaded = True

    @property
    def projection_data_iterator(self):
        for projection_path in self._files_with_suffix_in_path:
            yield raw2py(projection_path)[1]

    def execute_function_by_index(self, index: int, target: Callable, args: tuple = (), kwargs: dict = {},
                                  save: bool = False, overwrite: bool = False):
        """
        Register and execute function on projection and header at the given index..

            Prototype: func(header: EzrtHeader, image: np.ndarray, *args, **kwargs) -> EzrtHeader, np.ndarray

        If projections are not loaded to memory, the modified output is directly saved to disk. Use the overwrite flag
        to specify if data should be overwritten. Attention: if data is not overwritten, all modifications after the
        function call is done on the original (i.e. non-modified) image data - the modified data is saved in a new
        directory (named after the function) in the original file path.

        :param target: target function to call
        :param args: positional arguments of passed function as tuple
        :param kwargs: keyword arguments of passed function as dict
        :param save: flag if projections should be directly saved (mandatory if projections are not loaded beforehand)
        :param overwrite: flag if projections should be overwritten
        """
        if not self._projections_loaded and not save:
            raise ValueError('no projections are loaded and saving is disable - invalid setting, changes would be lost')

        if self._projections_loaded:
            # execute function (copy projection data, to avoid read-only issue)
            header, projection_data = target(self.headers[index], np.copy(self.projection_data[index]), *args, **kwargs)
            self.headers[index] = header
            self.projection_data[index] = projection_data
        else:
            for i, (modified_header, modified_image) in enumerate(self.iterate_function(target, args=args,
                                                                                        kwargs=kwargs, save=False,
                                                                                        overwrite=overwrite)):
                if i != index:
                    continue
                header, projection_data = modified_header, modified_image
                break
            else:
                raise IndexError('index out of range')

        if save:
            function_name = getattr(target, '__name__', 'unknown')

            if save and not overwrite:
                (self.folderpath / function_name).mkdir(parents=True, exist_ok=True)

            filepath = self._files_with_suffix_in_path[index]
            if not overwrite:
                filepath = filepath.parent / function_name / filepath.name

            py2raw(projection_data.astype(self.dtype), filepath, input_header=header)

    def execute_function_all(self, target: Callable, args: tuple = (), kwargs: dict = {}, save: bool = False,
                             overwrite: bool = False):
        """
        Register and execute function on all projections and headers.

            Prototype: func(header: EzrtHeader, image: np.ndarray, *args, **kwargs) -> EzrtHeader, np.ndarray

        If projections are not loaded to memory, the modified output is directly saved to disk. Use the overwrite flag
        to specify if data should be overwritten. Attention: if data is not overwritten, all modifications after the
        function call is done on the original (i.e. non-modified) image data - the modified data is saved in a new
        directory (named after the function) in the original file path.

        :param target: target function to call
        :param args: positional arguments of passed function as tuple
        :param kwargs: keyword arguments of passed function as dict
        :param save: flag if projections should be directly saved (mandatory if projections are not loaded beforehand)
        :param overwrite: flag if projections should be overwritten
        """
        if not self._projections_loaded and not save:
            raise ValueError('no projections are loaded and saving is disable - invalid setting, changes would be lost')
        for i, (header, image) in enumerate(self.iterate_function(target, args=args, kwargs=kwargs, save=save,
                                                                  overwrite=overwrite)):
            if not self._projections_loaded:
                continue

            self.headers[i] = header
            self.projection_data[i] = image

    def iterate_function(self, target: Callable, args: tuple = (), kwargs: dict = {}, save: bool = False,
                         overwrite: bool = False):
        """
        Register and execute function on projections and headers via generator.

            Prototype: func(header: EzrtHeader, image: np.ndarray, *args, **kwargs) -> EzrtHeader, np.ndarray

        If projections are not loaded to memory, the modified output is directly saved to disk. Use the overwrite flag
        to specify if data should be overwritten. Attention: if data is not overwritten, all modifications after the
        function call is done on the original (i.e. non-modified) image data - the modified data is saved in a new
        directory (named after the function) in the original file path.

        :param func: function to call
        :param args: positional arguments of passed function
        :param kwargs: keyword arguments of passed function
        :param save: flag if projections should be directly saved
        :param overwrite: flag if projections should be overwritten
        :yields: current header and projection data
        """
        function_name = getattr(target, '__name__', 'unknown')

        if save and not overwrite:
            (self.folderpath / function_name).mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f'Execute {function_name} on all projections / header')
            print('[', end='', flush=True)

        if self._projections_loaded:
            projection_data = self.projection_data
        else:
            projection_data = self.projection_data_iterator

        for header, projection_data, filepath in zip(self.headers, projection_data, self._files_with_suffix_in_path):
            if self.verbose:
                print('>', end='', flush=True)

            # execute function (copy projection data, to avoid read-only issue)
            header, projection_data = target(header, np.copy(projection_data), *args, **kwargs)
            if save:
                if not overwrite:
                    filepath = filepath.parent / function_name / filepath.name
                py2raw(projection_data.astype(self.dtype), filepath, input_header=header)

            yield header, projection_data

        if self.verbose:
            print(']')

    def save(self, overwrite: bool = True, new_folder_path: Path | str = 'files_with_header',
             projection_prefix: str = 'projection'):
        """Save all projections with adjusted header (either overwrite existing projection or use new folder).

        :param overwrite: if True overwrites loaded projections, if False a new folder is created to store manipulated
                          projections
        :param folder_name: name of new folder, if overwrite is False
        :param projection_prefix: prefix of newly saved projections (if overwrite is False)
        """
        if overwrite:
            for projection_data, header, filepath in zip(self.projection_data, self.headers,
                                                         self._files_with_suffix_in_path):
                py2raw(projection_data.astype(self.dtype), filepath, input_header=header)
            return

        if new_folder_path == '':
            raise ValueError('new folder path not specified')

        new_folder_path = Path(new_folder_path)
        new_folder_path.mkdir(parents=True, exist_ok=True)

        for i, (projection_data, header, filepath) in enumerate(zip(self.projection_data, self.headers,
                                                                    self._files_with_suffix_in_path)):
            save_path = new_folder_path / f'{projection_prefix}_{i:04d}.raw'
            py2raw(projection_data.astype(self.dtype), save_path, input_header=header)


class BatchImageConverter(BatchHeaderManipulator):
    """
    Convert all image or projection files in the folder with the stated suffix to the image format given.
    If the EZRT projection format is the desired output, the header is attached.
    Supported image formats can be seen in IMAGEFORMATS enum.
    """

    def __init__(self, folderpath: Path | str, file_pattern: str = '*.tif', ezrt_header: EzrtHeader | None = None,
                 imageformat_out: IMAGEFORMATS = IMAGEFORMATS.RAW_EZRT, load_metadata: bool = False,
                 verbose: bool = False):
        """
        Construct new BatchHeaderImageConverter instance from files with request suffix in given path.

        :param folderpath: folder in which files with suffix are searched
        :param file_pattern: file pattern to search for (see glob)
        :param ezrt_header: header to add or overwrite file with (optional for FhG formats; if given in this case,
                           this input header is used instead of the files own header)
        :param load_metadata: flag if metadata should be loaded (if not an empty dict is returned)
        :param verbose: print debug output
        """
        super().__init__(folderpath, file_pattern, ezrt_header, verbose)
        self.imageformat_out = imageformat_out
        self.load_metadata = load_metadata

    def save(self, new_folder_path: Path | str | None = None):
        """
        Save all files with adjusted header and in new format (either in source or in new folder)

        :param new_folder_path: target folder path (if None converted images are saved in the same folder as source)
        """
        if new_folder_path is None:
            new_folder_path = self.folderpath
        else:
            new_folder_path = Path(new_folder_path)

        new_folder_path.mkdir(parents=True, exist_ok=True)

        for i, filepath in enumerate(self._files_with_suffix_in_path):
            header = self.headers[i] if len(self.headers) > i else None
            output_path = new_folder_path / filepath.stem
            ImageConverter.convert_image(filepath, output_path, IMAGEFORMATS.AUTO, self.imageformat_out,
                                         ezrt_header=header, load_metadata=self.load_metadata)
