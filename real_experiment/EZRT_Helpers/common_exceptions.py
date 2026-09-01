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


class RosCtException(Exception):
    """RosCt exception"""


class DeviceException(Exception):
    """Device exception."""


class DeviceWarning(Warning):
    """Device warning."""


class DeviceInfo(Warning):
    """Device info."""


class ImageTimeoutException(Exception):
    """Image timeout exception."""


class ImageManagerException(Exception):
    """Image manager exception."""
