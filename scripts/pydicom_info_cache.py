#!/usr/bin/env python3

import pydicom
from pathlib import Path

pydicom_read_cache_static_filename_dict: dict[str, pydicom.Dataset] = dict()


def pydicom_read_cache(
    filename: Path | str, stop_before_pixels=True
) -> pydicom.Dataset:
    """
    Reads a DICOM file header and caches the result to improve performance on subsequent reads.

    Args:
        filename: The path to the DICOM file to be read.
        stop_before_pixels: If True, stops reading before pixel data (default: True).
    Returns:
        (pydicom.Dataset): A pydicom.Dataset containing the DICOM file's header data.
    """

    global pydicom_read_cache_static_filename_dict
    lookup_filename: str = str(filename)
    if lookup_filename in pydicom_read_cache_static_filename_dict:
        # print(f"Using cached value for {lookup_filename}")
        pass
    else:
        pydicom_read_cache_static_filename_dict[lookup_filename] = pydicom.dcmread(
            lookup_filename, stop_before_pixels=stop_before_pixels, force=True
        )
    return pydicom_read_cache_static_filename_dict.get(lookup_filename)
