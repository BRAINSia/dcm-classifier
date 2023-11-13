from pathlib import Path

import numpy as np
from dicomanonymizer.simpledicomanonymizer import anonymize_dicom_file
from pydicom import dcmread


def anonymize_data(input_dir: str, output_dir: str) -> None:
    """
    Anonymize DICOM data.
    This function will override any fields with sensitive information with dummy values.
    https://pypi.org/project/dicom-anonymizer/#description

    Additionally, pixel data is replaced with zeros.

    Args:
        input_dir (str): The input directory containing DICOM files from a study (patient).
        output_dir (str): The output directory where anonymized DICOM files will be saved.
    """
    if Path(output_dir).is_dir():
        pass
    else:
        Path(output_dir).mkdir(parents=True)

    all_dicom_files = list(Path(input_dir).rglob("*.dcm"))
    for dcm_file in all_dicom_files:
        new_file = Path(dcm_file.as_posix().replace(input_dir, output_dir))
        new_file.parent.mkdir(parents=True, exist_ok=True)
        anonymize_dicom_file(
            dcm_file.as_posix(), new_file.as_posix(), delete_private_tags=False
        )
        ds = dcmread(new_file.as_posix())
        arr = ds.pixel_array
        new_arr = np.zeros_like(arr)
        ds.PixelData = new_arr.tobytes()
        ds.save_as(new_file.as_posix())


if __name__ == "__main__":
    base_dir = "path_to_data_dir"
    dcm_data_dir = f"{base_dir}/path_to_sub"
    base_output_dir = f"{base_dir}/output_path"
    anonymize_data(dcm_data_dir, base_output_dir)
