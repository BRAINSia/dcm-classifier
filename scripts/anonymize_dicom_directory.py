#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from dicomanonymizer.simpledicomanonymizer import anonymize_dicom_file, generate_actions
from pydicom import dcmread


custom_private_tags = [
    (0x0018, 0x9075),
    (0x0018, 0x9076),
    (0x0018, 0x9087),
    (0x0018, 0x9089),
    (0x0018, 0x9117),
    (0x0018, 0x9147),
    (0x0018, 0x9602),
    (0x0018, 0x9603),
    (0x0018, 0x9604),
    (0x0018, 0x9605),
    (0x0018, 0x9606),
    (0x0018, 0x9607),
    (0x0019, 0x10E0),
    (0x0019, 0x10DF),
    (0x0019, 0x10D9),
    (0x0021, 0x105A),
    (0x0043, 0x1039),
    (0x0019, 0x100C),
    (0x0019, 0x100D),
    (0x0019, 0x100E),
    (0x0019, 0x1027),
    (0x0029, 0x1010),
    (0x0019, 0x000A),
    (0x0019, 0x000B),
    (0x0019, 0x000D),
    (0x0019, 0x000E),
    (0x0019, 0x000F),
    (0x0019, 0x0027),
    (0x0019, 0x0028),
    (0x2001, 0x1003),
    (0x2001, 0x1004),
    (0x2005, 0x10B0),
    (0x2005, 0x10B1),
    (0x2005, 0x10B2),
    (0x0029, 0x1001),
    (0x0029, 0x1090),
    (0x0065, 0x1009),
    (0x0065, 0x1037),
    (0x0018, 0x0010),  # Contrast/BolusAgent
]


def anonymize_data(input_dir: str, output_dir: str, custom_rules: dict) -> None:
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
            dcm_file.as_posix(),
            new_file.as_posix(),
            extra_anonymization_rules=custom_rules,
            delete_private_tags=True,
        )
        ds = dcmread(new_file.as_posix())
        arr = ds.pixel_array
        new_arr = np.zeros_like(arr)
        ds.PixelData = new_arr.tobytes()
        ds.save_as(new_file.as_posix())


if __name__ == "__main__":
    base_dir = "data_dir_path"
    dcm_data_dir = f"{base_dir}/dicom_subject_path"
    base_output_dir = f"{base_dir}/output_path"
    custom_rule_set = generate_actions(custom_private_tags, "keep")
    anonymize_data(dcm_data_dir, base_output_dir, custom_rule_set)
