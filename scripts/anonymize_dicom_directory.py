from pathlib import Path
from dicomanonymizer.simpledicomanonymizer import anonymize_dicom_file
from pydicom import dcmread


def anonymize_data(input_dir: str, output_dir: str) -> None:
    """
    Anonymize DICOM data.

    Args:
        input_dir (str): The input directory containing DICOM files.
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
        ds = dcmread(new_file.as_posix(), stop_before_pixels=True)
        ds.save_as(new_file.as_posix())


if __name__ == "__main__":
    base_dir = "/localscratch/Users/mbrzus/Stroke_Data/dcm_classifier_test_data"
    dcm_data_dir = f"{base_dir}/3772_MRI_BRAIN_W_WO_CONTRAST__3266456428582799"
    base_output_dir = f"{base_dir}/anonymized_data"
    anonymize_data(dcm_data_dir, base_output_dir)
