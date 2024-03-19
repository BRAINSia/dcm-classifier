import pytest

from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.dicom_config import required_DICOM_fields, optional_DICOM_fields
from pathlib import Path
import pydicom


current_file_path = Path(__file__).parent.resolve()
inference_model_path = list(
    Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx")
)[0]


def test_all_fields_dont_change(mock_ax_series, capsys):
    # ax_path: Path = (
    #     current_file_path.parent.parent
    #     / "tests"
    #     / "testing_data"
    #     / "anonymized_testing_data"
    #     / "anonymized_data"
    #     / "6"
    # )
    # assert ax_path.exists()
    # study = ProcessOneDicomStudyToVolumesMappingBase(
    #     study_directory=ax_path, inferer=inferer
    # )
    # study.run_inference()
    # captured = capsys.readouterr()
    # print(captured.out)
    all_fields = required_DICOM_fields + optional_DICOM_fields
    all_fields = [
        field
        for field in all_fields
        if field
        not in [
            "Diffusionb-value",
            # "Contrast/BolusAgent",  # check this, not in pydicom
            "Diffusionb-valueMax",
            # "InPlanePhaseEncodingDirection",  # check this, COL in pydicom, unknown in pipeline
            # "dB/dt",  # check this, not in pydicom had to change to dBdt but pipeline not getting value
            # "NumberOfAverages",  # check this, 4 in pydicom, unknown in pipeline
            # "InversionTime",  # check this, unknown in pydicom
            # "Variable Flip Angle Flag",  # check this, not retrieved in pipeline
        ]
    ]
    for series in mock_ax_series:
        for volume in series.get_volume_list():
            first_file = volume.get_one_volume_dcm_filenames()[0]
            ds = pydicom.dcmread(first_file, stop_before_pixels=True)
            print("\n\n\n")
            print(ds)
            for field in all_fields:
                # print(field)
                try:
                    assert field in volume.get_volume_dictionary()
                    if isinstance(ds[field].value, pydicom.dataelem.RawDataElement):
                        e = pydicom.dataelem.DataElement_from_raw(ds[field].value)
                    else:
                        e = ds[field].value
                    # print(e)
                    # print(volume.get_volume_dictionary()[field])

                    # if element is a list, check each element
                    if type(e) is pydicom.multival.MultiValue:
                        for i, v in enumerate(e):
                            assert volume.get_volume_dictionary()[field][i] == v
                    else:  # else check the element
                        assert volume.get_volume_dictionary()[field] == e
                except Exception as a:
                    # captured = capsys.readouterr()
                    # print(captured.out)
                    print(a)
                    print(volume.get_volume_dictionary()[field])
                    print(f"Field {field}")


def test_scanning_sequence_in_flair(mock_flair_series):
    for series in mock_flair_series:
        for volume in series.get_volume_list():
            assert volume.get_volume_dictionary()["ScanningSequence_IR"] == 1
            assert volume.get_volume_dictionary()["ScanningSequence_SE"] == 1


def test_adc_dcm_series_modality(mock_adc_series):
    for series in mock_adc_series:
        assert series.get_series_modality() == "adc"


def test_ax_dcm_series_acq_plane(mock_ax_series):
    for series in mock_ax_series:
        assert series.get_acquisition_plane() == "ax"


def test_sag_dcm_series_acq_plane(mock_sag_series):
    for series in mock_sag_series:
        assert series.get_acquisition_plane() == "sag"


def test_cor_dcm_series_acq_plane(mock_cor_series):
    for series in mock_cor_series:
        assert series.get_acquisition_plane() == "cor"


def test_t1_dcm_series_modality(mock_t1_series):
    for series in mock_t1_series:
        assert series.get_series_modality() == "t1w"


def test_flair_dcm_series_modality(mock_flair_series):
    for series in mock_flair_series:
        assert series.get_series_modality() == "flair"


def test_t2_dcm_series_modality(mock_t2_series):
    for series in mock_t2_series:
        assert series.get_series_modality() == "t2w"


# TODO: rewrite this for new behavior
# def test_no_valid_dcms():
#     with pytest.raises(FileNotFoundError) as ex:
#         inferer = ImageTypeClassifierBase(
#             classification_model_filename=inference_model_path
#         )
#         dicom_files_dir: Path = (
#             current_file_path.parent / "testing_data" / "dummy_directory"
#         )
#         study = ProcessOneDicomStudyToVolumesMappingBase(
#             study_directory=dicom_files_dir, inferer=inferer
#         )
#         study.run_inference()
#     assert "No DICOMs in: " in str(ex.value)

inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)


def test_dcm_vol_has_contrast(contrast_file_path):
    assert contrast_file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=contrast_file_path, inferer=inferer
    )
    study.run_inference()

    for series_number, series in study.series_dictionary.items():
        assert series.get_volume_list()[0].get_has_contrast() is True


def test_dcm_vol_no_contrast(no_contrast_file_path):
    assert no_contrast_file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=no_contrast_file_path, inferer=inferer
    )
    study.run_inference()

    for series_number, series in study.series_dictionary.items():
        assert series.get_volume_list()[0].get_has_contrast() is False


def test_t1w_dcm_volume_modality(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if 12 <= series_num <= 15:
                assert volume.get_volume_modality() == "t1w"


def test_ax_dcm_volume_acq_plane(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if 5 <= series_num <= 12 and series_num != 10:
                assert volume.get_acquisition_plane() == "ax"


def test_sag_dcm_volume_acq_plane(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if series_num == 10 or series_num == 13:
                assert volume.get_acquisition_plane() == "sag"


def test_cor_dcm_volume_acq_plane(mock_volume_study):
    for series_number, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if series_number == 15:
                assert volume.get_acquisition_plane() == "cor"
