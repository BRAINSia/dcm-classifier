import pytest
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from pathlib import Path


current_file_path = Path(__file__).parent.resolve()
inference_model_path = list(
    Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx")
)[0]


def test_adc_dcm_series_modality(
    mock_volumes, default_image_type_classifier_base, mock_adc_series
):
    for series in mock_adc_series:
        assert series.get_modality() == "adc"


def test_ax_dcm_series_acq_plane(mock_series_study, mock_ax_series):
    for series in mock_ax_series:
        assert series.get_acquisition_plane() == "ax"


def test_sag_dcm_series_acq_plane(mock_sag_series):
    for series in mock_sag_series:
        assert series.get_acquisition_plane() == "sag"
    # for series_number, series in mock_series_study.series_dictionary.items():
    #     if series_number == 2 or series_number == 10 or series_number == 13:
    #         assert series.get_acquisition_plane() == "sag"


def test_cor_dcm_series_acq_plane(mock_cor_series):
    for series in mock_cor_series:
        assert series.get_acquisition_plane() == "cor"
    # for series_number, series in mock_series_study.series_dictionary.items():
    #     if series_number == 3 or series_number == 15:
    #         assert series.get_acquisition_plane() == "cor"


def test_t1_dcm_series_modality(mock_t1_series):
    for series in mock_t1_series:
        assert series.get_modality() == "t1w"
    # for series_number, series in mock_series_study.series_dictionary.items():
    #     if 10 <= series_number <= 15 and series_number != 11:
    #         assert series.get_modality() == "t1w"


def test_flair_dcm_series_modality(mock_flair_series):
    for series in mock_flair_series:
        assert series.get_modality() == "flair"
    # for series_number, series in mock_series_study.series_dictionary.items():
    #     if series_number == 7:
    #         assert series.get_modality() == "flair"


def test_t2_dcm_series_modality(mock_t2_series):
    for series in mock_t2_series:
        assert series.get_modality() == "t2w"
    # for series_number, series in mock_series_study.series_dictionary.items():
    #     if series_number == 11:
    #         assert series.get_modality() == "t2w"


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


def test_t1w_dcm_volume_modality(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if 12 <= series_num <= 15:
                assert volume.get_modality() == "t1w"


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
