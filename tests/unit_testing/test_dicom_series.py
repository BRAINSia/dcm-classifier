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


def test_all_fields_dont_change():
    all_fields_path: Path = (
        current_file_path.parent
        / "testing_data"
        / "anonymized_testing_data"
        / "all_fields_data"
    )
    assert all_fields_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=all_fields_path, inferer=inferer
    )
    study.run_inference()

    all_fields = required_DICOM_fields + optional_DICOM_fields

    # remove fields that are not in the DICOM header
    all_fields = [
        field
        for field in all_fields
        if field
        not in [
            "Diffusionb-value",
            "Diffusionb-valueMax",
        ]
    ]
    for series_num, series in study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            first_file = volume.get_one_volume_dcm_filenames()[0]
            ds = pydicom.dcmread(first_file, stop_before_pixels=True)

            for field in all_fields:
                assert field in volume.get_volume_dictionary()
                if isinstance(ds[field].value, pydicom.dataelem.RawDataElement):
                    e = pydicom.dataelem.DataElement_from_raw(ds[field].value)
                else:
                    e = ds[field].value

                # if element is a list, check each element
                if type(e) is pydicom.multival.MultiValue:
                    for i, v in enumerate(e):
                        assert volume.get_volume_dictionary()[field][i] == v
                else:  # else check the element
                    assert volume.get_volume_dictionary()[field] == e


def test_get_series_and_study_uid(mock_tracew_series):
    for series in mock_tracew_series:
        assert series.get_series_uid() == "2.25.200346831984180887422376003959445101633"
        assert series.get_study_uid() == "2.25.106736773675271926686056457127502108539"


def test_series_modality_probabilities(mock_t1_series):
    for series in mock_t1_series:
        assert series.get_modality_probabilities().shape[1] == 13
        assert round(series.get_modality_probabilities()["GUESS_ONNX_t1w"][0]) == 1.0


def test_set_invalid_series_modality(mock_tracew_series):
    for series in mock_tracew_series:
        with pytest.raises(ValueError) as ex:
            series.set_series_modality(4)
        assert "ERROR: Can only set_modality with a string." in str(ex.value)


def test_set_invalid_series_modality_probabilities(mock_tracew_series):
    for series in mock_tracew_series:
        with pytest.raises(ValueError) as ex:
            series.set_modality_probabilities(4)
        assert "ERROR: Can only set_modality_probabilities with a pd.DataFrame." in str(
            ex.value
        )


def test_set_invalid_isotropic(mock_tracew_series):
    for series in mock_tracew_series:
        with pytest.raises(ValueError) as ex:
            series.set_is_isotropic(4)
        assert "ERROR: " in str(ex.value)


def test_get_is_isotropic(mock_tracew_series):
    for series in mock_tracew_series:
        assert series.get_is_isotropic() is False


def test_set_invalid_contrast(mock_tracew_series):
    for series in mock_tracew_series:
        with pytest.raises(ValueError) as ex:
            series.set_has_contrast(4)
        assert "ERROR: " in str(ex.value)


def test_set_invalid_acq_plane(mock_tracew_series):
    for series in mock_tracew_series:
        with pytest.raises(ValueError) as ex:
            series.set_acquisition_plane(4)
        assert "ERROR: " in str(ex.value)


def test_dwig_dcm_series_modality(get_dwi_study):
    for series in get_dwi_study.get_study_dictionary().values():
        assert series.get_series_modality() == "dwig"


# def test_b0_volume_modality(get_dwi_study):
#     for series_num, series in get_dwi_study.get_study_dictionary().items():
#         for vol in series.get_volume_list():
#             print(vol.get_volume_index())


def test_tracew_dcm_series_modality(mock_tracew_series):
    for series in mock_tracew_series:
        assert series.get_series_modality() == "tracew"


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


def test_dcm_series_no_contrast(no_contrast_file_path):
    assert no_contrast_file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=no_contrast_file_path, inferer=inferer
    )
    study.run_inference()

    for series_number, series in study.series_dictionary.items():
        assert series.get_has_contrast() is False


def test_dcm_series_has_contrast(contrast_file_path):
    assert contrast_file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=contrast_file_path, inferer=inferer
    )
    study.run_inference()

    for series_number, series in study.series_dictionary.items():
        assert series.get_has_contrast() is True
