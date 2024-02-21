import pytest

from dcm_classifier.image_type_inference import (
    ImageTypeClassifierBase,
    imagetype_to_integer_mapping,
)
from dcm_classifier.dicom_config import inference_features
from dcm_classifier.study_processing import (
    ProcessOneDicomStudyToVolumesMappingBase,
)
from pathlib import Path

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"


def test_study_with_search_series():
    study_path = relative_testing_data_path / "anonymized_data"
    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(
        study_path, {"test": 7}
    )
    assert study_to_volume_mapping_base.study_directory == study_path
    assert study_to_volume_mapping_base.search_series == {"test": 7}


def test_get_list_of_primary_volume_info(get_data_dir):
    # study_path = relative_testing_data_path.parent.parent.parent / "dcm_files"

    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(
        get_data_dir
    )
    volume_info_dictionaries = (
        study_to_volume_mapping_base.get_list_of_primary_volume_info()
    )
    print(len(volume_info_dictionaries))

    assert len(volume_info_dictionaries) == 16


# @pytest.mark.skip(reason="Need to add public data")
# def test_get_list_of_primary_volume_info():
#     test_data_dicom_dir: str = "XXXX"
#     study_path = relative_testing_data_path / test_data_dicom_dir
#
#     study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(study_path)
#     volume_info_dictionaries = (
#         study_to_volume_mapping_base.get_list_of_primary_volume_info()
#     )
#     assert len(volume_info_dictionaries) == 13


@pytest.mark.skip(reason="Need to add public data")
def test_get_list_of_primary_volume_info_with_search_series():
    test_data_dicom_dir: str = "XXXX"
    study_path = relative_testing_data_path / test_data_dicom_dir
    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(
        study_path, {"test": 7}
    )
    volume_info_dictionaries = (
        study_to_volume_mapping_base.get_list_of_primary_volume_info()
    )
    assert len(volume_info_dictionaries) == 4


# @pytest.mark.skip(reason="Need to add public data")
def test_get_study_dictionary_and_set_inferer():
    # test_data_dicom_dir: str = "XXXX"
    # study_path = relative_testing_data_path / test_data_dicom_dir
    study_path = (
        relative_testing_data_path / "anonymized_testing_data" / "anonymized_data"
    )

    default_classification_model_filename: Path = (
        Path(__file__).parents[2] / "models" / "rf_classifier.onnx"
    )

    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(study_path)
    study_dictionary = study_to_volume_mapping_base.get_study_dictionary()
    assert isinstance(study_dictionary, dict)
    assert len(study_dictionary) == 15
    study_to_volume_mapping_base.set_inferer(
        ImageTypeClassifierBase(
            classification_model_filename=default_classification_model_filename,
            classification_feature_list=inference_features,
            image_type_map=imagetype_to_integer_mapping,
            min_probability_threshold=0.4,
        )
    )
    assert (
        study_to_volume_mapping_base.inferer.classification_model_filename
        == default_classification_model_filename
    )
    assert (
        study_to_volume_mapping_base.inferer.classification_feature_list
        == inference_features
    )
    assert (
        study_to_volume_mapping_base.inferer.imagetype_to_int_map
        == imagetype_to_integer_mapping
    )
    assert study_to_volume_mapping_base.inferer.min_probability_threshold == 0.4


# @pytest.mark.skip(reason="Not implemented yet")
# def test_run_inference():
#     pass


def test_un_pathlike_dir():
    # Raises an attribute error because the study_directory attribute is not set but is called in print statement
    with pytest.raises(AttributeError) as ex:
        ProcessOneDicomStudyToVolumesMappingBase(-12345)
    assert (
        "'ProcessOneDicomStudyToVolumesMappingBase' object has no attribute 'study_directory'"
        in str(ex.value)
    )


# validate() is not implemented yet and only returns None
def test_validate(mock_volume_study):
    assert mock_volume_study.validate() is None
