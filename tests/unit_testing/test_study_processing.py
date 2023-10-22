import pytest

from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import (
    ProcessOneDicomStudyToVolumesMappingBase,
)
from pathlib import Path

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"


@pytest.mark.skip(reason="Need to add public data")
def test_get_list_of_primary_volume_info():
    test_data_dicom_dir: str = "XXXX"
    study_path = relative_testing_data_path / test_data_dicom_dir

    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(study_path)
    volume_info_dictionaries = (
        study_to_volume_mapping_base.get_list_of_primary_volume_info()
    )
    assert len(volume_info_dictionaries) == 13


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


@pytest.mark.skip(reason="Need to add public data")
def test_get_study_dictionary_and_set_inferer():
    test_data_dicom_dir: str = "XXXX"
    study_path = relative_testing_data_path / test_data_dicom_dir
    modality_columns = [
        "ImageTypeADC",
        "ImageTypeFA",
        "ImageTypeTrace",
        "SeriesVolumeCount",
        "EchoTime",
        "RepetitionTime",
        "FlipAngle",
        "PixelBandwidth",
        "SAR",
        "Diffusionb-valueCount",
        "Diffusionb-valueMax",
    ]

    imagetype_to_integer_mapping = {
        "adc": 0,
        "fa": 1,
        "tracew": 2,
        "t2w": 3,
        "t2starw": 4,
        "t1w": 5,
        "flair": 6,
        "field_map": 7,
        "dwig": 8,
        "dwi_multishell": 9,
        "fmri": 10,
    }

    default_classification_model_filename: Path = (
        Path(__file__).parents[2] / "models" / "rf_classifier.onnx"
    )

    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(study_path)
    study_dictionary = study_to_volume_mapping_base.get_study_dictionary()
    assert isinstance(study_dictionary, dict)
    assert len(study_dictionary) == 8
    study_to_volume_mapping_base.set_inferer(
        ImageTypeClassifierBase(
            classification_model_filename=default_classification_model_filename,
            classification_feature_list=modality_columns,
            image_type_map=imagetype_to_integer_mapping,
            mode="volume",
            min_probability_threshold=0.4,
        )
    )
    assert (
        study_to_volume_mapping_base.inferer.classification_model_filename
        == default_classification_model_filename
    )
    assert (
        study_to_volume_mapping_base.inferer.classification_feature_list
        == modality_columns
    )
    assert (
        study_to_volume_mapping_base.inferer.imagetype_to_int_map
        == imagetype_to_integer_mapping
    )
    assert study_to_volume_mapping_base.inferer.mode == "volume"
    assert study_to_volume_mapping_base.inferer.min_probability_threshold == 0.4


# @pytest.mark.skip(reason="Not implemented yet")
# def test_run_inference():
#     pass
