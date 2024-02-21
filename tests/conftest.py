import re
import subprocess
import pytest
from pathlib import Path

from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
import random
from dcm_classifier.image_type_inference import ImageTypeClassifierBase

# Previous code for generating mock volumes from botimage
adjacent_testing_data_path: Path = Path(__file__).parent / "testing_data"
current_file_path: Path = Path(__file__).parent
inference_model_path = list(
    Path(__file__).parent.parent.rglob("models/rf_classifier.onnx")
)[0]

# path to the testing data directory
test_data_dir_path: Path = Path(__file__).parent / "testing_data"
tar_path: Path = test_data_dir_path / "anonymized_testing_data.tar.gz"

# path to the anonymized testing data directory
testing_dicom_dir: Path = test_data_dir_path / "anonymized_testing_data"

# Check to see if tar file is unpacked or not
if not testing_dicom_dir.exists():
    testing_dicom_dir.mkdir(parents=True)
    subprocess.run(f"tar -xf {tar_path} -C {test_data_dir_path}", shell=True)

# run study on anonymized data
inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
study = ProcessOneDicomStudyToVolumesMappingBase(
    study_directory=(testing_dicom_dir / "anonymized_data"), inferer=inferer
)
study.run_inference()

s_test = study.series_dictionary.get(15)
print(s_test.get_volume_list()[0].get_one_volume_dcm_filenames()[0])

ax_series = [
    study.series_dictionary.get(6),
    study.series_dictionary.get(7),
    # study.series_dictionary.get(8),
    # study.series_dictionary.get(9),
    study.series_dictionary.get(11),
    study.series_dictionary.get(12),
    study.series_dictionary.get(14),
]
sag_series = [
    # study.series_dictionary.get(2),
    study.series_dictionary.get(10),
    study.series_dictionary.get(13),
]
cor_series = [
    # study.series_dictionary.get(3),
    study.series_dictionary.get(15)
]

t1_series = [
    study.series_dictionary.get(10),
    study.series_dictionary.get(12),
    study.series_dictionary.get(13),
    study.series_dictionary.get(14),
    study.series_dictionary.get(15),
]
flair_series = [study.series_dictionary.get(7)]
t2_series = [study.series_dictionary.get(11)]
adc_series = [study.series_dictionary.get(6)]


@pytest.fixture(scope="session")
def get_data_dir():
    return testing_dicom_dir / "anonymized_data"


@pytest.fixture(scope="session")
def mock_ax_series():
    return ax_series


@pytest.fixture(scope="session")
def mock_sag_series():
    return sag_series


@pytest.fixture(scope="session")
def mock_cor_series():
    return cor_series


@pytest.fixture(scope="session")
def mock_t1_series():
    return t1_series


@pytest.fixture(scope="session")
def mock_flair_series():
    return flair_series


@pytest.fixture(scope="session")
def mock_t2_series():
    return t2_series


@pytest.fixture(scope="session")
def mock_adc_series():
    return adc_series


@pytest.fixture(scope="session")
def contrast_file_path():
    return testing_dicom_dir / "contrast_data" / "with_contrast"


@pytest.fixture(scope="session")
def no_contrast_file_path():
    return testing_dicom_dir / "contrast_data" / "without_contrast"


@pytest.fixture(scope="session")
def mock_series_study():
    return study


@pytest.fixture(scope="session")
def mock_volume_study():
    return study


@pytest.fixture(scope="session")
def mock_volumes():
    """A fixture function that is used to mock DICOM volumes for testing
    :returns list_of_volumes: a list containing filepaths to volume directories
    """

    def generate_dcm_volumes(volumes: list[str | Path], output_path: str | Path):
        """A function that creates a directory and DICOM files for each volume in the list of provided volumes.
        The directory and files are written to the specified output path.
           :param volumes: a list of DICOM volume file paths
           :param output_path: the output path where the directories and DICOM files will be written to
           :return None
        """
        output_path = Path(output_path)
        for i in range(len(volumes)):
            vol_path: Path = output_path / f"volume_{i}"
            vol_path.mkdir(exist_ok=True, parents=True)

            volume_datasets = re.split(",\n(?={)", volumes[i])
            for j in range(1, len(volume_datasets), 2):
                ds = Dataset.from_json(volume_datasets[j])
                ds.file_meta = FileMetaDataset.from_json(volume_datasets[j - 1])
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                ds.save_as(
                    f"{output_path}/volume_{i}/test_dcm_{(j - 1) / 2}.dcm",
                    write_like_original=False,
                )

    # paths where mock volumes will be written to and read in from

    adc_volume_dir_path = adjacent_testing_data_path / "adc_volumes"
    dwi_volume_dir_path = adjacent_testing_data_path / "dwi_volumes"
    t2w_volume_dir_path = adjacent_testing_data_path / "t2w_volumes"
    other_volume_dir_path = adjacent_testing_data_path / "other_volumes"

    # checks if volume paths exist, if not mock volumes are created from the file generated by 'generate_mock_volume_json.py'
    if not (
        adc_volume_dir_path.exists()
        and dwi_volume_dir_path.exists()
        and t2w_volume_dir_path.exists()
        and other_volume_dir_path.exists()
    ):
        adc_volumes = list()
        dwi_volumes = list()
        t2w_volumes = list()
        other_volumes = list()

        mock_data_file: Path = adjacent_testing_data_path / "mock_data.txt"

        with open(mock_data_file) as file:
            mock_volumes_json_str = file.read()
        # uses regex to find all the volume delimiters within mock_data.txt file
        volume_type_list = re.findall(r"\w{3} VOLUME", mock_volumes_json_str)
        seperated_volumes = re.split(
            r"\w{3} VOLUME",
            mock_volumes_json_str,
        )
        # classifies volumes based on their delimiters and appends them to their corresponding list
        for i in range(len(volume_type_list)):
            if volume_type_list[i] == "ADC VOLUME":
                adc_volumes.append(seperated_volumes[i])
            elif volume_type_list[i] == "DWI VOLUME":
                dwi_volumes.append(seperated_volumes[i])
            elif volume_type_list[i] == "T2W VOLUME":
                t2w_volumes.append(seperated_volumes[i])
            else:
                other_volumes.append(seperated_volumes[i])
        # generates volumes in list and writes them to a path based on their image type
        generate_dcm_volumes(adc_volumes, adc_volume_dir_path)
        generate_dcm_volumes(dwi_volumes, dwi_volume_dir_path)
        generate_dcm_volumes(t2w_volumes, t2w_volume_dir_path)
        generate_dcm_volumes(other_volumes, other_volume_dir_path)

    # reads in volumes from mock data path
    list_of_volumes = []
    all_volumes_list = [
        x for x in list(adjacent_testing_data_path.rglob("**/*volumes/*")) if x.is_dir()
    ]
    for volume in all_volumes_list:
        volume_files_list = list(volume.rglob("*.dcm"))
        randomized_volume_files_list = random.sample(
            volume_files_list, len(volume_files_list)
        )
        list_of_volumes.append(randomized_volume_files_list)
    return sorted(list_of_volumes)


@pytest.fixture(scope="session")
def default_image_type_classifier_base():
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

    image_type_classifier_base = ImageTypeClassifierBase(
        classification_model_filename=default_classification_model_filename,
        classification_feature_list=modality_columns,
        image_type_map=imagetype_to_integer_mapping,
        min_probability_threshold=0.4,
    )

    return image_type_classifier_base
