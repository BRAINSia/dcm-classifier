# =========================================================================
#
#    Copyright NumFOCUS
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0.txt
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#  =========================================================================

import contextlib
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dicom_config import MISSING_VALUE, inference_features
from .dicom_series import DicomSingleSeries
from .utility_functions import (
    infer_diffusion_from_gradient,
)

imagetype_to_integer_mapping = {
    "t1w": 0,
    "gret2star": 1,
    "t2w": 2,
    "flair": 3,
    "b0": 4,
    "tracew": 5,
    "adc": 6,
    "fa": 7,
    "eadc": 8,
    "dwig": 9,
}

current_file_path = Path(__file__).absolute()
dcm_classifier_path = current_file_path.parent

# Environment variables that state how many CPUs or threads this process may use.
cpu_limit_variables: tuple[str, ...] = (
    # set by a batch scheduler for the job
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "NSLOTS",  # Sun Grid Engine; also read by ITK
    "NCPUS",  # PBS Professional
    "PBS_NUM_PPN",  # Torque
    "LSB_DJOB_NUMPROC",  # LSF
    # set by a caller, such as a nipype interface, to budget threads
    "OMP_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def _read_cgroup_cpu_quota(cgroup_directory: Path) -> float | None:
    """CPU quota of one cgroup directory in units of CPUs, or None when unlimited."""
    try:
        # cgroup v2: "<quota> <period>", with quota "max" when unlimited
        quota, period = (cgroup_directory / "cpu.max").read_text().split()
    except (OSError, ValueError):
        try:
            # cgroup v1: quota is -1 when unlimited
            quota = (cgroup_directory / "cpu.cfs_quota_us").read_text().strip()
            period = (cgroup_directory / "cpu.cfs_period_us").read_text().strip()
        except OSError:
            return None
    with contextlib.suppress(ValueError):
        if int(quota) > 0 and int(period) > 0:
            return int(quota) / int(period)
    return None


def cgroup_cpu_quota(
    proc_cgroup_file: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> float | None:
    """
    CPU time quota imposed on this process by Linux control groups, in units of CPUs.

    This is the limit set by ``docker run --cpus``, Kubernetes CPU limits, and AWS ECS task sizes.
    Such a quota does not change the CPU affinity of the process. The process's cgroup and all
    of its ancestors are checked, for both cgroup v1 and v2.

    :param proc_cgroup_file: File listing the cgroups of the process.
    :param cgroup_root: Mount point of the cgroup filesystem.
    :return: Smallest quota found, or None when there is none.
    :rtype: float | None
    """
    try:
        lines = proc_cgroup_file.read_text().splitlines()
    except OSError:
        return None
    quotas: list[float] = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, cgroup_path = fields[1], fields[2]
        if controllers == "":
            hierarchy_root = cgroup_root
        elif "cpu" in controllers.split(","):
            hierarchy_root = cgroup_root / controllers
            if not hierarchy_root.is_dir():
                hierarchy_root = cgroup_root / "cpu"
        else:
            continue
        directory = hierarchy_root / cgroup_path.strip("/")
        while True:
            quota = _read_cgroup_cpu_quota(directory)
            if quota is not None:
                quotas.append(quota)
            if directory in (hierarchy_root, directory.parent):
                break
            directory = directory.parent
    return min(quotas) if quotas else None


def allowed_cpu_count() -> int | None:
    """
    Number of CPUs this process has been granted, which can be fewer than the machine has.

    The result is the smallest of the limits that can be found:
    the CPU affinity of the process (``taskset``, cpusets from SLURM or PBS cgroups, ``docker run --cpuset-cpus``),
    the cgroup CPU quota (``docker run --cpus``, Kubernetes, AWS ECS), and
    the CPU or thread count given in the environment by a batch scheduler or by the caller (see ``cpu_limit_variables``).

    :return: Number of granted CPUs, at least 1, or None when no limit can be determined.
    :rtype: int | None
    """
    limits: list[int] = []
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    quota = cgroup_cpu_quota()
    if quota is not None:
        limits.append(math.ceil(quota))
    for variable in cpu_limit_variables:
        with contextlib.suppress(KeyError, ValueError):
            # OMP_NUM_THREADS may list one value per nesting level, "4,2"
            limits.append(int(os.environ[variable].split(",")[0]))
    limits = [limit for limit in limits if limit > 0]
    return min(limits) if limits else None


class ImageTypeClassifierBase:
    """
    Inference class for image type classification. The base implementation is for our standard model. This class
    can be customized by users to implement their own models for targeted for specific datasets.

    Attributes:
        classification_model_filename (Union[str, Path]): Path to the classification model file (base implementation requires ONNX file).

        classification_feature_list (List[str]): List of features used for classification.

        image_type_map (Dict[str, str]): Mapping between class name and model integer output.

        min_probability_threshold (float): Minimum probability threshold for classification, defaults to 0.4. If maximum class probability is below this threshold, the image type is set to "unknown".

        num_threads (int | None): Number of threads the ONNX Runtime session may use. Defaults to allowed_cpu_count(), or to the ONNX Runtime default (None) when no limit can be determined.

    """

    def __init__(
        self,
        classification_model_filename: str | Path | None = None,
        classification_feature_list: list[str] = inference_features,
        image_type_map: dict[str, int] = imagetype_to_integer_mapping,
        min_probability_threshold: float = 0.4,
        num_threads: int | None = None,
    ):
        """
        Initialize the ImageTypeClassifierBase.

        :param classification_model_filename: Path to the classification model file (base implementation requires ONNX file).
        :param classification_feature_list: List of features used for classification.
        :param image_type_map: Mapping between class name and model integer output.
        :param min_probability_threshold: Minimum probability threshold for classification, defaults to 0.4. If maximum class probability
        :param num_threads: Number of threads the ONNX Runtime session may use. Defaults to allowed_cpu_count(), or to the ONNX Runtime default when no limit can be determined.
        :type classification_model_filename: str | Path
        :type classification_feature_list: list[str]
        :type image_type_map: dict[str, int]
        :type min_probability_threshold: float
        :type num_threads: int | None

        """
        if num_threads is None:
            num_threads = allowed_cpu_count()
        if num_threads is not None and num_threads < 1:
            raise ValueError(f"num_threads must be at least 1, got {num_threads}")
        self.num_threads: int | None = num_threads
        self._inference_session: Any = None
        self._inference_session_model_filename: Path | None = None
        if classification_model_filename is None:
            self.classification_model_filename = (
                dcm_classifier_path / "models" / "ova_rf_classifier.onnx"
            )
        else:
            self.classification_model_filename: str | Path = Path(
                classification_model_filename
            )
        self.classification_feature_list: list[str] = classification_feature_list
        self.imagetype_to_int_map: dict[str, int] = image_type_map
        self.int_to_imagetype_map: dict[int, str] = self.get_int_to_type_map()
        self.min_probability_threshold: float = min_probability_threshold
        self.series: DicomSingleSeries | None = None
        self.series_number: int | None = None
        self.info_dict: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        # An ONNX Runtime session cannot be pickled; it is rebuilt on first use.
        state = self.__dict__.copy()
        state["_inference_session"] = None
        state["_inference_session_model_filename"] = None
        return state

    def get_inference_session(self) -> Any:
        """
        Get the ONNX Runtime session for the classification model, creating it on first use.

        The session is limited to ``num_threads`` threads and is reused across calls.

        :return: ONNX Runtime inference session.
        :rtype: onnxruntime.InferenceSession
        """
        model_filename = Path(self.classification_model_filename)
        if (
            self._inference_session is None
            or self._inference_session_model_filename != model_filename
        ):
            import onnxruntime as rt

            options = rt.SessionOptions()
            # 0 lets ONNX Runtime choose the thread count.
            options.intra_op_num_threads = self.num_threads or 0
            options.inter_op_num_threads = 1
            self._inference_session = rt.InferenceSession(
                model_filename.as_posix(), sess_options=options
            )
            self._inference_session_model_filename = model_filename
        return self._inference_session

    def get_min_probability_threshold(self) -> float:
        """
        Get the minimum probability threshold for classification.

        :return: Minimum probability threshold for classification.
        :rtype: float
        """
        return self.min_probability_threshold

    def get_int_to_type_map(self) -> dict:
        """
        Get the integer to image type mapping.

        :return: Dictionary mapping integers to image type names.
        :rtype: dict
        """
        return {v: k for k, v in self.imagetype_to_int_map.items()}

    def _update_diffusion_series_modality(self) -> None:
        """
        After processing and file organization into volumes and series is completed, this function is called to
        check if the series is a diffusion series. If so, the series modality is overriden to dwig.
        This function is called during inference
        """
        volume_list = self.series.get_volume_list()
        first_dcm_per_volume = [
            volume.get_one_volume_dcm_filenames()[0] for volume in volume_list
        ]
        diff_modality = infer_diffusion_from_gradient(first_dcm_per_volume)
        self.series.set_series_modality(diff_modality)

    def check_if_diffusion(self) -> None:  # pragma: no cover
        msg = "This method check_if_diffusion is deprecated, use  update_diffusion_series_modality instead."
        raise NotImplementedError(msg)

    def set_series(self, series: DicomSingleSeries) -> None:
        """
        Set the DICOM series for classification.

        :param series: DicomSingleSeries object representing the DICOM series.
        :type series: DicomSingleSeries
        """

        self.series = series
        self.series_number = series.get_series_number()
        self.info_dict = self.series.get_series_info_dict()

    def infer_acquisition_plane(self, feature_dict: dict = None) -> str:
        """
        Infer the acquisition plane based on DICOM information and image properties.

        This is an implementation of the decision tree for acquisition plane inference.
        It takes the ImageOrientationPatient_0, ImageOrientationPatient_5 from info_dict
        and returns the acquisition plane prediction. This can be implemented multiple ways. For more details, see the publication.

        To determine if the image is isotropic, we check if the all spacing components are within 10% of the cube root of the voxel volume.

        :param feature_dict: A dictionary containing features used for classification.
        :type feature_dict: dict

        :return: A string representing the inferred acquisition plane ("iso" for isotropic, "ax" for axial, "sag" for sagittal and "cor" for coronal).
        :rtype: str
        """
        # check if the volume was invalidated
        for field in [
            "ImageOrientationPatient_0",
            "ImageOrientationPatient_5",
        ]:
            if field == "INVALID_VALUE":
                return "INVALID"

        if float(feature_dict["ImageOrientationPatient_5"]) <= 0.5:
            return "ax"
        if float(feature_dict["ImageOrientationPatient_0"]) <= 0.707:
            return "sag"
        return "cor"

    def infer_isotropic(self, feature_dict: dict = None) -> bool | None:
        """
        Infer the acquisition plane based on DICOM information and image properties.

        :param feature_dict: A dictionary containing features used for classification.
        :type feature_dict: dict

        :return: A boolean representing whether the image is isotropic.
        :rtype: bool
        """
        # TODO: this might need to be changed if acquisition is 3d and spacing has more than 2 values
        # check if the volume was invalidated
        for field in [
            "PixelSpacing_0",
            "PixelSpacing_1",
            "SliceThickness",
        ]:
            if field not in feature_dict.keys():
                return None

        # check if the volume is isotropic
        spacing = np.array(feature_dict["PixelSpacing"])
        thickness = float(feature_dict["SliceThickness"])
        if np.allclose(spacing, thickness, rtol=0.1):
            return True
        return False

    def infer_contrast(self, feature_dict: dict = None) -> bool:
        """
        Infer whether the image has contrast based on DICOM information and image properties.

        :param feature_dict: A dictionary containing features used for classification.
        :type feature_dict: dict

        :return: A boolean representing whether the image has contrast.
        :rtype: bool
        """
        # check if the volume was invalidated

        field = "ContrastBolusAgent"
        if field not in feature_dict.keys():
            return False

        # check if the volume has contrast
        if "none" not in feature_dict[field].lower():
            return True
        return False

    def infer_modality(self, feature_dict: dict = None) -> (str, pd.DataFrame):
        """
        Infer the modality (image type) of the DICOM series based on a feature dictionary.

        This method uses an ONNX model for image type classification to predict the modality of the series.

        :param feature_dict: A dictionary containing features used for classification.
        :type feature_dict: dict

        :return: A tuple containing the inferred modality and a Pandas DataFrame containing classification results.
        :rtype: Tuple[str, pd.DataFrame]
        """
        # check if the volume was invalidated
        for field in self.classification_feature_list:
            if feature_dict[field] == "INVALID":
                return "INVALID", None

        # TODO: ensure the ordering of features is correct
        e_inputs: pd.DataFrame = pd.DataFrame([feature_dict])
        # Drop all dicom images without a SeriesNumber
        # e_inputs = e_inputs[e_inputs["SeriesNumber"].notna()]

        sess = self.get_inference_session()

        input_name: str = sess.get_inputs()[0].name
        label_name: str = sess.get_outputs()[0].name
        prob_name: str = sess.get_outputs()[1].name

        full_outputs: pd.DataFrame = pd.DataFrame()
        full_outputs["SeriesNumber"] = [self.series_number]
        model_inputs: pd.DataFrame = e_inputs.reindex(columns=inference_features)
        try:
            numeric_inputs: np.array = model_inputs.astype(np.float32).to_numpy()
        except ValueError:
            # Short circuit if the inputs are not sufficient for inference
            full_outputs["GUESS_ONNX"] = ["InvalidDicomInputs"]
            return "INVALID", full_outputs

        pred_onx_run_output = sess.run(
            [label_name, prob_name], {input_name: numeric_inputs}
        )
        pred_onx = pred_onx_run_output[0]
        probability_onx = pred_onx_run_output[1]
        full_outputs["GUESS_ONNX_CODE"] = [pred_onx]
        full_outputs["GUESS_ONNX"] = [pred_onx]

        for type_name, type_integer_code in self.imagetype_to_int_map.items():
            full_outputs["GUESS_ONNX"] = full_outputs["GUESS_ONNX"].replace(
                to_replace=type_integer_code, value=type_name
            )

        for col_index, class_probability in probability_onx[0].items():
            col_name = self.int_to_imagetype_map[int(col_index)]
            class_name: str = f"GUESS_ONNX_{col_name}"
            full_outputs[class_name] = [class_probability]

        del col_index, class_probability, col_name, class_name
        image_type: str = full_outputs.iloc[
            0, full_outputs.columns.get_loc("GUESS_ONNX")
        ]
        if max(probability_onx[0].values()) < self.min_probability_threshold:
            image_type = f"LOW_PROBABILITY_{image_type}"

        return image_type, full_outputs

    def run_inference(self):
        """
        Run image type inference for the specified DICOM series or volumes.

        This method performs image type classification and acquisition plane inference based on the provided features.

        """

        def validate_features(input_dict: dict) -> bool:
            """
            Validate the presence of required features in the input feature dictionary.

            This function checks if all the features specified in the `classification_feature_list` are present in
            the input feature dictionary.

            :param input_dict: A dictionary containing features for classification.
            :type input_dict: dict

            :return: True if all required features are present, False otherwise.
            :rtype: bool

            """
            missing_features = []
            for feature in self.classification_feature_list:
                if feature not in input_dict.keys():
                    missing_features.append(feature)
            if len(missing_features) > 0:
                warnings.warn(
                    f"Missing features for SeriesNumber {self.series_number}: {missing_features}\n"
                    f"Series contains: {self.series.volume_info_list[0].get_one_volume_dcm_filenames()}",
                    stacklevel=2,
                )
                return False
            return True

        # classify volumes
        for volume in self.series.get_volume_list():
            features_validated: bool = validate_features(volume.get_volume_dictionary())
            if features_validated:
                acquisition_plane: str = self.infer_acquisition_plane(
                    feature_dict=volume.get_volume_dictionary()
                )
                volume.set_acquisition_plane(acquisition_plane)
                isotropic = self.infer_isotropic(feature_dict=self.info_dict)
                volume.set_is_isotropic(isotropic)

                contrast = self.infer_contrast(feature_dict=self.info_dict)
                volume.set_has_contrast(contrast)

                modality, full_outputs = self.infer_modality(
                    feature_dict=volume.get_volume_dictionary()
                )
                volume.set_volume_modality(modality)
                volume.set_modality_probabilities(pd.DataFrame(full_outputs, index=[0]))
            else:
                # if features are not validated, set values to defaults
                volume.set_acquisition_plane("INVALID")
                volume.set_is_isotropic(False)
                volume.set_has_contrast(False)
                volume.set_volume_modality("INVALID")
                volume.set_modality_probabilities(pd.DataFrame())

        """
        Aggregates the modality and acquisition plane information from the volumes to the series level.
        """
        # if there is only a single volume in the series, the series modality and acquisition plane are set to the
        # volume's modality and acquisition plane
        if len(self.series.get_volume_list()) == 1:
            volume = self.series.get_volume_list()[0]
            modality = volume.get_volume_modality().replace("LOW_PROBABILITY_", "")

            self.series.set_series_modality(modality)
            bval = volume.get_volume_bvalue()
            if bval > 0 and modality not in ["tracew", "adc", "fa", "eadc", "dwig"]:
                self._update_diffusion_series_modality()

            self.series.set_modality_probabilities(volume.get_modality_probabilities())
            self.series.set_acquisition_plane(volume.get_acquisition_plane())
            self.series.set_is_isotropic(volume.get_is_isotropic())
            self.series.set_has_contrast(volume.get_has_contrast())
        else:
            # for acquisition plane we assume as volumes in series have the same acquisition plane
            # similarly we propagate the isotropic and contrast
            self.series.set_acquisition_plane(
                self.series.get_volume_list()[0].get_acquisition_plane()
            )
            self.series.set_is_isotropic(
                self.series.get_volume_list()[0].get_is_isotropic()
            )
            self.series.set_has_contrast(
                self.series.get_volume_list()[0].get_has_contrast()
            )

            # get all modalities from the series
            modalities = [
                volume.get_volume_modality().replace("LOW_PROBABILITY_", "")
                for volume in self.series.get_volume_list()
            ]
            unique_modalities = list(set(modalities))

            # get all bvalues in the series
            bvals = [
                volume.get_volume_bvalue() for volume in self.series.get_volume_list()
            ]
            unique_bvals = list(set(bvals))
            if len(unique_modalities) == 1:
                if MISSING_VALUE not in unique_bvals:
                    self._update_diffusion_series_modality()
                else:
                    self.series.set_series_modality(
                        self.series.get_volume_list()[0].get_volume_modality()
                    )
            elif "pd" in unique_modalities and "t2w" in unique_modalities:
                self.series.set_series_modality("PDT2")
            # DWI series usually contains b0 and dwig gradient volumes
            elif "dwig" in unique_modalities:
                self._update_diffusion_series_modality()
            # TRACEW series often contains b0 and tracew volumes
            # TODO: get more data for testing
            elif "tracew" in unique_modalities:
                self.series.set_series_modality("tracew")
            elif MISSING_VALUE not in unique_bvals:
                self._update_diffusion_series_modality()
            else:
                # if other scenarios are not met, we set the modality to the first volume's modality
                self.series.set_series_modality(
                    self.series.get_volume_list()[0].get_volume_modality()
                )

        """Override PERFUSION data"""
        # TODO: This is temporary fix and should be removed when the model is retrained with PERFUSION data and class
        for volume in self.series.get_volume_list():
            if volume.get_volume_dictionary().get("ImageType_PERFUSION", 0) == 1:
                self.series.set_series_modality("perfusion")
                break
