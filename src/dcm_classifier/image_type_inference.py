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

from .dicom_config import inference_features
from .dicom_series import DicomSingleSeries
from .utility_functions import (
    infer_diffusion_from_gradient,
)
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from typing import Any


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


class ImageTypeClassifierBase:
    """
    Inference class for image type classification. The base implementation is for our standard model. This class
    can be customized by users to implement their own models for targeted for specific datasets.

    Attributes:
        classification_model_filename (Union[str, Path]): Path to the classification model file (base implementation requires ONNX file).
        classification_feature_list (List[str]): List of features used for classification.
        image_type_map (Dict[str, str]): Mapping between class name and model integer output.
        mode (str): "series" or "volume" to run inference on series or volume level (a series could have multiple subvolumes).
        min_probability_threshold (float): Minimum probability threshold for classification, defaults to 0.4. If maximum class probability is below this threshold, the image type is set to "unknown".

    Methods:
        get_int_to_type_map(self) -> dict:

        set_series(self, series: DicomSingleSeries) -> None:

        infer_acquisition_plane(self, feature_dict: dict = None) -> str:

        infer_modality(self, feature_dict: dict = None) -> (str, pd.DataFrame):

        run_inference(self) -> None:
    """

    def __init__(
        self,
        classification_model_filename: str | Path,
        classification_feature_list: list[str] = inference_features,
        image_type_map: dict[str, int] = imagetype_to_integer_mapping,
        min_probability_threshold: float = 0.4,
    ):
        """
        Initialize the ImageTypeClassifierBase.

        Args:
            classification_model_filename (Union[str, Path]): Path to the classification model file (base implementation requires ONNX file).
            classification_feature_list (List[str]): List of features used for classification.
            image_type_map (Dict[str, str]): Mapping between class name and model integer output.
            mode (str): "series" or "volume" to run inference on series or volume level (a series could have multiple subvolumes).
            min_probability_threshold (float): Minimum probability threshold for classification, defaults to 0.4. If maximum class probability is below this threshold, the image type is set to "unknown".
        """
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

    def get_min_probability_threshold(self) -> float:
        """
        Get the minimum probability threshold for classification.

        Returns:
            float: Minimum probability threshold for classification.
        """
        return self.min_probability_threshold

    def get_int_to_type_map(self) -> dict:
        """
        Get the integer to image type mapping.

        Returns:
            dict: Dictionary mapping integers to image type names.
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
        raise NotImplementedError(
            "This method check_if_diffusion is deprecated, use  update_diffusion_series_modality instead."
        )

    def set_series(self, series: DicomSingleSeries) -> None:
        """
        Set the DICOM series for classification.

        Args:
            series (DicomSingleSeries): DicomSingleSeries object representing the DICOM series.
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

        Args:
            feature_dict (dict): Optional dictionary of additional features for inference.

        Returns:
            str: A string representing the inferred acquisition plane ("iso" for isotropic, "ax" for axial, "sag" for sagittal and "cor" for coronal).
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
        else:
            if float(feature_dict["ImageOrientationPatient_0"]) <= 0.707:
                return "sag"
            else:
                return "cor"

    def infer_isotropic(self, feature_dict: dict = None) -> bool | None:
        """
        Infer the acquisition plane based on DICOM information and image properties.
        Args:
            feature_dict: A dictionary containing features used for classification.

        Returns:
            bool: True if the image is isotropic, False otherwise.
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

    def infer_contrast(self, feature_dict: dict = None) -> bool | None:
        """
        Infer whether the image has contrast based on DICOM information and image properties.
        Args:
            feature_dict: A dictionary containing features used for classification.

        Returns:
            bool: True if the image has contrast, False otherwise.
        """
        # check if the volume was invalidated

        field = "Contrast/BolusAgent"
        if field not in feature_dict.keys():
            return None

        # check if the volume has contrast
        if "none" not in feature_dict[field].lower():
            return True
        return False

    def infer_modality(self, feature_dict: dict = None) -> (str, pd.DataFrame):
        """
        Infer the modality (image type) of the DICOM series based on a feature dictionary.

        This method uses an ONNX model for image type classification to predict the modality of the series.

        Args:
            feature_dict (dict): A dictionary containing features used for classification.

        Returns:
            Tuple(str, pd.DataFrame): A tuple containing:
                - A string representing the inferred modality (image type).
                - A Pandas DataFrame containing classification results, including class probabilities.
        """
        # check if all features are present for inference
        # if len(self.classification_feature_list) < len(inference_features):
        #     return "INVALID", None

        # check if the volume was invalidated
        for field in self.classification_feature_list:
            if feature_dict[field] == "INVALID":
                return "INVALID", None

        # TODO: ensure the ordering of features is correct
        import onnxruntime as rt

        e_inputs: pd.DataFrame = pd.DataFrame([feature_dict])
        # Drop all dicom images without a SeriesNumber
        # e_inputs = e_inputs[e_inputs["SeriesNumber"].notna()]

        # Load the ONNX model
        sess: rt.InferenceSession = rt.InferenceSession(
            self.classification_model_filename.as_posix()
        )

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

        If the `mode` is set to "series," inference is performed on the entire series, and modality and acquisition
        plane information is updated for the series.

        If the `mode` is set to "volume," inference is performed for each volume within the series, and modality and
        acquisition plane information is updated for each volume.

        Args:
            None

        Returns:
            None
        Raises:
            ValueError: If an unsupported `mode` is specified.
        """

        def validate_features(input_dict: dict) -> bool:
            """
            Validate the presence of required features in the input feature dictionary.

            This function checks if all the features specified in the `classification_feature_list` are present in
            the input feature dictionary.

            Args:
                input_dict (dict): A dictionary containing features for classification.

            Returns:
                bool: True if all required features are present, False otherwise.

            """
            missing_features = []
            for feature in self.classification_feature_list:
                if feature not in input_dict.keys():
                    missing_features.append(feature)
            if len(missing_features) > 0:
                warnings.warn(
                    f"Missing features for SeriesNumber {self.series_number}: {missing_features}\n"
                    f"Series contains: {self.series.volume_info_list[0].get_one_volume_dcm_filenames()}"
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

        """
        Aggregates the modality and acquisition plane information from the volumes to the series level.
        """
        # if there is only a single volume in the series, the series modality and acquisition plane are set to the
        # volume's modality and acquisition plane
        if len(self.series.get_volume_list()) == 1:
            volume = self.series.get_volume_list()[0]
            modality = volume.get_volume_modality().replace("LOW_PROBABILITY_", "")

            self.series.set_series_modality(modality)
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
                if -12345 not in unique_bvals:
                    self._update_diffusion_series_modality()
                else:
                    self.series.set_series_modality(
                        self.series.get_volume_list()[0].get_volume_modality()
                    )
            else:
                if "pd" in unique_modalities and "t2w" in unique_modalities:
                    self.series.set_series_modality("PDT2")
                # DWI series usually contains b0 and dwig gradient volumes
                elif "dwig" in unique_modalities:
                    self._update_diffusion_series_modality()
                # TRACEW series often contains b0 and tracew volumes
                elif "tracew" in unique_modalities:
                    self.series.set_series_modality("tracew")
                elif -12345 not in unique_bvals:
                    self._update_diffusion_series_modality()
                else:
                    # if other scenarios are not met, we set the modality to the first volume's modality
                    self.series.set_series_modality(
                        self.series.get_volume_list()[0].get_volume_modality()
                    )

        # reclassify volumes with more specific information
        for volume in self.series.get_volume_list():
            volume.set_series_modality(self.series.get_series_modality())
