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

from .dicom_series import DicomSingleSeries
from .namic_dicom_typing import (
    itk_read_from_dicomfn_list,
    infer_diffusion_from_gradient,
)
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from typing import Dict, List, Union, Optional, Any


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
        classification_model_filename: Union[str, Path],
        classification_feature_list: List[str] = modality_columns,
        image_type_map: Dict[str, int] = imagetype_to_integer_mapping,
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
        self.classification_model_filename: Union[str, Path] = Path(
            classification_model_filename
        )
        self.classification_feature_list: List[str] = classification_feature_list
        self.imagetype_to_int_map: Dict[str, int] = image_type_map
        self.int_to_imagetype_map: Dict[int, str] = self.get_int_to_type_map()
        self.min_probability_threshold: float = min_probability_threshold
        self.series: Optional[DicomSingleSeries] = None
        self.series_number: Optional[int] = None
        self.info_dict: Optional[Dict[str, Any]] = None

    def get_int_to_type_map(self) -> dict:
        """
        Get the integer to image type mapping.

        Returns:
            dict: Dictionary mapping integers to image type names.
        """
        return {v: k for k, v in self.imagetype_to_int_map.items()}

    def check_if_diffusion(self) -> None:
        """
        After processing and file organization into volumes and serieses is completed, this function is called to
        check if the series is a diffusion series. If so, the series modality is overriden to dwig.
        This function is called during inference
        """
        volume_list = self.series.get_volume_list()
        first_dcm_per_volume = [
            volume.get_one_volume_dcm_filenames()[0] for volume in volume_list
        ]
        diff_modality = infer_diffusion_from_gradient(first_dcm_per_volume)
        self.series.set_modality(diff_modality)

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

    def infer_isotropic(self, feature_dict: dict = None) -> Optional[bool]:
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
        # check if the volume was invalidated
        for field in self.classification_feature_list:
            if feature_dict[field] == "INVALID":
                return "INVALID", None

        import onnxruntime as rt

        e_inputs: pd.DataFrame = pd.DataFrame([feature_dict])
        # Drop all dicom images without a SeriesNumber
        e_inputs = e_inputs[e_inputs["SeriesNumber"].notna()]

        # Load the ONNX model
        sess: rt.InferenceSession = rt.InferenceSession(
            self.classification_model_filename.as_posix()
        )

        input_name: str = sess.get_inputs()[0].name
        label_name: str = sess.get_outputs()[0].name
        prob_name: str = sess.get_outputs()[1].name

        full_outputs: pd.DataFrame = pd.DataFrame()
        full_outputs["SeriesNumber"] = [self.series_number]
        model_inputs: pd.DataFrame = e_inputs[self.classification_feature_list]
        try:
            numeric_inputs: np.array = model_inputs.astype(np.float32).to_numpy()
        except ValueError:
            # Short circuit if the inputs are not sufficient for inference
            full_outputs["GUESS_ONNX"] = "InvalidDicomInputs"
            return "INVALID", full_outputs

        pred_onx_run_output = sess.run(
            [label_name, prob_name], {input_name: numeric_inputs}
        )
        pred_onx = pred_onx_run_output[0]
        probability_onx = pred_onx_run_output[1]
        full_outputs["GUESS_ONNX_CODE"] = pred_onx
        full_outputs["GUESS_ONNX"] = pred_onx

        for type_name, type_integer_code in self.imagetype_to_int_map.items():
            full_outputs["GUESS_ONNX"].replace(
                to_replace=type_integer_code, value=type_name, inplace=True
            )

        for col_index, class_probability in probability_onx[0].items():
            col_name = self.int_to_imagetype_map[int(col_index)]
            class_name: str = f"GUESS_ONNX_{col_name}"
            full_outputs[class_name] = class_probability

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
            features_validated: bool = validate_features(volume.get_volume_info_dict())
            if features_validated:
                acquisition_plane: str = self.infer_acquisition_plane(
                    feature_dict=volume.get_volume_info_dict()
                )
                volume.set_acquisition_plane(acquisition_plane)
                isotropic = self.infer_isotropic(feature_dict=self.info_dict)
                self.series.set_is_isotropic(isotropic)

                if volume.get_modality() == "dwig":
                    volume.set_modality_probabilities(pd.DataFrame())
                else:
                    modality, full_outputs = self.infer_modality(
                        feature_dict=volume.get_volume_info_dict()
                    )
                    volume.set_modality(modality)
                    volume.set_modality_probabilities(
                        pd.DataFrame(full_outputs, index=[0])
                    )

        # classify series
        if len(self.series.get_volume_list()) == 1:
            volume = self.series.get_volume_list()[0]
            self.series.set_modality(volume.get_modality())
            self.series.set_modality_probabilities(volume.get_modality_probabilities())
            self.series.set_acquisition_plane(volume.get_acquisition_plane())
            self.series.set_is_isotropic(self.series.get_is_isotropic())
        else:
            self.check_if_diffusion()
            self.series.set_acquisition_plane(
                self.series.get_volume_list()[0].get_acquisition_plane()
            )
            self.series.set_is_isotropic(
                self.series.get_volume_list()[0].get_is_isotropic()
            )
