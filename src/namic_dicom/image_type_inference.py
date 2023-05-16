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

from src.namic_dicom.dicom_series import DicomSingleSeries
from src.namic_dicom.namic_dicom_typing import itk_read_from_dicomfn_list
from pathlib import Path

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
    Inference class.
    """

    def __init__(
        self,
        classification_model_filename: Union[str, Path],
        classification_feature_list: List[str] = modality_columns,
        image_type_map: Dict[str, int] = imagetype_to_integer_mapping,
        mode: str = "series",
        min_probability_threshold: float = 0.4,
    ):
        """
        Args:
            classification_model_filename: path to the classification model file
            classification_feature_list: list of features used for classification
            image_type_map: map between class name and model integer output
            mode: "series" or "volume" to run inference on series or volume level
        """
        self.classification_model_filename: Union[str, Path] = Path(
            classification_model_filename
        )
        self.classification_feature_list: List[str] = classification_feature_list
        self.imagetype_to_int_map: Dict[str, int] = image_type_map
        self.int_to_imagetype_map: Dict[int, str] = self.get_int_to_type_map()
        self.mode: str = mode
        self.min_probability_threshold: float = min_probability_threshold
        self.series: Optional[DicomSingleSeries] = None
        self.series_number: Optional[int] = None
        self.info_dict: Optional[Dict[str, Any]] = None

    def get_int_to_type_map(self) -> dict:
        return {v: k for k, v in self.imagetype_to_int_map.items()}

    def set_series(self, series: DicomSingleSeries) -> None:
        self.series = series
        self.series_number = series.get_series_number()
        self.info_dict = self.series.get_series_info_dict()

    def infer_acquisition_plane(self, feature_dict: dict = None) -> str:
        """
        This is implementation of the decision tree for acquisition plane inference.
        It takes the ImageOrientationPatient_0, ImageOrientationPatient_5 from info_dict and returns the
        acquisition plane prediction.
        Returns:
        """
        volume = self.series.get_volume_list()[0]
        itk_im = itk_read_from_dicomfn_list(volume.get_one_volume_dcm_filenames())
        spacing = list(itk_im.GetSpacing())
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        cube_root = voxel_volume ** (1 / 3)
        is_iso = True
        for s in spacing:
            if abs(s - cube_root) / cube_root > 0.10:
                is_iso = False
        if is_iso:
            return "iso"

        if float(feature_dict["ImageOrientationPatient_5"]) <= 0.5:
            return "ax"
        else:
            if float(feature_dict["ImageOrientationPatient_0"]) <= 0.707:
                return "sag"
            else:
                return "cor"

    def infer_modality(self, feature_dict: dict = None) -> (str, pd.DataFrame):
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

        model_inputs: pd.DataFrame = e_inputs[self.classification_feature_list]
        numeric_inputs: np.array = model_inputs.astype(np.float32).to_numpy()
        pred_onx_run_output = sess.run(
            [label_name, prob_name], {input_name: numeric_inputs}
        )
        pred_onx = pred_onx_run_output[0]
        probability_onx = pred_onx_run_output[1]
        full_outputs: pd.DataFrame = pd.DataFrame()
        full_outputs["SeriesNumber"] = [self.series_number]
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
        if max(probability_onx[0].values()) < self.min_probability_threshold:
            image_type = "unknown"
        else:
            image_type: str = full_outputs.iloc[
                0, full_outputs.columns.get_loc("GUESS_ONNX")
            ]

        return image_type, full_outputs

    def run_inference(self):
        # validate features
        def validate_features(input_dict: dict) -> bool:
            missing_features = []
            for feature in self.classification_feature_list:
                if feature not in input_dict.keys():
                    missing_features.append(feature)
            if len(missing_features) > 0:
                print(
                    f"Missing features for SeriesNumber {self.series_number}: {missing_features}\n"
                    f"Series contains: {self.series.volume_info_list[0].get_one_volume_dcm_filenames()}"
                )
                return False
            return True

        if self.mode == "series":
            if validate_features(self.info_dict):
                acquisition_plane = self.infer_acquisition_plane(
                    feature_dict=self.info_dict
                )
                modality, full_outputs = self.infer_modality(
                    feature_dict=self.info_dict
                )
                self.series.set_modality(modality)
                self.series.set_acquisition_plane(acquisition_plane)
                self.series.set_modality_probabilities(
                    pd.DataFrame(full_outputs, index=[0])
                )
        elif self.mode == "volume":
            for volume in self.series.get_volume_list():
                if validate_features(volume.get_volume_info_dict()):
                    acquisition_plane = self.infer_acquisition_plane(
                        feature_dict=volume.get_volume_info_dict()
                    )
                    modality, full_outputs = self.infer_modality(
                        feature_dict=volume.get_volume_info_dict()
                    )
                    volume.set_modality(modality)
                    volume.set_acquisition_plane(acquisition_plane)
                    volume.set_modality_probabilities(
                        pd.DataFrame(full_outputs, index=[0])
                    )
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
