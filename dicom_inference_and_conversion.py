import itk
import pandas as pd
from pathlib import Path
from glob import glob
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.namic_dicom_typing import *
import tempfile
import os
import re
import shutil
from subprocess import run


def get_bids_mri_type(modality: str) -> str:
    if modality.lower() in ["adc", "tracew", "dwig", "dwi_multishell", "fa"]:
        return "dwi"
    else:
        return "anat"


def create_nifti_file(
    dcm2niix_path: str,
    subject: str,
    session: str,
    modality: str,
    plane: str,
    output_dir: str,
    dcm_dir: str,
    desc: str = None,
) -> None:
    run = 1
    if desc is not None:
        fname = f"{subject}_{session}_run-00{run}_acq-{plane}_desc-{desc}_{modality}"
    else:
        fname = f"{subject}_{session}_run-00{run}_acq-{plane}_{modality}"
    while os.path.exists(f"{output_dir}/{fname}.nii.gz"):
        fname = fname.replace(f"run-00{run}", f"run-00{run+1}")
        run += 1
    os.system(f"{dcm2niix_path} -o {output_dir} -f {fname} -z y {dcm_dir}")
    print(f"{dcm2niix_path} -o {output_dir} -f {fname} -n -z y {dcm_dir}")


def dicom_inference_and_conversion(
    session_dir: str, output_dir: str, model_path: str, dcm2niix_path: str
) -> str:
    """
    This function takes a session directory with dicom data and converts them to NIfTI files in BIDS format.
    :param session_dir: path to the session directory with dicom data
    :param output_dir: path to the output directory (base NIfTI directory)
    :param model_path: path to the model used for image type inference
    :param dcm2niix_path_path: path to the dcm2niix script
    :return: path to the NIfTI sub-*/ses-* directory with converted data
    """
    inferer = ImageTypeClassifierBase(
        classification_model_filename=model_path,
    )
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_dir, inferer=inferer
    )
    study.run_inference()
    for series_number, series in study.series_dictionary.items():
        info_dict = series.get_series_info_dict()
        sub = info_dict["PatientID"]
        ses = info_dict["AcquisitionDate"]
        sub_ses_dir = f"{output_dir}/sub-{sub}/ses-{ses}"
        if not os.path.exists(sub_ses_dir):
            run(["mkdir", "-p", sub_ses_dir])

        plane = series.get_acquisition_plane()
        modality = series.get_modality()
        if modality is None:
            modality = "unknown"
        if modality.lower() == "t2starw":
            continue
        dir_type = get_bids_mri_type(modality)
        final_dir_path = f"{sub_ses_dir}/{dir_type}"
        if not os.path.exists(final_dir_path):
            run(["mkdir", "-p", final_dir_path])
        if modality == "tracew":
            # TODO: Consider also saving the original 4D tracew image
            tracew_im_list = []
            b_val_list = []
            for volume in series.get_volume_list():
                b_value = volume.get_volume_bvalue()
                b_val_list.append(b_value)

                file_list = volume.get_one_volume_dcm_filenames()
                tracew_im_list.append(itk_read_from_dicomfn_list(file_list))
                temp_dir = tempfile.mkdtemp()
                file_list = volume.get_one_volume_dcm_filenames()
                for file in file_list:
                    shutil.copy(file, temp_dir)

                create_nifti_file(
                    dcm2niix_path=dcm2niix_path,
                    subject=sub,
                    session=ses,
                    modality=modality,
                    plane=plane,
                    output_dir=final_dir_path,
                    dcm_dir=temp_dir,
                    desc=f"postB{b_value}",
                )
            try:
                adc = compute_adc_from_multi_b_values(tracew_im_list, b_val_list)
                adc_run = 1
                fname = f"{sub}_{ses}_run-00{adc_run}_acq-{plane}_desc-post_adc"
                while os.path.exists(f"{final_dir_path}/{fname}.nii.gz"):
                    fname = fname.replace(f"run-00{adc_run}", f"run-00{adc_run + 1}")
                    adc_run += 1
                itk.imwrite(adc, f"{final_dir_path}/{fname}.nii.gz")
            except RuntimeError:
                pass
            except ZeroDivisionError:
                pass
        else:
            for volume in series.get_volume_list():
                temp_dir = tempfile.mkdtemp()
                file_list = volume.get_one_volume_dcm_filenames()
                for file in file_list:
                    shutil.copy(file, temp_dir)

                create_nifti_file(
                    dcm2niix_path=dcm2niix_path,
                    subject=sub,
                    session=ses,
                    modality=modality,
                    plane=plane,
                    output_dir=final_dir_path,
                    dcm_dir=temp_dir,
                )
    return sub_ses_dir
