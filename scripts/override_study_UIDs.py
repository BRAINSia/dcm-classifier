#!/usr/bin/env python3

from dcm_classify.study_processing import ProcessOneDicomStudyToVolumesMappingBase
import pydicom


def override_study_UIDs(session_dir: str):
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_dir, inferer=None
    )

    study_uid = pydicom.uid.generate_uid()
    for series_number, series in study.series_dictionary.items():
        series_uid = pydicom.uid.generate_uid()
        for volume in series.get_volume_list():
            for file in volume.get_one_volume_dcm_filenames():
                ds = pydicom.dcmread(file)
                # Override the UIDs attribute
                ds.StudyInstanceUID = study_uid
                ds.SeriesInstanceUID = series_uid
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                # Save the modified DICOM file
                ds.save_as(file)


if __name__ == "__main__":
    p = "/localscratch/Users/mbrzus/Stroke_Data/override_test_for_Hans/XY001_MRI_BRAIN_WO_CONTRAST__2996202614097427"
    override_study_UIDs(p)
