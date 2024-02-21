#!/bin/bash

export PYTHONPATH=/johnsonhj_archive_00/predict_hd_dicoms/dicomimageclassification/src
source ~/.venv/dcmclassify/bin/activate


cat > todo_list << EOF
PHD_073
PHD_017
PHD_PET_024
PHD_DTI_THP
fcMRI_024
fcMRI_120
fMRI-024
FMRI_024_PILOT
fMRI-120
FMRI_120_PILOT
fMRI_COMPAT
XFMRI_HD_024
MRI_HD_120
GN_024
HD_BAD_DATA
HD_GENE_024
HDPILOT
JHD_024
MARKERS_024
NAMIC_HD
PET_PILOT
PHD_001
PHD_002
PHD_007
PHD_017
PHD_024
PHD_027
PHD_028
PHD_029
PHD_030
PHD_032
PHD_039
PHD_041
PHD_045
PHD_048
PHD_050
PHD_052
PHD_054
PHD_061
PHD_071
PHD_073
PHD_083
PHD_096
PHD_120
PHD_144
PHD_156
PHD_159
PHD_175
PHD_176
PHD_177
PHD_178
PHD_179
PHD_180
PHD_181
PHD_DTI_THP
PHD_PET_024
SYM_024
EOF


while IFS= read -r proj; do
  data_table_file=${HOME}/dicom_data_tables/${proj}.xlsx
  # shellcheck disable=SC2086
  mkdir -p "$(dirname ${data_table_file})"
  if [ -f "${data_table_file}" ]; then
    echo "Skipping ${proj}"
  else
    echo "${proj} starting"
    if python3 create_dicom_fields_sheet.py --dicom_path "/johnsonhj_archive_00/predict_hd_dicoms/data/archive/${proj}" --out "${data_table_file}" ; then
	    echo "SUCCESS PROCESSING"
	    echo "FAILED PROCESSING"
            echo "python3 create_dicom_fields_sheet.py --dicom_path /johnsonhj_archive_00/predict_hd_dicoms/data/archive/${proj} --out ${data_table_file}"
	    break
    fi
    echo "${proj} stopping"
  fi
done < todo_list

#parallel -j 2 python3 create_dicom_fields_sheet.py --dicom_path /johnsonhj_archive_00/predict_hd_dicoms/data/archive/{} --out ${HOME}/dicom_data_tables/{}.xlsx :::: todo_list
