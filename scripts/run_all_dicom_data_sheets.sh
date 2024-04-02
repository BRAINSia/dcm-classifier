#!/bin/bash

echo To use this script, remove the multiline comment and fill in paths as needed.

# export PYTHONPATH=<PATH_TO_dcm-classifier_SRC_DIR>
# source <PATH_TO_PYTHON_VENV>/bin/activate
#
#
# cat > todo_list << EOF
# list
# of
# projects
# to
# process
# EOF


# while IFS= read -r proj; do
#   data_table_file=${HOME}/<PATH_TO_DATA_OUTPUTS>/${proj}.xlsx
# #   shellcheck disable=SC2086
#   mkdir -p "$(dirname ${data_table_file})"
#   if [ -f "${data_table_file}" ]; then
#     echo "Skipping ${proj}"
#   else
#     echo "${proj} starting"
#     if python3 create_dicom_fields_sheet.py --dicom_path "PATH_TO_PROJECTS/${proj}" --out "${data_table_file}" ; then
# 	    echo "SUCCESS PROCESSING"
# 	    echo "FAILED PROCESSING"
#             echo "python3 create_dicom_fields_sheet.py --dicom_path PATH_TO_PROJECTS/${proj} --out ${data_table_file}"
# 	    break
#     fi
#     echo "${proj} stopping"
#   fi
# done < todo_list
