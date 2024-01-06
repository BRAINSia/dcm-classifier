#!/bin/bash

export PYTHONPATH=/home/mbrzus/programming/dicomimageclassification/src
source ~/programming/.venv/dcmdevvenv/bin/activate


cat > todo_list << EOF
IOWA_STROKE_RETRO_DICOM
IOWA_CA_JAMIE_DICOM
EOF


if [ 1 -eq 1 ]; then

for proj in $(cat todo_list); do
  data_table_file=${HOME}/dicom_data_tables/${proj}.xlsx
  mkdir -p $(dirname ${data_table_file})
  if [ -f ${data_table_file} ]; then
    echo "Skipping ${proj}"
  else
    echo ${proj} starting
    python3 create_dicom_fields_sheet.py --dicom_path /localscratch/Users/mbrzus/Stroke_Data/${proj} --out ${data_table_file}
    if [ $? -ne 0 ]; then
	    echo "FAILED PROCESSING"
            echo python3 create_dicom_fields_sheet.py --dicom_path /localscratch/Users/mbrzus/Stroke_Data/${proj} --out ${data_table_file}
	    break
    fi
    echo ${proj} stopping
  fi
done

fi


#parallel -j 2 python3 create_dicom_fields_sheet.py --dicom_path /localscratch/Users/mbrzus/Stroke_Data/{} --out ${HOME}/stroke_dicom_data_tables/{}.xlsx :::: todo_list
