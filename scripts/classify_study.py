#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from dcm_classifier.classify_support import simple_classify_study

try:
    from dcm_classifier.image_type_inference import ImageTypeClassifierBase
except Exception as e:
    print(f"Missing module import {e}")
    print(
        f"Try setting export PYTHONPATH={Path(__file__).parent.parent.as_posix()}/src"
    )
    sys.exit(255)


def main():
    # Set up argparse
    description = (
        "Authors: Michal Brzus, Hans J. Johnson\n"
        "Classify image modality and acquisition plane of DICOM MR data by series and volumes.\n"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--session_directory",
        required=True,
        help="Path to the patient session directory with dicom data",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        help="Path to the model used for image type inference",
    )
    parser.add_argument(
        "-n",
        "--nifti_dir",
        required=False,
        default=None,
        help="Path to the directory where the NIFTI files are stored for each volume",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        help="Path to the output the newly organized dicom data",
    )
    parser.add_argument(
        "-j",
        "--json",
        required=False,
        default=None,
        help="Path to the output json file",
    )

    args = parser.parse_args()
    nifti_dir: Path | None = Path(args.nifti_dir) if args.nifti_dir else None
    output_dir: Path | None = Path(args.output) if args.output else None
    if args.model is None:
        inferer = ImageTypeClassifierBase()
    else:
        inferer = ImageTypeClassifierBase(classification_model_filename=args.model)
    session_directory: Path = Path(args.session_directory)
    json_dumppath: Path | None = Path(args.json) if args.json else None
    print(description)

    simple_classify_study(
        session_directory, inferer, json_dumppath, output_dir, nifti_dir
    )


if __name__ == "__main__":
    # Execute the script
    main()
