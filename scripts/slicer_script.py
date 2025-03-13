import os
import re
import sys

import slicer
from pathlib import Path
import qt

"""
NOTE: This script is meant to run by running the slicer startup with the `--python-script` flag. This script will load after slicer is loaded

for example;

` ./Slicer-5.2.2-linux-amd64/Slicer --python-script scripts/slicer_script.py`

"""

# Update this path to point to your input directory
input_dir = ""

# Compile a regex for detecting mask files (case-insensitive)
mask_regex = re.compile(r"mask", re.IGNORECASE)


def main():
    print("Starting Slicer script..., loading images from directory:", input_dir)
    slicer.util.selectModule(
        "ViewControllers"
    )  # Ensure the ViewControllers module is selected
    # Iterate over each subject folder in the input directory
    for subject in sorted(list(Path(input_dir).iterdir())):
        print(f"Processing subject: {subject.stem}")
        if subject.is_dir():
            # Clear the scene so only the current subject's images are shown
            slicer.mrmlScene.Clear(False)
            print(f"Loading images for subject: {subject.stem}")
            if subject.name == "bids":
                print("Skipping 'bids' directory")
                continue

            data_dir = subject
            if not data_dir.exists():
                print(f"Subject '{subject.stem}' has no 'data' folder. Skipping.")
                continue

            # Load all images for the current subject
            for image in data_dir.iterdir():
                name = image.stem  # derive a name from the file stem
                if mask_regex.search(image.name):
                    print(f"Loading '{image.name}' as label volume")
                    slicer.util.loadLabelVolume(str(image))
                else:
                    print(f"Loading '{image.name}' as volume")
                    slicer.util.loadVolume(str(image))
            slicer.app.layoutManager().setLayout(
                slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView
            )
            slicer.app.layoutManager().threeDWidget(0).hide()

            while True:
                user_input = (
                    input(
                        f"Subject '{subject.stem}' loaded.\nType 'n' to continue to the next subject, 'e' to exit: "
                    )
                    .strip()
                    .lower()
                )
                if user_input.lower() == "n":
                    break
                elif user_input.lower() == "e":
                    print("User canceled further processing. Exiting.")
                    sys.exit(0)
                else:
                    print("Invalid input. Please type 'next' or 'exit'.")

    print("Slicer script completed.")


if __name__ == "__main__":
    main()
