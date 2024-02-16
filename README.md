# Introduction

In this work, we developed a robust, easily extensible classification framework that extracts key features from well-characterized DICOM header fields for identifying image modality and acquisition plane. Our tool is crucial for eliminating error-prone human interaction and allowing automatization, increasing imaging applications' reliability and efficiency. We used Random Forrest and Decision Tree algorithms to determine the image modality and orientation. We trained on header meta-data of over 49000 scan volumes from multiple studies and achieved over 99% prediction accuracy on image modality and acquisition plane classification. The proposed framework will be available as an open-source python pip module for easy and fast adaptation.

This work was submitted for publication at the 2024 [SPIE Medical Imaging](https://spie.org/conferences-and-exhibitions/medical-imaging?SSO=1) conference.

This project was supported by [Botimageai](https://www.botimageai.com/).

# Instructions

Below are useful commands to start using the tool.

Clone git repo

`$ git clone https://github.com/BRAINSia/dcm-classifier.git`

Navigate to the cloned repo

`$ cd <repo path>`

Setup virtual environment

`$ python3 -m venv <venv_path> && source <venv_path>/bin/activate`


Install required packages

`$ pip install -r requirements.txt`

Run the script!

`python3 classify_study.py -m models/rf_classifier.onnx -d <path_to_dicom_session>`

# Testing
```bash
  pytest
# or to fail on warnings
  python3 -Werror::FutureWarning -m pytest
```
# Coverage Analysis
```bash
 coverage run --concurrency=multiprocessing --parallel-mode -m pytest tests --junitxml=tests/pytest.xml
 coverage combine
 coverage report --format=text -m |tee tests/pytest-coverage.txt
 coverage xml -o tests/coverage.xml
 coverage xml -o tests/coverage.xml
```

# Authors

1. **Michal Brzus**

    github: [mbrzus](https://github.com/mbrzus), email: michal-brzus@uiowa.edu

2. [**Hans J. Johnson**](https://engineering.uiowa.edu/people/hans-johnson)

    github: [BRAINSia](https://github.com/BRAINSia), email: hans-johnson@uiowa.edu
