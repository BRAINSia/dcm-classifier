# Introduction

In this work, we developed a robust, easily extensible classification framework that extracts key features from well-characterized DICOM header fields for identifying image modality and acquisition plane. Our tool is crucial for eliminating error-prone human interaction and allowing automatization, increasing imaging applications' reliability and efficiency. We used Random Forrest and Decision Tree algorithms to determine the image modality and orientation. We trained on header meta-data of over 49000 scan volumes from multiple studies and achieved over 99% prediction accuracy on image modality and acquisition plane classification.

This project was supported by several funding sources including:

 - UCSF SCOUTS RO1
 - NIH-NINDS R01NS114405 and NINDS R01 NS119896
 - [Botimageai](https://www.botimageai.com/).

## Paper

Click [here](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12931/3006568/DICOM-sequence-selection-for-medical-imaging-applications/10.1117/12.3006568.full#_=_) to view the published paper.


##  Citing
Please reference the manuscript:

`Michal Brzus, Cavan J. Riley, Joel Bruss, Aaron Boes, Randall Jones, Hans J. Johnson, "DICOM sequence selection for medical imaging applications," Proc. SPIE 12931, Medical Imaging 2024: Imaging Informatics for Healthcare, Research, and Applications, 1293108 (2 April 2024); https://doi.org/10.1117/12.3006568`

Additionally, please reference the citations located in the [citations directory](https://github.com/BRAINSia/dcm-classifier/tree/main/citations)

# Instructions

Below are instructions for installing and using the package as a user and developer.

## Documentation

The documentation for the package can be found [here](https://dcm-classifier.readthedocs.io/en/latest/index.html)

## Tutorials

Tutorial notebooks are provided in the scripts directory for training and using the classifier along with all the necessary scripts for training a custom model.

## User Instructions

**Pip install**

Users have the ability to simply pip install the package which will install the classifier and all necessary dependencies to run the classifier for ease of use. This will also provide the user with the pretrained model for classification.

`$ pip install dcm-classifier`

**Clone the repository**

If you prefer to clone the git repository:

`$ git clone https://github.com/BRAINSia/dcm-classifier.git`

Navigate to the cloned repo

`$ cd <repo path>`

Setup virtual environment

`$ python3 -m venv <venv_path> && source <venv_path>/bin/activate`

Install required packages

`$ pip install -r requirements.txt`

## Developer Instructions

For development, clone the repository and install the developer requirements in a virtual environment. Development allows for training of new models using the scripts directory.

`$ pip install -r requirements_dev.txt`

Install pre-commit hooks

`$ pre-commit install`

Run pre-commit hooks to ensure code quality

`$ pre-commit run -a`

Run the classify study script, the path to a model can be omitted and the default model provided in the package will be used.

`$ python3 <path_to_scripts_directory>/classify_study.py -d <path_to_dicom_session>`

or pass the path to a separate model

`$ python3 <path_to_scripts_directory>/classify_study.py -m models/ova_rf_classifier.onnx -d <path_to_dicom_session>`

### Testing

Testing in the dcm-classifier package is done using pytest. To run the tests, navigate to the root directory of the package

The testing data is stored in Git LFS so the following commands will be needed before running pytest

```bash
  git lfs fetch
  git lfs checkout
```

and now

```bash
  pytest
  # or to fail on warnings
  python3 -Werror::FutureWarning -m pytest
```
### Coverage Analysis

To run coverage analysis, navigate to the root directory of the package and run the following commands:
```bash
 coverage run --concurrency=multiprocessing --parallel-mode -m pytest tests --junitxml=tests/pytest.xml
 coverage combine
 coverage report --format=text -m |tee tests/pytest-coverage.txt
 coverage xml -o tests/coverage.xml
 coverage xml -o tests/coverage.xml
```
### Contributing

We welcome contributions from the community! Before getting started, please take a moment to review our [Contribution Guidelines](CONTRIBUTING.md) for instructions on how to contribute to this project. Whether you're fixing a bug, implementing a new feature, or improving documentation, your contributions are greatly appreciated!



# FAQs

1. **What is the purpose of this package?**

    The purpose of this package is to provide a tool for classifying DICOM images based on their header information. This tool can be used to automate the classification process and eliminate human error.


2. **What are the key features of this package?**

    The key features of this package include:
    - Classification of DICOM images based on header information
    - Automated classification process
    - Elimination of human error


3. **What are the future plans for this package and how can I contribute?**

    The future plans for this package include:
    - Adding support for more image modalities
    - Improving the classification accuracy
    - Adding support for more DICOM header fields

# Authors

1. **Michal Brzus**

    github: [mbrzus](https://github.com/mbrzus), email: michal-brzus@uiowa.edu

2. [**Hans J. Johnson**](https://engineering.uiowa.edu/people/hans-johnson)

    github: [BRAINSia](https://github.com/BRAINSia), email: hans-johnson@uiowa.edu

3. **Cavan Riley**

    github: [CavRiley](https://github.com/CavRiley), email: cavan-riley@uiowa.edu
