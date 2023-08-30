# Introduction

In this work, we developed a robust, easily extensible classification framework that extracts key features from well-characterized DICOM header fields to identify image modality and acquisition plane. 
Utilizing classical machine learning paradigms and a heterogeneous dataset of 9121 scans collected at 12 sites, using 23 scanners from 6 manufacturers, we achieved 99.4% accuracy during the K-Fold Cross-Validation for classifying 11 image modalities and 99.96% accuracy on image acquisition plane classification. Furthermore, we demonstrated model generalizability by achieving 98.6% accuracy on out-of-sample animal data. Our proposed framework can be crucial in eliminating error-prone human interaction, allowing automatization, and increasing imaging applications' reliability and efficiency.

This work was submitted for publication at the 2024 [SPIE Medical Imaging](https://spie.org/conferences-and-exhibitions/medical-imaging?SSO=1) conference.

This project was supported by [Botimageai](https://www.botimageai.com/).

# Instructions

Below are useful commands to start using the tool.

Clone git repo

`$ git clone https://research-git.uiowa.edu/SINAPSE/dicomimageclassification.git`

Navigate to the cloned repo

`$ cd <repo path>`

Setup virtual environment

`$ python3 -m venv <venv_path> && source <venv_path>/bin/activate`


Install required packages

`$ pip install -r REQUIREMENTS.txt`

Run the script!

`python3 classify_study.py -m models/rf_classifier.onnx -d <path_to_dicom_session>`

# Authors

1. **Michal Brzus**

    github: [mbrzus](https://github.com/mbrzus), email: michal-brzus@uiowa.edu

2. [**Hans J. Johnson**](https://engineering.uiowa.edu/people/hans-johnson)

    github: [BRAINSia](https://github.com/BRAINSia), email: michal-brzus@uiowa.edu
