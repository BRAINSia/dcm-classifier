# dcm-classifier

[![CI](https://github.com/BRAINSia/dcm-classifier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BRAINSia/dcm-classifier/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/dcm-classifier/badge/?version=latest)](https://dcm-classifier.readthedocs.io/en/latest/)
[![Coverage](https://codecov.io/gh/BRAINSia/dcm-classifier/graph/badge.svg)](https://app.codecov.io/gh/BRAINSia/dcm-classifier)
[![PyPI](https://img.shields.io/pypi/v/dcm-classifier)](https://pypi.org/project/dcm-classifier/)
[![Python versions](https://img.shields.io/pypi/pyversions/dcm-classifier)](https://pypi.org/project/dcm-classifier/)
[![Downloads](https://static.pepy.tech/badge/dcm-classifier/month)](https://pepy.tech/projects/dcm-classifier)
[![License](https://img.shields.io/github/license/BRAINSia/dcm-classifier)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://img.shields.io/badge/DOI-10.1117%2F12.3006568-blue)](https://doi.org/10.1117/12.3006568)

# Introduction

<!-- SPHINX-START -->

An MRI session produces many DICOM series, and most automated analysis tools need specific ones: a T1-weighted volume, a FLAIR, the diffusion series and its ADC map. Picking those series by hand is slow and error-prone, and it keeps a pipeline from running unattended in a clinical setting. The DICOM header's free-text fields, such as `SeriesDescription`, cannot be relied on to make the choice: the standard does not constrain them, and vendors fill them differently.

`dcm_classifier` makes that selection automatic. It reads well-characterized acquisition fields from the DICOM header, such as echo time, repetition time, flip angle, and image orientation, and identifies each volume's image modality and acquisition plane. It reads headers only, so it works with any DICOM-conforming MRI scanner, whatever the manufacturer or model.

- **Image modality** comes from a random forest (an ONNX model shipped with the package) over 28 header features. It distinguishes T1-weighted, T2-weighted, FLAIR, gradient-echo T2\*-weighted, b0, trace-weighted and gradient diffusion volumes, ADC, exponential ADC, and fractional anisotropy.
- **Acquisition plane** (axial, coronal, or sagittal) comes from a decision tree over the `ImageOrientationPatient` components. A volume is also flagged as isotropic when its pixel spacing and slice thickness agree within 10 percent.

The method paper below describes how the approach was developed and evaluated: on 257,786 MRI volumes from 60 sites and 77 scanner models, reduced to 49,261 volumes with distinct acquisition parameters, its 12-class modality model reached 98.9% accuracy in five-fold cross-validation, and its acquisition plane model 99.96%. On an independent minipig MRI dataset, modality accuracy was 95.7%. The shipped model has a different class set and feature set from the paper's, so those figures describe the published model, not this one exactly.

This project was supported by several funding sources including:

- UCSF SCOUTS RO1
- NIH-NINDS R01NS114405 and NINDS R01 NS119896
- [Botimageai](https://www.botimageai.com/).

## Paper

Click [here](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12931/3006568/DICOM-sequence-selection-for-medical-imaging-applications/10.1117/12.3006568.full#_=_) to view the published paper.

## Citing

Please reference the manuscript:

`Michal Brzus, Cavan J. Riley, Joel Bruss, Aaron Boes, Randall Jones, Hans J. Johnson, "DICOM sequence selection for medical imaging applications," Proc. SPIE 12931, Medical Imaging 2024: Imaging Informatics for Healthcare, Research, and Applications, 1293108 (2 April 2024); https://doi.org/10.1117/12.3006568`

Additionally, please reference the citations located in the [citations directory](https://github.com/BRAINSia/dcm-classifier/tree/main/citations)

## Used in

`dcm_classifier` selects the input series for these automated stroke pipelines:

- Michal Brzus, Jamie Kaminski, Trevor Viohl, Joel Bruss, Daniel Tranel, Aaron D. Boes, Hans J. Johnson, "Rapid, automated prediction of post-stroke cognitive impairment for ischemic stroke," Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis, 134071Y (2025); https://doi.org/10.1117/12.3047488
- Michal Brzus, Joseph Griffis, Cavan J. Riley, Joel Bruss, Carrie Shea, Hans J. Johnson, Aaron D. Boes, "A Clinical Neuroimaging Platform for Rapid, Automated Lesion Detection and Personalized Post-Stroke Outcome Prediction," medRxiv (2025); https://doi.org/10.1101/2025.05.09.25327310

<!-- SPHINX-END -->

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

dcm-classifier requires Python 3.12 or later.

**Clone the repository**

If you prefer to clone the git repository:

`$ git clone https://github.com/BRAINSia/dcm-classifier.git`

Navigate to the cloned repo

`$ cd <repo path>`

Setup virtual environment

`$ python3 -m venv <venv_path> && source <venv_path>/bin/activate`

Install the package and its required packages

`$ pip install .`

**Threads on shared machines and clusters**

The classifier's ONNX Runtime session uses as many threads as the process
has been granted, not as many as the machine has. Pass
`ImageTypeClassifierBase(num_threads=N)` to set the count yourself.
Otherwise it uses the smallest of these limits that it finds:

- the CPU affinity of the process (`taskset`, scheduler cpusets,
  `docker run --cpuset-cpus`);
- the cgroup CPU quota (`docker run --cpus`, Kubernetes, AWS ECS);
- batch scheduler variables: `SLURM_CPUS_PER_TASK`, `SLURM_CPUS_ON_NODE`
  (SLURM), `NSLOTS` (Sun Grid Engine), `NCPUS` (PBS Professional),
  `PBS_NUM_PPN` (Torque), and `LSB_DJOB_NUMPROC` (LSF);
- a thread budget from the caller: `OMP_NUM_THREADS` or
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`.

With none of these, ONNX Runtime chooses.

## Developer Instructions

For development, clone the repository and install the developer requirements in a virtual environment. Development allows for training of new models using the scripts directory.

`$ pip install -e ".[training]" --group dev`

The `--group` option needs pip 25.1 or newer. With uv, the equivalent is `uv sync --extra training --group dev`. The `training` extra can be left out when only the package and its tests are being changed.

Install pre-commit hooks

`$ pre-commit install`

Run pre-commit hooks to ensure code quality

`$ pre-commit run -a`

Installing the package provides the `dcm-classify-study` command. The path to a model can be omitted and the default model provided in the package will be used.

`$ dcm-classify-study -d <path_to_dicom_session>`

or pass the path to a separate model

`$ dcm-classify-study -m models/ova_rf_classifier.onnx -d <path_to_dicom_session>`

A session with a volume whose slices cannot be ordered by position (for
example a reformat whose slices differ in orientation) is refused. Add
`--allow_fallback_ordering` to accept such volumes; each is reported in the
`Order` column, and its image geometry may be wrong.

The same command is available from a source checkout as `python3 scripts/classify_study.py`.

### Limitations

Siemens mosaic images, which tile many slices into one image, are not unpacked into slices. A series whose `ImageType` contains `MOSAIC` is still classified, but its geometry and classification may be wrong. Such a series raises a warning when the study is loaded, and `get_is_mosaic()` on the series or volume returns `True`. Convert mosaic data to one slice per file before classification when the result matters.

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
