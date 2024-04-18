:orphan:

=========================
Installing dcm-classifier
=========================

Install Using Pip
-----------------

The dcm-classifier package is available on PyPi and can be installed using pip using the following command::

    pip install dcm-classifier

This option will install all the necessary dependencies for the package to run. For more control, we recommend cloning the repository and installing the package in development mode.

Cloning the Repository
----------------------

The dcm-classifier repository is hosted on GitHub `here <https://github.com/BRAINSia/dcm-classifier>`_. You can clone the repository using the following command::

    git clone git@github.com:BRAINSia/dcm-classifier.git


For Developers:

Developers will likely want to set up a python virtual environment with the necessary dependencies. This can be done using the following commands::

    cd dcm-classifier
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements-dev.txt

For Users:

Users will want to set up a virtual environment with the necessary dependencies. This can be done using the following commands::

    cd dcm-classifier
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt


Tutorials
---------

The dcm-classifier repository also provides a set of tutorials for running the classifier as well as for training new models. These jupyter notebook tutorials are available in scripts directory of the repository
