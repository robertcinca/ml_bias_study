# ml_bias_explainability
A Python library that helps to explain model bias.

To set up the tool, please follow the steps below.

## 1) Environments Setup

Please see below for common environment setup options. Feel free to use your own too. The code has been tested in Python 3.7, 3.8 and 3.9.

### OPT 1: Using conda

If you have Conda installed, you can use the following steps to install the required packages:

1) To create the environment: `conda create --name ml_bias python=3.7 --no-default-packages`

2) To enter the environment: `conda activate ml_bias`

3) To install the required packages: `conda install -c conda-forge --file requirements.txt`

#### Troubleshooting

1) Installation takes too long or Conda can't resolve packages. Try installing things directly with pip from within the Conda environment:

`pip install -r requirements.txt`

2) There may be an error with finding the package Cufflinks. That's because in Conda the package is known as `cufflinks-py` instead of `cufflinks`. You can either modify the `requirements.txt` file or run the following one-liner to install it:

` conda install -c conda-forge cufflinks-py `

### OPT 2: Using Pipenv

This repository also uses `pipenv` for dependency management. Please follow these steps to get this set up:

1) Fristly install Pipenv: ``pip install --user pipenv``

2) To create/enter the virtual environment: ``pipenv shell``

3) To install the packages: ``pipenv install``

Please note that it may take a while for packages to be installed.

If you get an error that says pipenv is not found, run the above commands with `python -m` in front.

### OPT 3: Using requirements.txt

Alternatively, please feel free to use your own setup preference, and installed the packages defined in either `requirements.txt` or `requirements_full.txt`:

1) `pip install -r requirements.txt`

The `requirements.txt` file contains the list of essential packages using Python 3.9. Installing these packages will lead to other packages that are called by these packages to be installed.

The `requirements_full.txt` file was created with the full list of packages that were installed, using Python 3.9. 

Note that this was tested in Python 3.9. The packages should be compatible with Python Versions 3.7, 3.8, 3.9 and 3.10. For other versions of Python, the package versions may need some modification.

## 2) Running the Tutorial using Jupyter Notebook

To start up the tutorial, please run the following commands from within the newly created environment.

1) To set up the jupyter kernel: ``python -m ipykernel install --user --name=ml_explainability_bias``

2) To start the jupyter notebook, run this from the command line: ``jupyter notebook``

3) You will then need to go to ``Kernel-->Change kernel`` in the menu bar and select ``ml_explainability_bias``.

4) Once this is setup, in future you can just enter your environment (e.g., by running ``pipenv shell`` or ``conda activate ml_bias``) followed by ``jupyter-notebook``.

5) The tutorial is located in the root directory, and is called `introduction_to_bias.ipynb`.
