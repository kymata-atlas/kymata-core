[![Unit Tests Passed](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/perform-unit-tests.yml/badge.svg)](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/perform-unit-tests.yml)
[![Linting Checks Passed](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/lint-and-check-formatting.yml/badge.svg)](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/lint-and-check-formatting.yml)

# The Kymata Toolbox

This package forms part of the Kymata Atlas codebase.
Maintained by the Kymata Research Group ([kymata.org](https://kymata.org); UCL & University of Cambridge).

## Table of contents

- [Analysing a dataset](#analysing-a-dataset-with-kymata)
  1. [Locating your raw EMEG dataset](#1-locate-your-raw-emeg-dataset)
  2. [Preprocess the data](#2-preprocess-the-data)
  3. [Gridsearch](#3-run-the-gridsearch)
  4. [Plotting](#4-plot-the-results)
  5. [IPPM](#5-visualise-processing-pathways)
- [Installing](#installing)

## Analysing a dataset with Kymata

### 1. Locate your raw EMEG dataset

You'll need the following files:

- <participant_name>_run1_raw.fif
- <participant_name>_recording_config.yaml

### 2. Preprocess the data

The repository holds the Kymata preprocessing code that runs on the MRC-CBU HPC system.

This code comprises the 'Kymata back-end', including preprocessing steps, gridsearch procedures, expression plotting and IPPM generation.

Run the following invokers from `invokers/` in order:

- `invoker_run_data_cleansing.py`
  - This does:
    1. first-pass filtering 
    2. maxfiltering
    3. second-pass filtering
    4. eog removal
- `invoker_create_trialwise_data.py`
  - This does:
    1. Splits the data into trials
  - **This is all you need for sensor-space gridsearch.**
- `invoker_run_hexel_current_estimation.py`
- `invoker_estimate_noise_covariance.py`
  - This is only necessary if running the gridsearch in source space (hexels).

### 3. Run the gridsearch

- `run_gridsearch.py`
  - This will output a `.nkg` file, which can then be loaded (see `demos/demo_save_load.ipynb`).

#### Doing this at the CBU

An easier way to do this (see [Troubleshooting](docs/troubleshooting_cbu.md)) may be to use the shell script `submit_gridsearch.sh`, which sets up the Apptainer environment the right way.
Either run it locally with `./submit_gridsearch.sh`, or run it on the CBU queue with `sbatch submit_gridsearch.sh`.

### 4. Plot the results

- `invoker_run_nkg_plotting.py`

See also `demos/demo_plotting.ipynb`.

### 5. Visualise processing pathways

See `demos/demo_ippm.ipynb`.

## Installing

### Getting the toolbox

Clone this repository!

### Setting up to run with `poetry`

First, confirm you have the correct version of Python installed. Navigate to the root directory. Type
```
$ pyenv versions
```
This should confirm that python 3.11 or above is installed. If it isn't already there,
install it using `pyenv install`. You should be able to confirm
you are using the correct version using 

```
python -V
```
To install the python packages you will need to use Poetry. Assuming you have installed [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer), 
type:
```
$ poetry install
```
to load the pakages needed.

At this point, you should be able to either run the xx from the terminal
```
$ poetry run __init__.py
```
or activate in this environment in an IDE such as PyCharm.

### Run tests

To run the tests, run:
```
$ poetry run pytest
```
