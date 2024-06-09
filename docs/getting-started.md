# Getting started with the Kymata Toolbox

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
$ python -V
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Running tests, linting, and generating documentation

This will be done automatically via Github actions.

To run the tests manually, run:
```
$ poetry run pytest
```
To run linting manually, run:
```
$ poetry run ruff check
```
To serve the documentation locally, run:
```
$ poetry run mkdocs serve check
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

## Troubleshooting at the CBU

### You see `pyenv: Command not found`, `poetry: Command not found`

On the CBU nodes, `pyenv` only works in `bash`, so make sure you are using this.
```
bash
```

### You see `/lib64/libm.so.6: version 'GLIBC_2.29' not found` when running gridsearch

You are running it on a cbu node that does not have the right libraries installed. You could try it on a node which does (such as `lws-gpu02`), or (prefered) use `submit_gridsearch.sh` which will implement apptainer which layers the right libraries over the top of the node. 

### You see `ModuleNotFoundError: No module named 'numpy'`

You are probably running `submit_gridsearch.sh`, and it currently has Andy's toolbox location hard-coded.
Update to point it at your copy.

### You see `ModuleNotFoundError: No module named 'kymata'`

You're not using the poetry environment.  You'll need to run this with Apptainer.

First make sure the toolbox is installed with `poetry`, so the `kyamata` package is available within the virtual environment:

```shell
apptainer shell -B /imaging/projects/cbu/kymata /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif
export VENV_PATH=~/poetry/
cd /path/to/kymata-toolbox
$VENV_PATH/bin/poetry install
```
Now (within the Apptainer) you can run it using `poetry`, e.g.:
```shell
$VENV_PATH/bin/poetry run python invokers/invoker_create_trialwise_data.py
```

## Hyperparameter Selection and Denoising Experiments for Denoising Strategies

TODO:
    - Push scaler changes
    - Test hyperparameter settings for each system
    - Build automated/manual hyperparameter selector

### Introduction

Our objective in this doc is two-fold:
1) Evaluate various denoising strategies
2) Find heuristics and methods for hyperparameter selection

To evaluate the denoising strategies, we need to identify prototypical clustering problems and examine how each strategy fares. Our experiments will focus on the right hemisphere and the following functions will be investigated.

#### Prototypical Sparse Cluster: TVL loudness (short-term)

![TVL loudness (short-term)](assets/images/tvl_l_short.png)

#### Prototypical Multiple Peaks Cluster: TVL loudness chan 2 (instantaneous)

![TVL loudness chan 2 (instantaneous)](assets/images/tvl_l_chan2_instant.png)

#### Prototypical Blurred Cluster: Heeger horizontal ME GP2

![Heeger horizontal ME GP2](assets/images/heeger_hori_me_gp2.png)

### Max Pooling

Try varying the bin size and the significant clusters

### Gaussian Mixture Model

Scaled vs Not scaled (at the end)
BIC vs AIC (BIC is suggested as better for explanatory models)

### DBSCAN

Scaling appears to decrease the performance of DBSCAN. On the x-axis, we have a distance of 5 between successive data points; on the y-axis, the differences are much more minute, ranging from 10^-10 to 10^-100. One can interpret this as the latency dimension having a higher weight than the magnitude dimension. 

Hence, any points on the same x-axis timeframe will be counted in the same radius, which is what we want. 

The performance of DBSCAN depends upon the size of the "thinnest" cluster in the database. In other words, if we can identify the values of minPts and eps that identify the thinnest cluster as significant, we are done. The method to achieve this computes the 4-dist (4th closest point) for every point and plots a 4-dist graph. The graph helps us identify the thinnest cluster by showing where the 4-dist increases substantially, highlighting that these points are sparse. Experiments from [1] show that setting minEps to 4 is optimal since increasing minEps past 4 does not lead to markedly different results but does increase the computational overhead. 

### Mean Shift

Scaled vs Not scaled
estimate bandwidth works fine

### References

[1] - https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf