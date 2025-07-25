# Getting started with Kymata Core

## Set Up

This provides an overview of how to set up _Kymata Core_ locally.

Please be aware that this codebase is released publicly to ensure the transparency of the results in the [Kymata Atlas](https://kymata.org). While 
we welcome users using this codebase, we are unable to prioritise installation support.


### Prerequisites

* **Python**

   Confirm you have the correct version of Python installed. Type
   ```sh
   $ pyenv versions
   ```
   This should confirm that python 3.11 or above is installed. If it isn't already there,
   install it using `pyenv install`. You should be able to confirm
   you are using the correct version using
   ```sh
   $ python -V
   ```
  
* **Poetry**

  This package uses [Poetry](https://python-poetry.org/) to manage packages. See [python-poetry.org](https://python-poetry.org/docs/#installing-with-the-official-installer) for installation instructions.

### Installation

1. Clone this repository:
   ```sh
   $ git clone https://github.com/kymata-atlas/kymata-core.git
   ```
3. To install the python packages you will need to use Poetry. Assuming you have installed [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer), 
   type:
   ```sh
   $ poetry install
   ```
   to load the pakages needed.

4. At this point, you should be able to either run the xx from the terminal
   ```sh
   $ poetry run invokers/run_gridsearch.py
   ```
   or activate in this environment in an IDE such as PyCharm.

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

## Analysing a dataset with Kymata

### 1. Locate your raw EMEG dataset

You'll need the following files:

- `<participant_name>_run1_raw.fif`
- `<participant_name>_recording_config.yaml`

### 2. Preprocess the data

_Kymata Core_ holds the Kymata preprocessing code that comprises the 'Kymata back-end', including preprocessing steps, gridsearch procedures, expression plotting and IPPM generation.

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

Run the following invoker from invokers/:

```
invokers/run_gridsearch.py
```

This will output a `.nkg` file, which can then be loaded (see `demos/demo_save_load.ipynb`).

!!! notes 

    If running at the CBU, an easier way to do this (see [Troubleshooting](#CBU-troubleshooting)) may be to use the shell script `submit_gridsearch.sh`, which sets up the Apptainer environment the right way.
    Either run it locally with `./submit_gridsearch.sh`, or run it on the CBU queue with `sbatch submit_gridsearch.sh`.

### 4. Plot the results

- `invoker_run_nkg_plotting.py`

See also `demos/demo_plotting.ipynb`.

### 5. Visualise processing pathways

See `demos/demo_ippm.ipynb`.

[](){ #CBU-troubleshooting }
## Troubleshooting on the CBU compute cluster

- You see `Acccess denied permission error: 403` when you try to use github.

This is because your git instance at the CBU is not passing the correct authorisation credentials to your GitHub account. You will [have to create a new public key in ~/.ssh/ in your cbu home folder](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent), and then use this to [create an SSH key in your github settings](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Then, create (or modify) the `config` file in `~/.ssh/`:

```shell
Host github.com
        LogLevel DEBUG3
        User git
        Hostname github.com
        PreferredAuthentications publickey
        IdentityFile /home/<username>/.ssh/<name of private key>
```

- You see `ModuleNotFoundError: No module named 'numpy'`

You are probably running `submit_gridsearch.sh`, and it currently has Andy's `kymata-core` location hard-coded.
Update to point it at your copy.


- You want to print offscreen.

If plotting on the CBU cluster (off-screen), use

  ```shell
  xvfb-run -a poetry run python -m kymata.invokers.invoker_run_nkg_plotting
  ```
