# kymata-hpc Readme

Maintained by the [Kymata Research Group](https://kymata.org) (UCL / University of Cambridge / Tsinghua).

## Description

The repository holds the Kymata preprocessing code that runs on the MRC-CBU HPC system. This code comprises most of the 'Kymata back-end', including preprocessing steps and statistical procedures. The output of this code is the input to kymata-web.

<img src="assets/overview_graphic.png" width="400" height="754">

## Setting up to run with Poetry

First, confirm you have the correct version of Python installed. Navigate to the root directory. Type
```
$ pyenv versions
```
This should confirm that python 3.8.9 or above is installed. If it isn't already there,
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

# Run tests

To run the tests, run:
```
$ poetry run pytest
```