[![Unit Tests Passed](https://github.com/kymata-atlas/kymata-core/actions/workflows/perform-unit-tests.yml/badge.svg)](https://github.com/kymata-atlas/kymata-core/actions/workflows/perform-unit-tests.yml)
[![Linting Checks Passed](https://github.com/kymata-atlas/kymata-core/actions/workflows/lint-and-check-formatting.yml/badge.svg)](https://github.com/kymata-atlas/kymata-core/actions/workflows/lint-and-check-formatting.yml)
<br />
<br />
<div>
      <a href="https://github.com/kymata-atlas/kymata-core">
        <img src="https://kymata.org/assets/img/toolbox_logo.png" alt="Logo" height="112" align="left">
      </a>
    <div>
        <div>
            <h3>Kymata Core</h3>
            <p>Core codebase for the Kymata Atlas
            <br>
            <a href="https://kymata-atlas.github.io/kymata-core"><strong>Explore the docs »</strong></a>
            </p>
        </div>
    </div>

</div>
<br>
<p align="center">
        <a href="#About The Project">Overview</a>
        ·
        <a href="#Getting Started">Setup</a>
        ·
        <a href="#Citing the Codebase">Citing</a>
        ·
        <a href="#Licence">Licence</a>
</p>

## About The Project

_Kymata Core_ is the central codebase underlying the [Kymata Atlas](https://kymata.org).[^1] It is maintained by the
[Kymata Research Group](https://kymata.org).

The central pipeline includes:
* Standard preprocessing and source localisation steps for neural sensor data (MEG, EEG, ECoG);
* Gridsearch approaches for function mapping;
* Information Processing Pathway Map[^2] generation (both offline generation and evaluation);
* Plotting functionality

The codebase is released under an MIT license to ensure the transparency of the results in the Kymata Atlas. While
comments and issues are welcomed, we are unable to prioritise installation or usage support (please see our code of
conduct).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

This provides an overview of how to set Kymata Core locally.

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Please refer to the [documentation](https://kymata-atlas.github.io/kymata-core), or see the `demos/` folder for example code, including test
data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citing the codebase

Please use the following reference when citing the codebase or the `kymata` package: 

> TBC

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## Licence

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REFERENCES -->
## References

[^1]: Thwaites, Wieser, Soltan, Wingfield, Parish, Yang, Lakra, Woolgar, Zhang (2024) _The Kymata Atlas_ (In Prep.)
[^2]: Thwaites, Zhang and Woolgar (2024) _Information Processing Pathway Maps: A New Frontier in Cortical Processing_ (In Prep.)

