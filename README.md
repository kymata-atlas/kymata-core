[![Unit Tests Passed](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/perform-unit-tests.yml/badge.svg)](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/perform-unit-tests.yml)
[![Linting Checks Passed](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/lint-and-check-formatting.yml/badge.svg)](https://github.com/kymata-atlas/kymata-toolbox/actions/workflows/lint-and-check-formatting.yml)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="docs/assets/images/toolbox_logo.png" alt="Logo" width="200" height="112">
  </a>

  <h3 align="center">The Kymata Toolbox</h3>

  <p align="center">
    Core functionality for the Kymata Atlas
    <br />
    <a href="#xxx"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#About The Project">Overview</a>
    ·
    <a href="#Getting Started">Setup</a>
    ·
    <a href="#Citing the Toolbox">Citing</a>
    ·
    <a href="#Licence">Licence</a>

  </p>
</div>

## About The Project

The _Kymata Toolbox_ is the core codebase underlying the _Kymata Atlas_, a repository of human Information Processing
Pathway Maps ['IPPMs'] of the human brain[^1]. The codebase covers is used to generate these maps, including both
standard and experimental pipelines. It is maintained by the [Kymata Research Group](https://kymata.org).

The core pipeline includes:
* Preprocessing of electrophysiological data (MEG, EEG, ECoG)
* Standard and experimental Kymata gridsearch approaches;
* Information Processing Pathway Map generation (offline generation, and evaluation)
* Plotting functionality

The codebase is released under xxx to ensure the transparency of the results in the Kymata Atlas[^2]. While comments and
issues are xxx, we are unable to prioritise xxx or xxx (please see our code of conduct).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This provides an overview of how to set the Kymata Toolbox locally.

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

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone this repository:
   ```sh
   $ git clone https://github.com/your_username_/Project-Name.git
   ```
3. To install the python packages you will need to use Poetry. Assuming you have installed [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer), 
   type:
   ```sh
   $ poetry install
   ```
   to load the pakages needed.

4. At this point, you should be able to either run the xx from the terminal
   ```sh
   $ poetry run __init__.py
   ```
   or activate in this environment in an IDE such as PyCharm.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citing the Toolbox

Please use the following xxx in all citations: 

> Thwaites, Wingfield, Parish, Yang, Lakra, Zhang (2024) _The Kymata Toolbox_ XYZ

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## Licence

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## References

[^1]: Thwaites et al (2024) _Information Processing Pathway Maps_ XYZ
[^2]: Thwaites et al (2024) _The Kymata Atlas_ XYZ