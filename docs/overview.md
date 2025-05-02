# Overview of Kymata Core

_Kymata Core_ is the central codebase used for the generation and maintenance of the Kymata Atlas, [kymata.org](https://kymata.org). It
is released as open source in order to maintain openness and transparency in scientific research, and is maintained by
the [Kymata Research Group](https://kymata.org).

It is comprised of both a python package, `kymata`, and a number of invokers that instantiate the main Kymata Atlas pipeline, including:

* Standard preprocessing and source localisation steps for neural sensor data (MEG, EEG, ECoG);
* Gridsearch approaches for transform mapping;
* Information Processing Pathway Map generation (both offline generation and evaluation);
* Plotting functionality

The codebase is released publicly to ensure the transparency of the results in the Atlas. While comments and issues are
welcomed, we are unable to prioritise installation support.

[ :simple-git: Go to the Github repository](https://github.com/kymata-atlas/kymata-core){ .md-button .md-button--primary }
[Go to code documentation](https://github.com/kymata-atlas/kymata-core){ .md-button }

## Citing the codebase or the `kymata` package

Please use the following reference in all citations: 

> T.B.C

## Usage

Please refer to the [documentation](https://kymata-atlas.github.io/kymata-core), or see the `demos/` folder for example code, including test
data.