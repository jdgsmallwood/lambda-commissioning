# lambda

[![PyPI - Version](https://img.shields.io/pypi/v/lambda.svg)](https://pypi.org/project/lambda)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lambda.svg)](https://pypi.org/project/lambda)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Development Instructions

1. Install hatch - I recommend using pipx (https://github.com/pypa/pipx) to install system-wide. 
    - This could also be done using a virtual environment or similar.
2. Checkout the git repo.
3. From the command line run ``hatch shell" to sync dependencies and install the package in editable mode.
4. Test setup by running 
```python
python
>> from lambda_commissioning.example import example_function
>> example_function()
```

## Writing new code
Ensure that all new code is placed within either the src/lambda_commissioning folder or the tests/ folder.

## Setting up paths

In the  ```default_config``` file are the ```dataPath``` and the ```outputsPath``` config options. This can be existing directories, or the defaults (which should be the user home directory plus ```lambda-commisioning/data/``` and ```lambda-commisioning/outputs/```). The paths can be changed by editing the default config before installing the package or after installation if in editable mode. 

## Command Line Tools

Command line tool ```run-diagnostics``` developed for reading in the ```hdf5``` data captures and outputting diagnostic plots, and statistics.

```
run-diagnostics --help
```

There are three apps associated with this tool:

```
run-diagnostics file-header --help
```
Outputs useful header information, such as the number of time samples, channels,
baselines and antennas. 

```
run-diagnostics autos --help
```
The autos app outputs the waterfall plots for the auto-correlations for each 
antenna. As well as calculating some useful stats. These plots get saved to the
autocorrelations folder, in the data capture directory ```${OUTPATH}/capture/autocorrelations/```.

```
run-diagnostics vis --help
```

The vis command performs the same plotting as the autos command but for a subset of visibilities, plotting the amplitudes, and phase waterfall plots, as well as the fringes for a single channel for the same baselines. These are saved in the following directories:

```${OUTPATH}/capture/vis_amp/```
```${OUTPATH}/capture/vis_phase/```
```${OUTPATH}/capture/vis_fringes/```

## License

`lambda-commissioning` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
