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

## License

`lambda-commissioning` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
