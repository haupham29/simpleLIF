# snnax

## Description
Spiking Neural Networks implemented on top of [diffrax](https://github.com/patrick-kidger/diffrax).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone the repository:

```
git clone https://anonymous.4open.science/r/snnax-main
cd snnax
pip install .
```

Make sure to have jax installed. For CPU:

```
pip install "jax[cpu]"
```

For NVIDIA GPU:

```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Until the new event-handling is officially a part of diffrax, you will need to install a local version with the correct modifications. (For the purpose of anonymity at submission time, this repo is provided as a separate file in the supplementary material called diffrax.zip)

```
cd diffrax
pip install .
```

## Usage

For some usage examples see the [notebooks](./notebooks/).

## Contributing

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
