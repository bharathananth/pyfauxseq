---
hide:
  - navigation
---

# Getting Started

## Installation

### Stable release

To install Python Implementation of the R package fauxseq, run this command in your terminal:

```sh
uv add pyfauxseq
```

Or if you prefer to use `pip`:

```sh
pip install pyfauxseq
```

### From source

The source files for Python Implementation of the R package fauxseq can be downloaded from the [Github repo](https://github.com/bharathananth/pyfauxseq).

You can either clone the public repository:

```sh
git clone git://github.com/bharathananth/pyfauxseq
```

## Generate synthetic rhythmic RNA-seq data
```python
import pyfauxseq as pf
```
For timeseries data under one condition:
```python
sim_data = pf.generate_rhythmic_rnaseq()
```
For timeseries data under two condition:
```python
sim_data = pf.generate_diffrhythmic_rnaseq()
```
The parameters to customize the data are listed in [API Documentation](api_documentation.md)
