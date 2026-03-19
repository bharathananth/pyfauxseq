# pyfauxseq
### A python Implementation of the R package fauxseq

![GitHub License](https://img.shields.io/github/license/:user/:repo)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/:user/:repo/total)
![PyPI - Downloads](https://img.shields.io/pypi/:period/:packageName)
![PyPI - Version](https://img.shields.io/pypi/v/:packageName)


## What pyfauxseq does

This package can generate synthetic timeseries RNA-seq data with a fraction of rhythmic genes and empirical relationship between mean expression and variability across replicates of genes.

`pyfauxseq` improves upon previous tools as follows:
- generates negative-binomial count data with empirically-estimated mean-dispersion properties.
- generates data with either differential expression or differential rhythmicity or both.

## How to install pyfauxseq
```python -m pip install pyfauxseq```

## Get started with pyfauxseq
Get started with synthetic RNA-seq data with the ground truth with the default parameters using
```python
import pyfauxseq as pf
sim_data = pf.generate_rhythmic_rnaseq()
sim_data["counts"] # the count data
sim_data["params"] # parameters of the rhythmic genes
sim_data["exp_design"] # time labels of the individual sample (columns) of count data
```
## How to cite pyfauxseq
Please cite this software using CITATION.CFF or the "Cite this repository" link in the right sidebar.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
