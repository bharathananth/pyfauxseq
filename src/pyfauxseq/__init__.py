"""Top-level package for Python Implementation of the R Package fauxseq."""

from importlib.metadata import version

from .generate_rnaseq import generate_diffrhythmic_rnaseq, generate_rhythmic_rnaseq

__version__ = version("pyfauxseq")

__all__ = [
    "generate_diffrhythmic_rnaseq",
    "generate_rhythmic_rnaseq",
    "normalize_counts",
]
