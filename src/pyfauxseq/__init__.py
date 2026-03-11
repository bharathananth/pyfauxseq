"""Top-level package for Python Implementation of the R Package fauxseq."""

__author__ = """Bharath Ananthasubramaniam"""
__email__ = "bharath.ananthasubramaniam@hu-berlin.de"

from .generate_rnaseq import generate_diffrhythmic_rnaseq, generate_rhythmic_rnaseq

__all__ = [
    "generate_diffrhythmic_rnaseq",
    "generate_rhythmic_rnaseq",
    "normalize_counts",
]
