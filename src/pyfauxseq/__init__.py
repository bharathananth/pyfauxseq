"""Top-level package for Python Implementation of the R Package fauxseq."""

__author__ = """Bharath Ananthasubramaniam"""
__email__ = 'bharath.ananthasubramaniam@hu-berlin.de'

from .generate_rnaseq import generate_rhythmic_rnaseq, generate_diffrhythmic_rnaseq

__all__ = ["generate_rhythmic_rnaseq", "generate_diffrhythmic_rnaseq"]
