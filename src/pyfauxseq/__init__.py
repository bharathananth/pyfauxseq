"""Top-level package for Python Implementation of the R Package fauxseq."""

__author__ = """Bharath Ananthasubramaniam"""
__email__ = 'bharath.ananthasubramaniam@hu-berlin.de'

from .generate_rnaseq import generate_rhy_rnaseq, generate_diffrhy_rnaseq

__all__ = ["generate_rhy_rnaseq", "generate_diffrhy_rnaseq"]
