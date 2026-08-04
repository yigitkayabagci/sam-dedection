"""Tests, and the stand-ins they need.

`reference_engines.py` lives here rather than under `tools/` because nothing
in the deployed pipeline imports it: it is a PyTorch stand-in for the TensorRT
engines, used to exercise the integration path without a GPU.
"""
