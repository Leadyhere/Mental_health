"""Single indirection point for the tensor backend.

Verified directly on this machine that PyTorch 2.12 (CPU wheel) imports and
runs fine on the installed Python 3.14 -- no fallback is needed today. This
module exists so that if a future environment ever can't resolve a torch
wheel for its Python version, only this file needs to change (e.g. to a
numpy-only GRU implementation) rather than every module in the package.
"""
from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE
