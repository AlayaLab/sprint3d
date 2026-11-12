"""fastpixal3d: one image to a textured GLB in a few seconds, built on Pixal3D.

The heavy modules (torch, Pixal3D) are imported lazily; ``import fastpixal3d``
only loads the configuration and the lever table.
"""
from .config import EXPORT, Config, ExportSettings, Levers, Sampling
from .levers import LEVERS, lever

__version__ = "0.1.0"

__all__ = ["Config", "Levers", "Sampling", "EXPORT", "ExportSettings", "LEVERS", "lever", "FastPixal3D", "__version__"]


def __getattr__(name):
    if name == "FastPixal3D":
        from .pipeline import FastPixal3D

        return FastPixal3D
    raise AttributeError(name)
