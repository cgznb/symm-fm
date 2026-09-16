"""Conditional 3D SymmFlow for longitudinal breast MRI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ispy2-symmflow3d")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
