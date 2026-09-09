"""
Wind farm optimization.
"""

import importlib
from pathlib import Path

from . import callbacks as callbacks
from . import constraints as constraints
from . import input as input
from . import objectives as objectives
from . import output as output
from . import problems as problems
from .core import FarmConstraint as FarmConstraint
from .core import FarmObjective as FarmObjective
from .core import FarmOptProblem as FarmOptProblem

try:
    tomllib = importlib.import_module("tomllib")
    source_location = Path(__file__).parent
    if (source_location.parent / "pyproject.toml").exists():
        with open(source_location.parent / "pyproject.toml", "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    else:
        __version__ = importlib.metadata.version(__package__ or __name__)
except ModuleNotFoundError:
    __version__ = importlib.metadata.version(__package__ or __name__)
