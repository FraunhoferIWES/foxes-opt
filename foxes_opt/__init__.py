"""
Wind farm optimization.
"""

from .core import FarmOptProblem, FarmObjective, FarmConstraint  # noqa: F401

from . import problems  # noqa: F401
from . import constraints  # noqa: F401
from . import objectives  # noqa: F401

from importlib.metadata import version

__version__ = version(__package__ or __name__)
