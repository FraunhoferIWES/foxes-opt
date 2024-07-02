"""
Wind farm optimization.
"""

from .core import FarmOptProblem, FarmObjective, FarmConstraint # noqa: F401

from . import problems # noqa: F401
from . import constraints # noqa: F401
from . import objectives # noqa: F401

try:
    from importlib.resources import files

    __version__ = files(__package__).joinpath("VERSION").read_text()
except ImportError:
    from importlib.resources import read_text

    __version__ = read_text(__package__, "VERSION")