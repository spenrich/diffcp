"""TODO: docstring."""
# flake8: noqa

__version__ = "1.0.13"

from diffcp.cone_program import solve_and_derivative, \
                                solve_and_derivative_batch, \
                                solve_and_derivative_internal, SolverError
from diffcp.cones import ZERO, POS, SOC, PSD, EXP
from diffcp import utils
import _diffcp
