"""Beam search over Boolean meanings, with ancestry kept separately from vectors."""

from dataclasses import dataclass

import numpy as np
from sympy import Symbol
from sympy.logic.boolalg import And, Not, Or
import torch

from .kernels import pack_vectors, score_atoms, score_compositions

# The final score axis, flat candidate identity, and ancestry share this order.
AND = 0
OR = 1
AND_NOT = 2
OPERATION_COUNT = 3
