from __future__ import annotations

from attr import dataclass
from sympy import  Symbol
from sympy.logic.boolalg import And, Or, Not

def logic_str(expr):
    if isinstance(expr, Symbol):
        return f"{expr}"

    if expr.func is Not:
        return f"(NOT {logic_str(expr.args[0])})"

    if expr.func is And:
        inner = " AND ".join(f"{logic_str(arg)}" for arg in expr.args)
        return f"({inner})"

    if expr.func is Or:
        inner = " OR ".join(f"{logic_str(arg)}" for arg in expr.args)
        return f"({inner})"

    return f"{expr}"


# For clarity sake: every formula element is a Node with recursive flattening.
class Node:
    def flatten(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def flatten_sympy(self):
        raise NotImplementedError


@dataclass(frozen=True)
class Leaf(Node):
    label: str
    token_id: int
    token: str

    def flatten(self):
        return f"{self.label}:{self.token}"

    def __len__(self):
        return 1
    
@dataclass(frozen=True)
class Not(Node):
    child: Node

    def flatten(self):
        return f"(NOT {self.child.flatten()})"

    def __len__(self):
        return 1 + len(self.child)
    
@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} AND {self.right.flatten()})"

    def __len__(self):
        return len(self.left) + len(self.right)
    
@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} OR {self.right.flatten()})"

    def __len__(self):
        return len(self.left) + len(self.right)
