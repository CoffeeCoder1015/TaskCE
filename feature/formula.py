from __future__ import annotations

from attr import dataclass

@dataclass(frozen=True)
class Leaf:
    label: str
    token_id: int
    token: str

    def flatten(self):
        return f"{self.label}:{self.token}"


@dataclass(frozen=True)
class Not:
    child: Node

    def flatten(self):
        return f"(NOT {self.child.flatten()})"


@dataclass(frozen=True)
class And:
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} AND {self.right.flatten()})"


@dataclass(frozen=True)
class Or:
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} OR {self.right.flatten()})"

# For clarity sake: every formula node is either a leaf or an operator.
type Node = Leaf | Not | And | Or
