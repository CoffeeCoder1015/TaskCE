from __future__ import annotations

from attr import dataclass


# For clarity sake: every formula element is a Node with recursive flattening.
class Node:
    def flatten(self):
        raise NotImplementedError


@dataclass(frozen=True)
class Leaf(Node):
    label: str
    token_id: int
    token: str

    def flatten(self):
        return f"{self.label}:{self.token}"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def flatten(self):
        return f"(NOT {self.child.flatten()})"


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} AND {self.right.flatten()})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} OR {self.right.flatten()})"
