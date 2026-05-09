from __future__ import annotations

from attr import dataclass
from sympy import false, Symbol, simplify, true
from sympy.logic.boolalg import And as SympyAnd
from sympy.logic.boolalg import BooleanFalse, BooleanTrue
from sympy.logic.boolalg import Not as SympyNot
from sympy.logic.boolalg import Or as SympyOr


def to_tree_structure(sympy_expression,metadat):
    # Turn the sympy expression back into the custom tree
    if isinstance(sympy_expression, Symbol):
        symbol = str(sympy_expression)
        try:
            label, token_id, token = metadat[symbol]
        except KeyError as error:
            raise ValueError(f"Missing metadata for symbol {symbol!r}") from error
        return Leaf(label=label, token_id=token_id, token=token)

    if sympy_expression is True or isinstance(sympy_expression, BooleanTrue):
        return Constant(value=True)
    if sympy_expression is False or isinstance(sympy_expression, BooleanFalse):
        return Constant(value=False)

    if sympy_expression.func is SympyNot:
        return Not(child=to_tree_structure(sympy_expression.args[0], metadat))

    if sympy_expression.func in (SympyAnd, SympyOr):
        node_type = And if sympy_expression.func is SympyAnd else Or
        children = [
            to_tree_structure(child, metadat)
            for child in sympy_expression.args
        ]
        if not children:
            return Constant(value=sympy_expression.func is SympyAnd)

        tree = children[0]
        for child in children[1:]:
            tree = node_type(left=tree, right=child)
        return tree

    raise TypeError(
        f"Unsupported SymPy expression {sympy_expression!r} "
        f"of type {type(sympy_expression).__name__}"
    )


# For clarity sake: every formula element is a Node with recursive flattening.
class Node:
    def flatten(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def simplify_tree(self):
        return simplify_tree(self)

    def flatten_sympy(self):
        raise NotImplementedError


def simplify_tree(node):
    expression, metadata = node.flatten_sympy()
    return to_tree_structure(simplify(expression), metadata) 


@dataclass(frozen=True)
class Constant(Node):
    value: bool

    def flatten(self):
        return "TRUE" if self.value else "FALSE"

    def __len__(self):
        return 1

    def simplify_tree(self):
        return simplify_tree(self)

    def flatten_sympy(self):
        return (true if self.value else false), {}


@dataclass(frozen=True)
class Leaf(Node):
    label: str
    token_id: int
    token: str

    def flatten(self):
        return f"{self.label}:{self.token}"

    def __len__(self):
        return 1
    
    def flatten_sympy(self):
        str_rep = f"{self.label}:{self.token}"
        metadata = {str_rep:[self.label,self.token_id,self.token]}
        return ( Symbol(str_rep),metadata )



@dataclass(frozen=True)
class Not(Node):
    child: Node

    def flatten(self):
        return f"(NOT {self.child.flatten()})"

    def __len__(self):
        return 1 + len(self.child)
    
    def flatten_sympy(self):
        formula, metadata = self.child.flatten_sympy()
        return (SympyNot(formula, evaluate=False),metadata)


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} AND {self.right.flatten()})"

    def __len__(self):
        return len(self.left) + len(self.right)
    
    def flatten_sympy(self):
        leftformula,leftmetadata = self.left.flatten_sympy()
        rightformula,rightmetadata = self.right.flatten_sympy()
        return (
            SympyAnd(leftformula, rightformula),
            leftmetadata | rightmetadata,
        )


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def flatten(self):
        return f"({self.left.flatten()} OR {self.right.flatten()})"

    def __len__(self):
        return len(self.left) + len(self.right)

    def flatten_sympy(self):
        leftformula,leftmetadata = self.left.flatten_sympy()
        rightformula,rightmetadata = self.right.flatten_sympy()
        return (
            SympyOr(leftformula, rightformula),
            leftmetadata | rightmetadata,
        )
