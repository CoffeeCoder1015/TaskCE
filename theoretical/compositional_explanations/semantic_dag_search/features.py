"""Atomic feature preparation for semantic-DAG search."""

from dataclasses import dataclass
import sys

import numpy as np
import sympy
from sympy import Symbol
from sympy.logic.boolalg import And as SymAnd
from sympy.logic.boolalg import Not as SymNot
from sympy.logic.boolalg import Or as SymOr
import torch


@dataclass(frozen=True)
class PreparedFeatures:
    formulas: tuple[sympy.Basic, ...]
    vectors: torch.Tensor
    sparse_vectors: torch.Tensor
    sizes: torch.Tensor
    semantic_keys: frozenset[bytes]

    @property
    def example_count(self) -> int:
        return self.vectors.shape[1]


def render_formula(formula: sympy.Basic) -> str:
    """Render a SymPy Boolean expression in the analysis CSV syntax."""
    if isinstance(formula, Symbol):
        return str(formula)
    if formula.func is SymNot:
        return f"(NOT {render_formula(formula.args[0])})"
    if formula.func is SymAnd:
        return f"({' AND '.join(render_formula(arg) for arg in formula.args)})"
    if formula.func is SymOr:
        return f"({' OR '.join(render_formula(arg) for arg in formula.args)})"
    return str(formula)


def semantic_key(vector: torch.Tensor) -> bytes:
    """Pack one Boolean trajectory into a stable CPU key."""
    cpu_vector = vector.detach().to(
        device="cpu",
        dtype=torch.bool,
    ).contiguous()
    return np.packbits(
        cpu_vector.numpy(),
        bitorder="little",
    ).tobytes()


def prepare_initial_features(
    feature_vectors: list[tuple[sympy.Basic, object]],
) -> list[tuple[sympy.Basic, torch.Tensor]]:
    """Validate atomic features and keep one formula per semantic trajectory."""
    if not feature_vectors:
        raise ValueError("Semantic-DAG search requires at least one feature.")

    prepared_features = []
    vectors_by_formula = {}
    formulas_by_trajectory = {}
    example_count = None

    for formula, vector in feature_vectors:
        cpu_vector = torch.as_tensor(
            vector,
            dtype=torch.bool,
            device="cpu",
        ).contiguous()
        if cpu_vector.ndim != 1:
            raise ValueError("Feature vectors must be one-dimensional.")
        if example_count is None:
            example_count = cpu_vector.shape[0]
        elif cpu_vector.shape[0] != example_count:
            raise ValueError(
                "All feature vectors must contain the same number of examples."
            )

        if formula in vectors_by_formula:
            if not torch.equal(vectors_by_formula[formula], cpu_vector):
                raise ValueError(
                    f"Atomic formula {formula!s} maps to multiple vectors."
                )
            continue
        vectors_by_formula[formula] = cpu_vector

        trajectory = semantic_key(cpu_vector)
        matching_formulas = formulas_by_trajectory.setdefault(
            trajectory,
            [],
        )
        matching_formulas.append(formula)
        if len(matching_formulas) == 1:
            prepared_features.append((formula, cpu_vector))

    for matching_formulas in formulas_by_trajectory.values():
        if len(matching_formulas) < 2:
            continue
        formulas = " == ".join(
            render_formula(formula)
            for formula in matching_formulas
        )
        print(
            "\033[31m"
            f"WARNING: identical initial feature trajectories: {formulas}"
            "\033[0m",
            file=sys.stderr,
        )

    return prepared_features


@torch.inference_mode()
def prepare_feature_matrix(
    feature_vectors: list[tuple[sympy.Basic, object]],
    device: torch.device,
) -> PreparedFeatures:
    """Compile validated atoms for repeated matrix scoring."""
    if not feature_vectors:
        raise ValueError("Semantic-DAG search requires at least one feature.")

    formulas, vectors = zip(*feature_vectors, strict=True)
    dense_cpu = torch.stack(
        [
            torch.as_tensor(
                vector,
                dtype=torch.bool,
                device="cpu",
            ).contiguous()
            for vector in vectors
        ]
    )
    sparse_cpu = dense_cpu.to_sparse_csr().to(dtype=torch.float32)
    return PreparedFeatures(
        formulas=tuple(formulas),
        vectors=dense_cpu.to(device=device),
        sparse_vectors=sparse_cpu.to(device=device),
        sizes=dense_cpu.sum(dim=1).to(
            device=device,
            dtype=torch.float32,
        ),
        semantic_keys=frozenset(
            semantic_key(vector)
            for vector in dense_cpu
        ),
    )
