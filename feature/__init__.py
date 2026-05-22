"""Compositional explanation tools for activation patterns.

The feature package builds interpretable binary feature vectors from tokenized
datasets, represents feature formulas, and searches for logical compositions
that explain activation patterns. Its scope is compositional explanations: it
connects captured model activations to candidate feature formulas for later
experiment-level analysis.
"""
from feature.batch import batched
from feature.find import (
    SKIP_TOKENS,
    count_model_token_ids,
    filter_token_ids,
    select_feature_token_ids,
    token_should_be_skipped,
    top_token_counts,
)
from feature.formula import And, Leaf, Not, Or

__all__ = [
    "And",
    "Leaf",
    "Not",
    "Or",
    "SKIP_TOKENS",
    "batched",
    "count_model_token_ids",
    "filter_token_ids",
    "select_feature_token_ids",
    "token_should_be_skipped",
    "top_token_counts",
]
