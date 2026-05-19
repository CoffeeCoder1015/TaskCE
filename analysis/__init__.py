"""Post-search and post-data analysis workspace.

This package is reserved for analysis that happens after capture and feature
search have produced their outputs. Planned responsibilities include the
ablation pipeline for CompExp results, aggregate reporting, follow-up research
angles, and other experiment interpretation utilities that should stay separate
from data capture and feature construction.
"""

from analysis.saving import (
    build_neuron_search_results_dataframe,
    save_neuron_search_results_csv,
)

__all__ = [
    "build_neuron_search_results_dataframe",
    "save_neuron_search_results_csv",
]
