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
from analysis.ablation import (
    AblationRunConfig,
    AblationRunResult,
    plot_ablation_results,
    run_ablation,
)
from analysis.ablation_analysis import (
    AblationAnalysisConfig,
    AblationAnalysisResult,
    run_ablation_analysis,
)
from analysis.activation_diagnostics import (
    save_activation_diagnostics,
    save_binary_activation_count_diagnostics,
    save_raw_activation_alpha_diagnostics,
)

__all__ = [
    "AblationAnalysisConfig",
    "AblationAnalysisResult",
    "AblationRunConfig",
    "AblationRunResult",
    "build_neuron_search_results_dataframe",
    "plot_ablation_results",
    "run_ablation",
    "run_ablation_analysis",
    "save_neuron_search_results_csv",
    "save_activation_diagnostics",
    "save_binary_activation_count_diagnostics",
    "save_raw_activation_alpha_diagnostics",
]
