from dataclasses import dataclass

from sympy import Symbol

import numpy as np
import sympy
import torch
from .formula import logic_str
from .kernels import batched_iou
from .packing import pack_vectors
from sympy.logic import And, Or, Not

@dataclass(frozen=True)
class SearchResult:
    activation_index: int
    best_formula: str
    best_score: float


def calculate_ious(neuron, vectors, batch_size):
    output = []
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        iou = batched_iou(neuron, batch)
        output.append(iou)
    return torch.cat(output).detach()



@dataclass
class SearchConfig:
    maximum_formula_length: int = 5
    beam_size: int = 10
    max_iterations: int = 10
    iou_calculation_batch_size: int = 16384

def LevelSearch(neurons: torch.Tensor,concept_atoms: list[sympy.Expr], concept_vectors: torch.Tensor, config=SearchConfig()):
    """Search packed [word, neuron] activations using [concept, word] masks."""
    results = []
    for activation_index in range(neurons.shape[1]):
        neuron = neurons[:, activation_index].contiguous()
        print("Neuron shape:",neuron.shape)
        print("Feature shape:",concept_vectors[0].shape)

        ious = calculate_ious(neuron, concept_vectors, config.iou_calculation_batch_size)

        nonzero_indices = torch.argsort(ious, descending=True, stable=True)
        nonzero_indices = nonzero_indices[ious[nonzero_indices] > 0]
        nonzero_vectors = concept_vectors[nonzero_indices]
        # Each CPU formula corresponds to the same row in the GPU tensor.
        nonzero_formulas = [concept_atoms[index] for index in nonzero_indices.cpu().tolist()]
        print("Pre/Post zero filtering:", len(concept_atoms), len(nonzero_formulas))

        beam_size = config.beam_size
        beam_vectors = nonzero_vectors[:beam_size].clone()
        beam_formulas = nonzero_formulas[:beam_size]
        beam_ious = ious[nonzero_indices[:beam_size]]

        best_formula = ""
        best_score = 0

        iterations = 0
        while beam_formulas and iterations < config.max_iterations:
            formula_lengths = [formula.count(Symbol) for formula in beam_formulas]
            print(
                f"neuron={activation_index} iteration={iterations + 1} "
                f"beam={len(beam_formulas)} "
                f"formula_length={min(formula_lengths)}..{max(formula_lengths)}",
                flush=True,
            )
            # The beam is score ordered, so its first row is the current best.
            current_score = beam_ious[0].item()
            if current_score > best_score:
                best_formula = logic_str(beam_formulas[0])
                best_score = current_score

            parent_indices = [
                index for index, length in enumerate(formula_lengths)
                if length < config.maximum_formula_length
            ]
            generated_count = 0
            if parent_indices:
                parent_vectors = beam_vectors[parent_indices].unsqueeze(1)
                expansion_concepts = nonzero_vectors.unsqueeze(0)
                # [parent, concept, operation, word]; operations are AND, OR, AND NOT.
                compositions = torch.stack(
                    (
                        parent_vectors & expansion_concepts,
                        parent_vectors | expansion_concepts,
                        parent_vectors & ~expansion_concepts,
                    ),
                    dim=2,
                )
                neighbor_vectors = compositions.flatten(0, 2)
                generated_count = neighbor_vectors.shape[0]
                unique_vectors, inverse_indices = torch.unique(
                    neighbor_vectors, dim=0, return_inverse=True
                )
                # For each unique mask, keep the first original composition row.
                original_row_indices = torch.arange(generated_count, device=neuron.device)

                # magic trick from codex to not manually make a reverse mapping array
                unique_to_original = torch.full(
                    (unique_vectors.shape[0],), generated_count,
                    dtype=torch.long, device=neuron.device,
                )
                unique_to_original.scatter_reduce_(
                    0, inverse_indices, original_row_indices, reduce="amin"
                )

                scored_ious = calculate_ious(neuron, unique_vectors, config.iou_calculation_batch_size)
                # Start in generation order so stable score sorting preserves the old tie rule.
                generation_order = torch.argsort(unique_to_original)
                score_order = torch.argsort(scored_ious[generation_order], descending=True, stable=True)
                ranked_unique_indices = generation_order[score_order]
                selected_unique_indices = ranked_unique_indices[:beam_size]
                selected_original_indices = unique_to_original[selected_unique_indices]
                (
                    selected_parent_rows,
                    selected_concept_indices,
                    selected_operation_indices,
                ) = torch.unravel_index(selected_original_indices, compositions.shape[:3])
                # Parent coordinates index the eligible subset; recover source beam rows.
                source_parent_indices = torch.tensor(parent_indices, device=neuron.device)
                selected_parent_indices = source_parent_indices[selected_parent_rows]

                next_beam_vectors = unique_vectors[selected_unique_indices]
                next_beam_ious = scored_ious[selected_unique_indices]
                # Transfer only the selected [parent, concept, operation] records to the CPU.
                selected_compositions = torch.stack(
                    (selected_parent_indices, selected_concept_indices, selected_operation_indices),
                    dim=1,
                ).cpu().tolist()
                next_beam_formulas = []
                for parent, concept, operation in selected_compositions:
                    parent_formula = beam_formulas[parent]
                    concept_formula = nonzero_formulas[concept]
                    if operation == 0:
                        formula = And(parent_formula, concept_formula)
                    elif operation == 1:
                        formula = Or(parent_formula, concept_formula)
                    else:
                        formula = And(parent_formula, Not(concept_formula))
                    next_beam_formulas.append(formula)

                # All source formulas have been read; replace the aligned beam together.
                beam_vectors = next_beam_vectors
                beam_ious = next_beam_ious
                beam_formulas = next_beam_formulas
            else:
                beam_vectors = beam_vectors[:0]
                beam_ious = beam_ious[:0]
                beam_formulas = []
            iterations += 1
            print(
                f"neuron={activation_index} iteration={iterations} "
                f"generated={generated_count} retained={len(beam_formulas)} "
                f"best_score={best_score:.6f} best_formula={best_formula}",
                flush=True,
            )

        results.append((best_formula, best_score))
        return results  # Temporary stopper while checking the first neuron's search.


    return results

def search_all(
    activation_vectors,
    feature_vectors,
    device=None,
    config=None,
):
    config = config or SearchConfig()

    if device is not None:
        device =  torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Pack on the CPU, then keep one contiguous word vector per neuron column.
    activation_vectors = pack_vectors(np.asarray(activation_vectors).T, device).T
    
    atoms = []
    binary_vectors = []
    # decouple
    for formula, binary_vector in feature_vectors:
        atoms.append(formula)
        binary_vectors.append(binary_vector)

    # Stack and pack on the CPU so concepts reach the GPU in one transfer.
    concept_vectors = pack_vectors(np.stack(binary_vectors), device)

    pairs = LevelSearch(activation_vectors,atoms,concept_vectors , config)
    return [
        SearchResult(activation_index, formula, score)
        for activation_index, (formula, score) in enumerate(pairs)
    ]
