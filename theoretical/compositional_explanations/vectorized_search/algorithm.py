import heapq
from dataclasses import dataclass

from sympy import Symbol

import sympy
import torch
from .formula import logic_str
from sympy.logic import And, Or, Not

@dataclass
class SearchState:
    formula: sympy.Expr
    vector: torch.Tensor


@dataclass(frozen=True)
class SearchResult:
    activation_index: int
    best_formula: str
    best_score: float


# Helper functions for compute
def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_binary_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.bool, device=device)

def prepare_feature_vectors(feature_vectors, device):
    return [
        (formula, to_binary_tensor(binary_vector, device))
        for formula, binary_vector in feature_vectors
    ]


def batched_iou(neuron,binary_vecs,neuron_count=None):
    intersections = (binary_vecs & neuron).sum(dim=1)
    if neuron_count is None:
        neuron_count = neuron.sum()
    unions = binary_vecs.sum(dim=1) + neuron_count - intersections
    raw_scores = intersections.float() / unions.clamp_min(1).float()
    return torch.where(unions > 0, raw_scores, torch.zeros_like(raw_scores))

def calculate_ious(neuron, vector_list, batch_size):
    output = []
    neuron_count = neuron.sum()
    for i in range(0, len(vector_list), batch_size):
        batch = vector_list[i:i+batch_size]
        vectors = torch.stack(batch)
        iou = batched_iou(neuron, vectors, neuron_count)
        output.append(iou)
    return torch.cat(output).detach().cpu().tolist()



def length_penalty_factor(formula, penalty):
    length = formula.count(Symbol)
    return max(0.0, 1.0 - penalty * (length - 1))


def get_compositions(current_formula, current_vector, feature_formula, feature_vector):
    new_compositions = []

    # No checks for repetition
    # new_compositions.append(SearchState(And(current_formula, feature_formula), current_vector & feature_vector))
    # new_compositions.append(SearchState(Or(current_formula, feature_formula), current_vector | feature_vector))
    # new_compositions.append(SearchState(And(current_formula, Not(feature_formula)), current_vector & ~feature_vector))
    
    # Checks for repetition
    c_and_f = current_vector & feature_vector
    if torch.any(c_and_f).item() and not torch.equal(c_and_f, current_vector) and not torch.equal(c_and_f, feature_vector):
        new_compositions.append(SearchState(And(current_formula, feature_formula), c_and_f))

    c_or_f = current_vector | feature_vector
    if torch.any(feature_vector & ~current_vector).item() and not torch.equal(c_or_f, feature_vector):
        new_compositions.append(SearchState(Or(current_formula, feature_formula), c_or_f))

    c_and_nf = current_vector & ~feature_vector
    if current_formula != feature_formula and torch.any(c_and_nf).item() and not torch.equal(c_and_nf, current_vector):
        new_compositions.append(SearchState(And(current_formula, Not(feature_formula)), c_and_nf))

    return new_compositions

@dataclass
class SearchConfig:
    maximum_formula_length: int = 5
    beam_size: int = 10
    max_iterations: int = 10
    length_penalty: float = 0.0
    iou_calculation_batch_size: int = 16384

def LevelSearch(neurons: torch.Tensor, concepts: list[tuple[sympy.Expr, torch.Tensor]], config=SearchConfig()):
    """Serial mock: [examples, neurons] and shared [examples] concepts yield per-column pairs."""
    results = []
    for activation_index in range(neurons.shape[1]):
        neuron = neurons[:, activation_index].contiguous()
        print("Neuron shape:",neuron.shape)
        print("Feature shape:",concepts[0][1].shape)

        states = [SearchState(formula, vector) for formula, vector in concepts]
        vectors = [state.vector for state in states]
        ious = calculate_ious(neuron, vectors, config.iou_calculation_batch_size)

        nonzero_features = [(iou, state) for iou, state in zip(ious, states) if iou > 0]
        nonzero_features.sort(key=lambda x: x[0], reverse=True)
        print("Pre/Post zero filtering:", len(states), len(nonzero_features))

        beam_size = config.beam_size
        queue = []
        # Load queue
        queue_id = 0
        penalty = config.length_penalty
        score_track = {}
        for iou, state in nonzero_features:
            heapq.heappush(queue, (-iou, queue_id, state))
            score_track[state.formula] = iou
            queue_id += 1
        if len(queue) > beam_size:
            queue = heapq.nsmallest(beam_size, queue)
            heapq.heapify(queue)

        best_formula = ""
        best_score = 0
        best_iou = 0

        iterations = 0
        while queue and iterations < config.max_iterations:
            formula_lengths = [state.formula.count(Symbol) for _, _, state in queue]
            print(
                f"neuron={activation_index} iteration={iterations + 1} "
                f"beam={len(queue)} "
                f"formula_length={min(formula_lengths)}..{max(formula_lengths)}",
                flush=True,
            )
            neighbors = []
            for rank_heuristic, _, state in queue:
                formula = state.formula
                vector = state.vector
                current_iou = abs(rank_heuristic)
                current_score = current_iou * length_penalty_factor(formula, penalty)
                if current_score > best_score:
                    best_formula = logic_str(formula)
                    best_score = current_score
                if current_iou > best_iou:
                    best_iou = current_iou

                length = formula.count(Symbol)
                if length >= config.maximum_formula_length:
                    continue # Do not expand formula at max length

                for _, neighbor_state in nonzero_features:
                    for n in get_compositions(formula, vector, neighbor_state.formula, neighbor_state.vector):
                        neighbors.append(n)

            new_queue = []
            if neighbors:
                neighbor_vectors = [n.vector for n in neighbors]
                scored_ious = calculate_ious(neuron, neighbor_vectors, config.iou_calculation_batch_size)
                for state, iou in zip(neighbors, scored_ious):
                    key = state.formula
                    prior_score = score_track.get(key, 0)
                    score = iou * length_penalty_factor(state.formula, penalty)
                    if score > prior_score:
                        score_track[key] = score
                        heapq.heappush(new_queue, (-iou, queue_id, state))
                        queue_id += 1

            queue = new_queue
            if len(queue) > beam_size:
                queue = heapq.nsmallest(beam_size, queue)
                heapq.heapify(queue)
            iterations += 1
            print(
                f"neuron={activation_index} iteration={iterations} "
                f"generated={len(neighbors)} retained={len(queue)} "
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
    device = resolve_device(device)
    activation_vectors = to_binary_tensor(activation_vectors, device)
    feature_vectors = prepare_feature_vectors(feature_vectors, device)
    pairs = LevelSearch(activation_vectors, feature_vectors, config)
    return [
        SearchResult(activation_index, formula, score)
        for activation_index, (formula, score) in enumerate(pairs)
    ]
