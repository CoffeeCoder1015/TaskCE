import heapq
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np

from sympy import Symbol, simplify_logic

import sympy
import torch
# from feature.formula import And, Not, Or 
from feature.formula import logic_str
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


def batched_iou(neuron,binary_vecs):
    intersections = (binary_vecs & neuron).sum(dim=1)
    unions = (binary_vecs | neuron).sum(dim=1)
    raw_scores = intersections.float() / unions.clamp_min(1).float()
    return torch.where(unions > 0, raw_scores, torch.zeros_like(raw_scores))

def calculate_ious(neuron, vector_list, batch_size):
    ious = []
    for i in range(0, len(vector_list), batch_size):
        batch = vector_list[i:i+batch_size]
        vectors = torch.stack(batch)
        iou = batched_iou(neuron, vectors)
        ious.extend(iou.tolist())
    return ious


# Hash keys to not visit them again
def tensor_key(vector):
    vector = vector.detach().to(device="cpu", dtype=torch.bool).contiguous()
    return vector.numpy().tobytes()

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
class searchConfig:
    formula_length: int = 6
    pruned_queue_size: int = 30 # The beam size
    max_iterations: int = 50
    length_penalty: float = 0.01
    iou_calculation_batch_size: int = 16384

def LevelSearch(neuron: torch.Tensor,feature_vectors:list[tuple[sympy.Expr,torch.Tensor]],config=searchConfig()):
    print("Neuron shape:",neuron.shape)
    print("Feature shape:",feature_vectors[0][1].shape)

    states = [SearchState(formula, vector) for formula, vector in feature_vectors]
    vectors = [state.vector for state in states]
    ious = calculate_ious(neuron, vectors, config.iou_calculation_batch_size)
    
    nonzero_features = [(iou, state) for iou, state in zip(ious, states) if iou > 0]
    nonzero_features.sort(key=lambda x: x[0], reverse=True)
    print("Pre/Post zero filtering:", len(states), len(nonzero_features))

    beam_size = config.pruned_queue_size
    queue = []
    # Load queue
    queue_id = 0
    penalty = config.length_penalty
    score_track = {}
    for iou, state in nonzero_features:
        heapq.heappush(queue, (-iou, queue_id, state))
        score_track[tensor_key(state.vector)] = iou
        queue_id += 1
    if len(queue) > beam_size:
        queue = heapq.nsmallest(beam_size, queue)
        heapq.heapify(queue)
    
    best_formula = ""
    best_score = 0
    best_iou = 0

    iterations = 0
    while queue and iterations < config.max_iterations:
        neighbors = []
        for rank_heuristic, _, state in queue:
            formula = state.formula
            vector = state.vector
            current_iou = abs(rank_heuristic)
            current_score = current_iou * length_penalty_factor(formula, penalty)
            if current_score > best_score:
                print("score:", current_score, logic_str(formula))
                best_formula = logic_str(formula)
                best_score = current_score
            if current_iou > best_iou:
                print("iou:", current_iou, logic_str(formula))
                best_iou = current_iou

            length = formula.count(Symbol)
            if length >= config.formula_length:
                continue # Do not expand formula at max length

            for _, neighbor_state in nonzero_features:
                for n in get_compositions(formula, vector, neighbor_state.formula, neighbor_state.vector):
                    neighbors.append(n)
            
        new_queue = []
        if neighbors:
            neighbor_vectors = [n.vector for n in neighbors]
            scored_ious = calculate_ious(neuron, neighbor_vectors, config.iou_calculation_batch_size)
            for state, iou in zip(neighbors, scored_ious):
                key = tensor_key(state.vector)
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

    return best_formula, best_score

def Search(neuron_activation: torch.Tensor,feature_vectors:list[tuple[sympy.Expr,torch.Tensor]],config=searchConfig()):
    print("Neuron shape:",neuron_activation.shape)
    print("Feature shape:",feature_vectors[0][1].shape)

    states = [SearchState(formula, vector) for formula, vector in feature_vectors]
    vectors = [state.vector for state in states]
    ious = calculate_ious(neuron_activation, vectors, config.iou_calculation_batch_size)
    
    nonzero_features = [(iou, state) for iou, state in zip(ious, states) if iou > 0]
    nonzero_features.sort(key=lambda x: x[0], reverse=True)
    print("Pre/Post zero filtering:", len(states), len(nonzero_features))

    beam_size = config.pruned_queue_size
    queue = []
    # Load queue
    queue_id = 0
    penalty = config.length_penalty
    score_track = {}
    for iou, state in nonzero_features:
        heapq.heappush(queue, (-iou, queue_id, state))
        score_track[tensor_key(state.vector)] = iou
        queue_id += 1
    if len(queue) > beam_size:
        queue = heapq.nsmallest(beam_size, queue)
        heapq.heapify(queue)
    
    best_formula = ""
    best_score = 0
    best_iou = 0

    iterations = 0
    while queue and iterations < config.max_iterations:
        rank_heuristic, _, state = heapq.heappop(queue)
        formula = simplify_logic(state.formula)
        vector = state.vector

        current_iou = abs(rank_heuristic)
        current_score = current_iou * length_penalty_factor(formula, penalty)
        if current_score > best_score:
            print("score:", current_score, logic_str(formula))
            best_formula = logic_str(formula)
            best_score = current_score
        if current_iou > best_iou:
            print("iou:", current_iou, logic_str(formula))
            best_iou = current_iou

        length = formula.count(Symbol)
        if length >= config.formula_length:
            continue # Do not expand formula at max length

        neighbors = []
        for _, neighbor_state in nonzero_features:
            for n in get_compositions(formula, vector, neighbor_state.formula, neighbor_state.vector):
                neighbors.append(n)

        if neighbors:
            neighbor_vectors = [n.vector for n in neighbors]
            scored_ious = calculate_ious(neuron_activation, neighbor_vectors, config.iou_calculation_batch_size)
            for state, iou in zip(neighbors, scored_ious):
                key = tensor_key(state.vector)
                prior_score = score_track.get(key, 0)
                score = iou * length_penalty_factor(state.formula, penalty)
                if score > prior_score:
                    score_track[key] = score
                    heapq.heappush(queue, (-iou, queue_id, state))
                    queue_id += 1
        
        if len(queue) > beam_size:
            queue = heapq.nsmallest(beam_size, queue)
            heapq.heapify(queue)

        iterations+=1

    return best_formula, best_score

def to_numpy_bool(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.bool_, copy=False)
    return np.asarray(value, dtype=np.bool_)

def prepare_multiprocessing_feature_vectors(feature_vectors):
    return [
        (formula, to_numpy_bool(binary_vector))
        for formula, binary_vector in feature_vectors
    ]

def activation_index_chunks(total_activations, num_workers):
    if total_activations == 0:
        return []
    worker_count = min(max(1, num_workers), total_activations)
    chunk_size, remainder = divmod(total_activations, worker_count)
    chunks = []
    start = 0
    for worker_index in range(worker_count):
        stop = start + chunk_size + (1 if worker_index < remainder else 0)
        chunks.append((start, stop))
        start = stop
    return chunks

def search_worker(activation_vectors,activation_indicies:list[int],feature_vectors,device=None,config=searchConfig()):
    device = resolve_device(device)
    feature_vectors = prepare_feature_vectors(feature_vectors, device)
    activation_vectors = to_binary_tensor(activation_vectors, device)
    results = []
    
    for local_activation_index, global_activation_index in enumerate(activation_indicies):
        print(f"Running LevelSearch for global index {global_activation_index}")
        best_formula, best_score = LevelSearch(
            activation_vectors[:,local_activation_index],
            feature_vectors,
            config,
        )
        results.append(
            SearchResult(
                activation_index=global_activation_index,
                best_formula=best_formula,
                best_score=best_score,
            )
        )

    return results

def search_worker_from_args(args):
    return search_worker(*args)

def search_all(activation_vectors, feature_vectors, num_workers=1, device=None, config=searchConfig()):
    if num_workers <= 1:
        return search_worker(
            activation_vectors,
            list(range(activation_vectors.shape[1])),
            feature_vectors,
            device,
            config
        )

    activation_vectors = to_numpy_bool(activation_vectors)
    feature_vectors = prepare_multiprocessing_feature_vectors(feature_vectors)
    chunks = activation_index_chunks(activation_vectors.shape[1], num_workers)
    
    if not chunks:
        return []

    worker_args = [
        (
            activation_vectors[:, start:stop], # Pre-chunk so each subprocess doesn't get the entire copy of activation vectors
            list(range(start, stop)),
            feature_vectors,
            device,
            config,
        )
        for start, stop in chunks
    ]

    context = mp.get_context("spawn")
    with context.Pool(processes=len(worker_args)) as pool:
        result_chunks = pool.map(search_worker_from_args, worker_args)

    results = [
        result
        for chunk in result_chunks
        for result in chunk
    ]
    return sorted(results, key=lambda result: result.activation_index)
