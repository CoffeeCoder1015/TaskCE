import heapq

from matplotlib.pylab import logistic
from sympy import Symbol

import torch
# from feature.formula import And, Not, Or 
from feature.formula import logic_str
from sympy.logic import And, Or, Not

def batched_iou(neuron,binary_vecs):
    intersections = (binary_vecs & neuron).sum(dim=1)
    unions = (binary_vecs | neuron).sum(dim=1)
    raw_scores = intersections.float() / unions.clamp_min(1).float()
    return torch.where(unions > 0, raw_scores, torch.zeros_like(raw_scores))

def formula_iou(neuron,formulas_vectors,batch_size):
    candidates = []
    for i in range(0, len(formulas_vectors), batch_size):
        batch = formulas_vectors[i:i+batch_size]
        formulas = [formula for formula, _ in batch]
        binary_vectors =[bin_vec for _, bin_vec in batch] 
        vectors = torch.stack(binary_vectors)
        iou = batched_iou(neuron,vectors)
        candidates.extend(zip(iou.tolist(),formulas,binary_vectors))
    return candidates

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

# Hash keys to not visit them again
def tensor_key(vector):
    vector = vector.detach().to(device="cpu", dtype=torch.bool).contiguous()
    return vector.numpy().tobytes()

def length_penalty_factor(formula, penalty):
    length = formula.count(Symbol)
    return max(0.0, 1.0 - penalty * (length - 1))

def get_compositions(current_formula, current_vector, feature_formula, feature_vector):
    new_compositions = []

    if torch.any(current_vector & ~feature_vector).item():
        new_compositions.append(
            (
                And(current_formula, feature_formula),
                current_vector & feature_vector,
            )
        )
    if torch.any(feature_vector & ~current_vector).item():
        new_compositions.append(
            (
                Or(current_formula, feature_formula),
                current_vector | feature_vector,
            )
        )
    if (
        current_formula != feature_formula
        and torch.any(current_vector & feature_vector).item()
    ):
        new_compositions.append(
            (
                And(current_formula, Not(feature_formula)),
                current_vector & ~feature_vector,
            )
        )
    return new_compositions

def LevelSearch(neuron,feature_vectors):
    device = resolve_device()
    print("Neuron shape:",neuron.shape)
    print("Feature shape:",feature_vectors[0][1].shape)
    neuron = to_binary_tensor(neuron,device)
    scored_features = formula_iou(neuron,prepare_feature_vectors(feature_vectors,device),4096)
    nonzero_features = [feature for feature in scored_features if feature[0] > 0]
    nonzero_features.sort(key=lambda x: x[0],reverse=True)
    print("Pre/Post zero filtering:",len(scored_features),len(nonzero_features))
    
    beam_size = 30
    formula_length = 6
    queue = []
    # Load queue
    queue_id = 0
    penalty = 0.01
    score_track = {}
    for item in nonzero_features:
        heapq.heappush(queue,(-item[0],queue_id,item[1],item[2]))
        score_track[tensor_key(item[2])] = item[0]
        queue_id+=1
    if len(queue) > beam_size:
        queue = heapq.nsmallest(beam_size, queue)
        heapq.heapify(queue)
    
    best_score = 0
    best_iou = 0
    max_expansions = beam_size * max(0, formula_length - 1)
    expansions = 0
    while queue and expansions < max_expansions:
        # rank_heuristic, _, formula, vector = heapq.heappop(queue)

        neighbors = []
        for rank_heuristic, _, formula, vector in queue:
            current_iou = abs(rank_heuristic)
            current_score = current_iou * length_penalty_factor(formula,penalty)
            if current_score > best_score:
                print("score:",current_score,logic_str(formula))
                best_score = current_score
            if current_iou > best_iou:
                print("iou:",current_iou,logic_str(formula))
                best_iou = current_iou

            for _, neighbor_formula, neighbor_vector in nonzero_features:
                for n in get_compositions(formula,vector,neighbor_formula,neighbor_vector):
                    neighbors.append(n)
            
        new_queue = []
        scored_neighbors = formula_iou(neuron,neighbors,16384)
        for item in scored_neighbors:
            key = tensor_key(item[2])
            prior_score = score_track.get(key,0)
            score = abs(item[0]) * length_penalty_factor(item[1],penalty)
            if score  > prior_score:
                score_track[key] = score
                heapq.heappush(new_queue,(-item[0],queue_id,item[1],item[2]))
                queue_id+=1
        
        queue = new_queue
        if len(queue) > beam_size:
            queue = heapq.nsmallest(beam_size, queue)
            heapq.heapify(queue)

def Search(neuron,feature_vectors):
    device = resolve_device()
    print("Neuron shape:",neuron.shape)
    print("Feature shape:",feature_vectors[0][1].shape)
    neuron = to_binary_tensor(neuron,device)
    scored_features = formula_iou(neuron,prepare_feature_vectors(feature_vectors,device),4096)
    nonzero_features = [feature for feature in scored_features if feature[0] > 0]
    nonzero_features.sort(key=lambda x: x[0],reverse=True)
    print("Pre/Post zero filtering:",len(scored_features),len(nonzero_features))
    
    beam_size = 30
    formula_length = 6
    queue = []
    # Load queue
    queue_id = 0
    penalty = 0.01
    score_track = {}
    for item in nonzero_features:
        formula = item[1]
        heapq.heappush(queue,(-item[0],queue_id,formula,item[2]))
        score_track[formula] = item[0]
        queue_id+=1
    if len(queue) > beam_size:
        queue = heapq.nsmallest(beam_size, queue)
        heapq.heapify(queue)
    
    best_score = 0
    best_iou = 0
    max_expansions = beam_size * max(0, formula_length - 1)
    expansions = 0
    while queue and expansions < max_expansions:
        rank_heuristic, _, formula, vector = heapq.heappop(queue)

        current_iou = abs(rank_heuristic)
        current_score = current_iou * length_penalty_factor(formula,penalty)
        if current_score > best_score:
            print("score:",current_score,logic_str(formula))
            best_score = current_score
        if current_iou > best_iou:
            print("iou:",current_iou,logic_str(formula))
            best_iou = current_iou

        neighbors = []
        for _, neighbor_formula, neighbor_vector in nonzero_features:
            for n in get_compositions(formula,vector,neighbor_formula,neighbor_vector):
                neighbors.append(n)
            
        scored_neighbors = formula_iou(neuron,neighbors,16384)
        for item in scored_neighbors:
            formula = item[1]
            key = formula
            prior_score = score_track.get(key,0)
            score = abs(item[0]) * length_penalty_factor(formula,penalty)
            if score  > prior_score:
                score_track[key] = score
                heapq.heappush(queue,(-item[0],queue_id,formula,item[2]))
                queue_id+=1
        
        if len(queue) > beam_size:
            queue = heapq.nsmallest(beam_size, queue)
            heapq.heapify(queue)

def search_worker(neurons,feature_vectors):
    shape = neurons.shape
    for i in range(shape[1]):
        LevelSearch(neurons[:,i],feature_vectors)
