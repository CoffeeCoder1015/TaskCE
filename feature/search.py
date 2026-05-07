import heapq

import torch
from feature.formula import And, Not, Or

def batched_iou(neuron,binary_vecs):
    intersections = (binary_vecs & neuron).sum(dim=1)
    unions = (binary_vecs | neuron).sum(dim=1)
    raw_scores = intersections.float() / unions.clamp_min(1).float()
    return torch.where(unions > 0, raw_scores, torch.zeros_like(raw_scores))

def score_formulas(neuron,formulas_vectors,batch_size):
    candidates = []
    for i in range(0, len(formulas_vectors), batch_size):
        batch = formulas_vectors[i:i+batch_size]
        formulas = [formula for formula, _ in batch]
        binary_vectors =[bin_vec for _, bin_vec in batch] 
        vectors = torch.stack(binary_vectors)
        iou = batched_iou(neuron,vectors)
        length_penalties = torch.tensor(
            [
                0.99 ** (len(formula) - 1)
                for formula in formulas
            ],
            dtype=iou.dtype,
            device=iou.device,
        )
        scores = (iou * length_penalties).tolist()
        candidates.extend(zip(scores,formulas,binary_vectors))
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

def tensor_key(vector):
    vector = vector.detach().to(device="cpu", dtype=torch.bool).contiguous()
    return (tuple(vector.shape), vector.numpy().tobytes())

def get_compositions(current_formula, current_vector, feature_formula, feature_vector):
    return [
        (And(left=current_formula, right=feature_formula), current_vector & feature_vector),
        (Or(left=current_formula, right=feature_formula), current_vector | feature_vector),
        (And(left=current_formula, right=Not(feature_formula)), current_vector & ~feature_vector),
    ]

def Search(neuron,feature_vectors):
    device = resolve_device()
    print("Neuron shape:",neuron.shape)
    print("Feature shape:",feature_vectors[0][1].shape)
    neuron = to_binary_tensor(neuron,device)
    scored_features = score_formulas(neuron,prepare_feature_vectors(feature_vectors,device),4096)
    nonzero_features = [feature for feature in scored_features if feature[0] > 0]
    print("Pre/Post zero filtering:",len(scored_features),len(nonzero_features))
    
    beam_size = 30
    formula_length = 6
    queue = []
    # Load queue
    queue_id = 0
    for item in nonzero_features:
        heapq.heappush(queue,(-item[0],queue_id,item[1],item[2]))
        queue_id+=1
    if len(queue) > beam_size:
        queue = heapq.nsmallest(beam_size, queue)
        heapq.heapify(queue)
    
    best_iou = 0

    max_expansions = beam_size * max(0, formula_length - 1)
    expansions = 0
    visited = set()
    while queue and expansions < max_expansions:
        iou, _, formula, vector = heapq.heappop(queue)
        iou = abs(iou)
        if iou > best_iou:
            best_iou = iou
            print(iou,formula.flatten())

        # Check visit
        key = tensor_key(vector)
        if key in visited:
            print("Already visited:",iou,formula.flatten())
            continue
        visited.add(key)

        neighbors = []
        for _, neighbor_formula, neighbor_vector in nonzero_features:
            for n in get_compositions(formula,vector,neighbor_formula,neighbor_vector):
                # Early skip
                if tensor_key(n[1]) in visited:
                    continue
                neighbors.append(n)
            
        scored_neighbors = score_formulas(neuron,neighbors,4096)
        for item in scored_neighbors:
            heapq.heappush(queue,(-item[0],queue_id,item[1],item[2]))
            queue_id+=1
        
        if len(queue) > beam_size:
            queue = heapq.nsmallest(beam_size, queue)
            heapq.heapify(queue)

def search_worker(neurons,feature_vectors):
    shape = neurons.shape
    for i in range(shape[1]):
        Search(neurons[:,i],feature_vectors)
