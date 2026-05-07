from attr import dataclass
import torch

@dataclass
class SearchConfig:
    beam_size=5
    formula_length=5
    complexity_penalty=1.0
    score_batch_size=4096
    max_expansions=None

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

def Search(neuron,feature_vectors,config=SearchConfig()):
    device = resolve_device()
    print(neuron,neuron.shape)
    print(feature_vectors[0],feature_vectors[0][1].shape,len(feature_vectors))
    scored_features = score_formulas(to_binary_tensor(neuron,device),prepare_feature_vectors(feature_vectors,device),4096)
    print(scored_features[0],len(scored_features))
    
