from typing import Callable

from attr import dataclass
from datasets import Dataset

@dataclass
class EvalConfig:
    name:str
    dataset:Dataset
    data_formatter: Callable
    eval_fn: Callable
    
