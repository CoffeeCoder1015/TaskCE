from typing import Callable

from attr import dataclass
from datasets import Dataset


@dataclass
class CaptureConfig:
    name: str
    dataset: Dataset
    data_formatter: Callable
    label_field: str = "answer"
