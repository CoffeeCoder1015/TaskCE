"""" 
Evaluation provider of the tasks and finetuned model
""""
from .evalConfig import EvalConfig
from .evaluator import Evaluate

__all__ = ["Evaluate","EvalConfig"]