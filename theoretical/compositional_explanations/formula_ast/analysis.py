"""Load compositional search CSVs for subsequent formula experiments."""

from pathlib import Path

import pandas as pd
import z3


def parse_formula(text: str):
    tokens = []
    last = 0
    # Lexer 
    for i in range(0,len(text)):
        if text[i] in "()":
            if i-last > 0:
                tokens.append(text[last:i])
            tokens.append(text[i])
            last = i+1
        elif text[i] == " ":
            if i-last > 0:
                tokens.append(text[last:i])
            last = i+1

    # Parser
