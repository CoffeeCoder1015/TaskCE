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
    i = 0
    stack = [-1]
    context = {-1:{"op":[]}}
    while i < len(tokens):
        if tokens[i] == "(":
            context[i] = {"op":[]}
            stack.append(i)
            i += 1
            continue

        latest = stack[-1]
        if tokens[i] == ")":
            latest = stack.pop()
            expr = None
            match context[latest]["opr"]:
                case "AND":
                    expr  = z3.And(*context[latest]["op"])
                case "OR":
                    expr  = z3.Or(*context[latest]["op"])
                case "NOT":
                    expr  = z3.Not(*context[latest]["op"])
            context[stack[-1]]["op"].append(expr)
            i += 1
            continue

        if tokens[i] in [ "AND", "OR", "NOT" ]:
            context[latest]["opr"] = tokens[i]
            i += 1
            continue

        context[latest]["op"].append(z3.Bool(tokens[i]))

        i += 1

    return context[-1]["op"][0]