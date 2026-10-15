from analysis import parse_formula
from pathlib import Path
import pandas as pd


csv_path = Path("../data/snli_beam_results.csv")
results = pd.read_csv(csv_path)

results.formula[1:2].map(parse_formula)