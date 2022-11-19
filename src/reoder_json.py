"""
Reorders json predictions
"""
import pandas as pd

(
    pd.read_json("predictions.json")
    .reset_index()
    .sort_values("index")
    .set_index("index")
    .to_json()
)
