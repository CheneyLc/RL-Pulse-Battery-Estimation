"""Figure 2: samples used for the pulse-voltage curves and the dataset Sankey diagram."""
import pandas as pd

from ..common import material_files

U_cols = [f"U{i}" for i in range(1, 21)]


def process_data():
    parts = []
    for material, files in material_files.items():
        for path in files:
            df = pd.read_excel(path)
            df["Mat"] = material
            parts.append(df[["Mat", "No.", "SOC", "SOH"] + U_cols])
    return {"samples.csv": pd.concat(parts, ignore_index=True)}
