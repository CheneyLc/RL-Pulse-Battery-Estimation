"""Compute the data behind every figure and save it as CSV next to the
plotting notebooks:

    <figures>/main/<Figure>/data/*.csv
    <figures>/supplementary/<Figure>/data/*.csv

where <figures> is "figures" for Dataset 1 and "figures_dataset2" for Dataset 2.

Usage
-----
    python run_all.py                                   # Dataset 1, main + supplementary
    python run_all.py --part main                       # main-text figures only
    python run_all.py --part supplementary              # supplementary figures only
    python run_all.py --sections Figure4 Figure5        # selected figures only
    python run_all.py --dataset 2                       # Dataset 2
    python run_all.py --dataset 2 --part supplementary  # options can be combined
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from Code import common
from Code.main import data_density, figure2, figure3, figure4, figure5, figure6
from Code.supplementary import model_generality

ROOT = Path(__file__).resolve().parent
FIG_ROOTS = {"dataset1": ROOT / "figures", "dataset2": ROOT / "figures_dataset2"}

PARTS = {
    "main": ["Figure2", "Figure3", "Figure4", "Figure5", "Figure6"],
    "supplementary": ["Model_generality"],
}
ALL_SECTIONS = PARTS["main"] + PARTS["supplementary"]
PART_OF = {s: part for part, sections in PARTS.items() for s in sections}

# Figures 3d, 5j and 5k share the data-density result.
DENSITY_TARGETS = ["Figure3", "Figure5"]


def save(section, run, tables):
    d = FIG_ROOTS[run] / PART_OF[section] / section / "data"
    for name, df in tables.items():
        path = d / name
        path.parent.mkdir(parents=True, exist_ok=True)
        keep_index = not isinstance(df.index, pd.RangeIndex)
        df.to_csv(path, index=keep_index, index_label="row" if keep_index else None)
        print(f"   saved {path.relative_to(ROOT)}  {df.shape}")


def run_main(sections, run):
    if "Figure2" in sections:
        print("\n=== Figure 2 ===")
        save("Figure2", run, figure2.process_data())

    if not any(s in sections for s in ["Figure3", "Figure4", "Figure5", "Figure6"]):
        return

    df = common.load_all_data(common.material_files)
    df_train_used, df_test = common.split_train_test(df)
    M =common.build_full_models(df_train_used)

    if "Figure6" in sections:
        print("\n=== Figure 6 ===")
        save("Figure6", run, figure6.process_data(df_test, M))
    if "Figure3" in sections:
        print("\n=== Figure 3 ===")
        save("Figure3", run, figure3.process_data(df_train_used, df_test, M))
    if "Figure4" in sections:
        print("\n=== Figure 4 ===")
        save("Figure4", run, figure4.process_data(df_test, M))
    if "Figure5" in sections:
        print("\n=== Figure 5 ===")
        save("Figure5", run, figure5.process_data(df_train_used, df_test, M))

    targets = [s for s in DENSITY_TARGETS if s in sections]
    if targets:
        print("\n=== Data density (Figures 3d, 5j, 5k) ===")
        res = data_density.process_data(df_train_used, df_test)
        for s in targets:
            save(s, run, res)


def run_supplementary(sections, run):
    if "Model_generality" in sections:
        print("\n=== Model generality ===")
        save("Model_generality", run, model_generality.process_data())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["main", "supplementary", "all"], default="all")
    ap.add_argument("--sections", nargs="+", choices=ALL_SECTIONS,
                    help="run only these figures (overrides --part)")
    ap.add_argument("--dataset", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()

    if args.sections:
        sections = args.sections
    elif args.part == "all":
        sections = ALL_SECTIONS
    else:
        sections = PARTS[args.part]

    t0 = time.time()
    run = common.use_dataset(args.dataset)
    print(f"##### {run}")
    run_main([s for s in sections if PART_OF[s] == "main"], run)
    run_supplementary([s for s in sections if PART_OF[s] == "supplementary"], run)
    print(f"\nDone in {time.time() - t0:.0f} s. Data written to {FIG_ROOTS[run]}")


if __name__ == "__main__":
    main()
