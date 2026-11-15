# Reinforcement Learning Reveals Minimal Data Volume Requirement for Pulse-test Based Battery State Estimation

This repository contains the computation code and plotting notebooks associated with the
manuscript below. Every figure of the manuscript (main text and supplementary information) can be
regenerated with this code.

> **Data download.** The pulse-test data are not included in this repository. Download them from
> Zenodo: [doi.org/10.5281/zenodo.24236435](https://doi.org/10.5281/zenodo.24236435), and copy the
> `data/` folder of the Zenodo record into the root of this repository (next to `run_all.py`).
> The Zenodo record also contains the computed data (CSV) and figures (PNG) of the main-text figures
> of Dataset 1.

**Associated manuscript**

*Reinforcement Learning Reveals Minimal Data Volume Requirement for Pulse-test Based Battery State Estimation*

## 1. Project Overview

The project is organized into two technical parts:

1. Computation, in which the pulse-test data are used to train and evaluate the estimation models,
   and the data behind every figure are written to CSV files.
2. Plotting, in which one Jupyter notebook per figure reads those CSV files and draws the figure.

The figures are grouped into two parts, which can be run independently:

- `main`: the figures of the main text (Figures 2–6), based on an XGBoost cascade
  (pulse voltages → material → SOC → SOH);
- `supplementary`: the model-generality analysis, which repeats the cascade with four other model
  families (RF, ExtraTrees, SVM, MLP).

Both parts can be computed for either of the two data sets described in Section 2.1.

The workflow is:

`data/ -> run_all.py -> figures/<part>/<Figure>/data/*.csv -> <Figure>.ipynb -> <Figure>_<nn>_<title>.png`

## 2. Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- run_all.py
|-- data/                  (download from Zenodo)
|   |-- dataset1/
|   `-- dataset2/
|-- Code/
|   |-- common.py
|   |-- main/
|   |   |-- figure2.py
|   |   |-- figure3.py
|   |   |-- figure4.py
|   |   |-- figure5.py
|   |   |-- figure6.py
|   |   `-- data_density.py
|   `-- supplementary/
|       `-- model_generality.py
|-- figures/
|   |-- run_all_figures.ipynb
|   |-- main/
|   |   |-- Figure2/
|   |   |-- Figure3/
|   |   |-- Figure4/
|   |   |-- Figure5/
|   |   `-- Figure6/
|   `-- supplementary/
|       `-- Model_generality/
`-- figures_dataset2/
    `-- (same layout as figures/)
```

`figures/` is used for Dataset 1 and `figures_dataset2/` for Dataset 2. Each figure folder contains
the plotting notebook `<Figure>.ipynb`; `run_all.py` writes the CSV files into its `data/` folder and
the notebook saves the figures next to it. The repository contains only the notebooks; the raw data
(`data/`) and the results of the main-text figures of Dataset 1 (`figures/main/`) are on Zenodo
(see the download note at the top).

### 2.1 Datasets

Each Excel workbook holds one battery material at one pulse width. Every row is one pulse test with
the battery number `No.`, `SOC` (%), `SOH` and the pulse voltages `U1`–`U20`.

**Dataset 1** (`data/dataset1/`, Supplementary Table S3): batteries tested with a 5,000 ms pulse width.

| Material        | Capacity (Ah) | Pulse width (ms) | Battery number |  Sample number | SOC range (%) | SOH range  |
| --------------- | ------------: | ---------------: | -------------: | -------------: | ------------- | ---------- |
| NMC             |           2.1 |             5000 |             67 |            670 | 5–50         | 0.61–0.92 |
| LMO             |            10 |             5000 |             95 |           1523 | 5–85         | 0.51–0.95 |
| LFP             |            35 |             5000 |             56 |            909 | 5–90         | 0.74–0.96 |
| **Total** |               |                  |  **218** | **3102** |               |            |

Files: `NMC_2.1Ah_W_5000.xlsx`, `LMO_10Ah_W_5000.xlsx`, `LFP_35Ah_W_5000.xlsx`.

**Dataset 2** (`data/dataset2/`, Supplementary Table S5): additional batteries tested with nine pulse
widths. The battery and sample numbers are per pulse width; every battery is tested at all nine widths.

| Material        | Capacity (Ah) | Pulse width (ms)                           | Battery number in each width | Sample number in each width | SOC range (%) | SOH range  |
| --------------- | ------------: | ------------------------------------------ | ---------------------------: | --------------------------: | ------------- | ---------- |
| NMC             |            21 | 30, 50, 70, 100, 300, 500, 700, 1000, 3000 |                           52 |                         975 | 5–90         | 0.75–1.0  |
| LMO             |            24 | 30, 50, 70, 100, 300, 500, 700, 1000, 3000 |                           64 |                        1018 | 5–90         | 0.64–1.0  |
| LFP             |            20 | 30, 50, 70, 100, 300, 500, 700, 1000, 3000 |                          102 |                        1487 | 5–75         | 0.64–0.83 |
| **Total** |               |                                            |                **218** |              **3480** |               |            |

Files: `<material>_<capacity>Ah_W_<pulse width>.xlsx`, e.g. `LFP_20Ah_W_300.xlsx` (27 files).
In the computations, the nine pulse-width files of each material are pooled into one data set
(31,320 samples in total).

The released Dataset 2 files also contain samples with SOH slightly above 1 (NMC: 339 samples per
pulse width, maximum 1.011; LMO: 152 samples per pulse width, maximum 1.121); they are used as provided.

## 3. Folder-Level Description

### 3.1 `data/`

The pulse-test workbooks of the two data sets (Section 2.1). They are not part of this repository:
download them from Zenodo (see the download note at the top) and copy the `data/` folder into the
repository root.
They are read directly by the code; no preprocessing is needed.

### 3.2 `Code/`

The computation code.

- `common.py`: data-set selection (`use_dataset`), data loading and the full-feature XGBoost models used by Figures 3–6 (material classifier, per-material SOC and SOH
  estimators)
- `main/figure2.py`: samples for the pulse-voltage curves and the dataset Sankey diagram
- `main/figure3.py`: material classification; data-sufficiency grids over the number of pulses and
  over positive/negative pulse combinations × training sample ratio; Q-learning search for the
  minimal data requirement
- `main/figure4.py`: SOC estimation with the per-material SOC models
- `main/figure5.py`: SOH estimation with (SOC-aware) and without (SOC-unaware) SOC input
- `main/figure6.py`: propagation of material and SOC errors to SOH estimation
- `main/data_density.py`: kNN density of the test samples with respect to the training samples and
  the SOH error (Figures 3d, 5j, 5k)
- `supplementary/model_generality.py`: the cascade with RF, ExtraTrees, SVM and MLP, ten
  feature-selection seeds, and Q-learning on the resulting performance landscapes

The pulse voltages `U1`–`U20` form five pulses of four voltages each; the models use these 20 features:

```python
PULSE_FEATURES = {
    1: ["U1", "U2", "U3", "U4"],        # positive pulse
    2: ["U5", "U6", "U7", "U8"],        # negative pulse
    3: ["U9", "U10", "U11", "U12"],     # positive pulse
    4: ["U13", "U14", "U15", "U16"],    # negative pulse
    5: ["U17", "U18", "U19", "U20"],    # positive pulse
}
```

The main experiment settings are:

| Setting                                 | Value                                                                            | Location                                                      |
| --------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Data sets                               | `DATASETS` (files and pulse widths of Datasets 1 and 2)                        | `common.py`                                                 |
| Training sample ratios                  | 0.1, 0.2, …, 1.0                                                                | `figure3.sample_ratios`, `model_generality.sample_ratios` |
| Numbers of pulses                       | 1–5                                                                             | `figure3.pulse_nums`, `model_generality.pulse_nums`       |
| Repeats per grid cell (main)            | 3 (seeds 0, 1, 2)                                                                | `figure3.pulse_count_grid`, `figure3.pos_neg_grid`        |
| Q-learning cost and tolerance           | cost = number of pulses + 5 × ratio; tolerance 0.01 (SOH, SOC), 0.02 (material) | `figure3.run_task_rl`, `figure3.rl_optimum`               |
| kNN density                             | k = 10                                                                           | `data_density.compute_knn_density`                          |
| Model families (supplementary)          | RF, ExtraTrees, SVM, MLP                                                         | `model_generality.MODEL_NAMES`, `build_models`            |
| Feature-selection seeds (supplementary) | 10, 20, …, 100                                                                  | `model_generality.FEATURE_SEEDS`                            |

### 3.3 `run_all.py`

Runs the computations and writes the CSV files into the figure folders. Figures that share a
computation (the full-feature models of Figures 3–6 and the data-density result of Figures 3d, 5j
and 5k) are computed once per run. The options are:

- `--dataset`: `1` (default) or `2`
- `--part`: `main`, `supplementary` or `all` (default)
- `--sections`: selected figure folders, e.g. `Figure4 Figure5` (overrides `--part`)

### 3.4 `figures/` and `figures_dataset2/`

The figure folders of Dataset 1 and Dataset 2, with the same layout. Each figure folder contains:

- `data/`: the CSV files written by `run_all.py`
- `<Figure>.ipynb`: the plotting notebook, which only reads `data/`
- `<Figure>_<nn>_<title>.png`: the figures, saved when the notebook is run

This repository contains only the notebooks. The data and figures of `figures/main/` (main-text
figures, Dataset 1) are published on Zenodo; all folders are filled by running `run_all.py` and the
notebooks.
`run_all_figures.ipynb` executes all plotting notebooks of the folder.

The figure folders are:

| Folder                             | Figures                                                                                                                                                                                                                                                                    | Data                                                                                                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `main/Figure2`                   | 01 mean pulse-voltage curves at SOC 10/30/50 % per material and the three materials at SOC 50 %; Sankey diagram `dataset_sankey_material_SOC_SOH.html`                                                                                                                   | `samples.csv`                                                                                                                                                                                        |
| `main/Figure3`                   | 01, 02 confusion matrix of material classification; 03 data sufficiency, number of pulses × training ratio; 04 data sufficiency, positive/negative pulse combinations × training ratio; Q-learning result (table); 05 mean kNN data density (Figure 3d)                  | `material_predictions.csv`, `classes.csv`, `pulse_count_*.csv`, `pos_neg_*.csv`, `rl_optimum.csv`, `rl_V_*.csv`, `df_res.csv`                                                            |
| `main/Figure4`                   | 01–03 predicted vs. true SOC per material; 04–06 SOC error per material; 07, 08 all test samples                                                                                                                                                                         | `soc_predictions.csv`                                                                                                                                                                                |
| `main/Figure5`                   | 01–06 SOC-aware SOH estimation (parity and error per material); 07–12 SOC-unaware SOH estimation; 13 relative R² improvement; 14 SOH error vs. SOC; 15, 16 SOH error per material; 17 SOH MAE vs. data density (Figure 5j); 18 marginal change in SOH error (Figure 5k) | `soh_predictions.csv`, `df_cmp.csv`, `df_res.csv`                                                                                                                                                |
| `main/Figure6`                   | 01, 05 SOH error in three scenarios (control, SOC bias, wrong material); 02 SOC error with a wrong material; 03 SOC error vs. SOH error; 04 SOH error under SOC perturbation                                                                                               | `res_impact.csv`, `df_soc_err.csv`, `df_soh_err.csv`, `df_gain.csv`, `df_summary.csv`                                                                                                        |
| `supplementary/Model_generality` | mean and standard deviation of SOH MAE, SOC MAE and material accuracy (number of pulses × training ratio) for each model family; Q-learning results                                                                                                                       | `dataset_summary.csv`, `raw_results.csv`, `<model>/*_mean.csv`, `<model>/*_std.csv`, `RL_optimal_data_requirement.csv`, `RL_optimal_for_each_seed.csv`, `RL_configuration_frequency.csv` |

## 4. Environment

```bash
conda create -n Pulse python=3.10 -y
conda activate Pulse
pip install -r requirements.txt
python -m ipykernel install --user --name pulse --display-name "Python (Pulse)"
```

`matplotlib-inline` must be 0.1.6; later versions do not work with matplotlib 3.5.3.

## 5. Usage

### 5.1 Computations

Run from the repository root:

```bash
conda activate Pulse
python run_all.py                                   # Dataset 1, main and supplementary
python run_all.py --part main                       # Dataset 1, main-text figures only
python run_all.py --part supplementary              # Dataset 1, supplementary figures only
python run_all.py --sections Figure4 Figure5        # Dataset 1, selected figures only
python run_all.py --dataset 2                       # Dataset 2, main and supplementary
python run_all.py --dataset 2 --part main           # Dataset 2, main-text figures only
```

On Windows, set `PYTHONIOENCODING=utf-8` first if the console cannot print the characters `±`, `²` or `→`.

Approximate run times on a 32-thread desktop:

| Part                             | Dataset 1 | Dataset 2     |
| -------------------------------- | --------- | ------------- |
| main (Figures 2–6)              | 10 min    | 15 min        |
| supplementary (Model generality) | 45 min    | several hours |

Most of the main-part time is spent in the two data-sufficiency grids of Figure 3.

### 5.2 Figures

Open `figures/<part>/<Figure>/<Figure>.ipynb` (or the same notebook in `figures_dataset2/`) with the
kernel `Python (Pulse)` and run all cells; the figures are saved next to the notebook.
Alternatively, run `figures/run_all_figures.ipynb` (or `figures_dataset2/run_all_figures.ipynb`),
which executes all notebooks of that folder; set `PARTS` to `["main"]`, `["supplementary"]` or both.

## 6. Reproducibility

All random seeds are fixed, so the computations give the same numbers on every run with the library
versions in `requirements.txt`. The only exception is the horizontal jitter of the points in the strip
plots of Figure 5 (14–16), which is drawn by seaborn without a fixed seed and changes from run to run;
the box plots and all numbers are unaffected.
