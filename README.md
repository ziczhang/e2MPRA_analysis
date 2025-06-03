# e2MPRA_analysis

This repository provides analysis scripts for the e2MPRA (epigenetic and expression Multiplexed Reporter Assay) study, implemented in Python. The code supports quality control, normalization, and visual/statistical analyses of MPRA and epigenomic data (ATAC-seq, CUT&Tag).

> **Note**: This repository includes Python source files in the `src` directory. Data visualization and figure generation for publication are carried out in the `ipynb` directory using Jupyter Notebooks.

## Directory Structure
```
e2MPRA_analysis/ 
├── src/
│ ├── parser.py
│ ├── quality_check.py
│ ├── ctlLibAnalysis.py
│ ├── HepG2lib_analysis.py
│ └── WTC11lib_analysis.py
├── ipynb/
│ └── [figure-generating notebooks]
└── data/
　└── [input data files]
```


## Modules

### `parser.py`
Implements the `CountTableParser` class:
- Loads and formats raw MPRA and epigenomic count data.
- Applies TMM normalization.
- Computes log fold changes (logFC) per replicate and across replicates.
- Handles `lentiMPRA`, `ATAC`, and `H3K27ac` assays.

### `quality_check.py`
Provides the `QuarityChecker` class for diagnostic plotting:
- Barcode count distribution per CRE.
- UMI count distribution and correlation between replicates.
- LogFC replicate correlation plots.
- Activity distributions across feature classes.

### `ctlLibAnalysis.py`
Contains the `CtlLibAnalyzer` class:
- Visualizes logFC distributions across control feature types.
- Performs statistical testing (Mann–Whitney U) across features.
- Generates violin and scatter plots for assay correlations.

### `HepG2lib_analysis.py`
Defines `HepG2LibAnalyzer` class for analyzing synthetic HepG2 CRE libraries:
- Assesses motif-site correlation with activity (Class 1).
- Evaluates combinatorial motif interactions (Class 2).
- Performs permutation-based activity analysis (Class 3).
- Provides statistical testing and multiple plotting utilities.

### `WTC11lib_analysis.py`
Implements `WTC11LibAnalyzer` for the WTC11 library:
- Analyzes single nucleotide substitution effects using regression.
- Maps perturbation effects along CREs using MAD scores.
- Annotates motif positions and visualizes variant effects across assays.

## Requirements

All scripts are written in Python and require the following packages:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `statsmodels`, `scipy`
- `statannotations`, `networkx` (for HepG2 analysis)
- `scikit-learn` (optional for some visualization utilities)

## License

MIT License

Copyright (c) 2025 Zicong Zhang
