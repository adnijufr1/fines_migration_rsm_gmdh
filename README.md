# Fines-migration permeability decline — RSM and GMDH models

Reproducibility code for the hybrid-RSM and GMDH models used to predict
fines-migration-induced permeability decline in sandstone reservoirs.

## Scripts

| Script | Model | Data used |
|---|---|---|
| `rsm/rsm_hybrid_model.py` | Hybrid Response Surface Methodology (Ridge regression on degree-3 polynomial features) | `rsm/MASTER_Train_Data_Hybrid.csv`, `rsm/MASTER_Test_Data_Hybrid.csv` (7 features) |
| `gmdh/GMDH_Master_Script.m` | GMDH (Group Method of Data Handling) | `gmdh/MASTER_Train_Data.csv`, `gmdh/MASTER_Test_Data.csv` (5 features) |

Both scripts use the same underlying 436/146 train/test split; the two data file
pairs differ only in which input columns are included (RSM uses 7 features
including zeta potential; GMDH uses the reduced 5-feature set).

## Setup

### Python (RSM)
```
pip install -r requirements.txt
cd scripts
python rsm_hybrid_model.py
```

### MATLAB (GMDH)
Requires the GMDH toolbox functions `gmdhbuild`, `gmdhpredict`, and `gmdheq` —
Run from the `gmdh/` or `gmdh_functions/`  directory with
`data/MASTER_Train_Data.csv` and `data/MASTER_Test_Data.csv` on the MATLAB path.
```
GMDH_Master_Script
```

## Dependencies

- Python: see `requirements.txt`
- MATLAB: GMDH-type Polynomial Neural Networks toolbox by Gints Jekabsons
  (`gmdhbuild`, `gmdhpredict`, `gmdheq`) — see Acknowledgments below.

## Data

`rsm/MASTER_Train_Data_Hybrid.csv` / `rsm/MASTER_Test_Data_Hybrid.csv` and
`gmdh/MASTER_Train_Data.csv` / `gmdh/MASTER_Test_Data.csv` originate from the
same 436/146 sample split, saved separately per feature set used by each model.

## Acknowledgments

The GMDH model in this repository (`gmdh/GMDH_Master_Script.m`) is built on
the GMDH-type Polynomial Neural Networks toolbox for MATLAB by Gints Jekabsons:

> Jekabsons G. GMDH-type Polynomial Neural Networks for Matlab, 2010,
> available at http://www.cs.rtu.lv/jekabsons/

We thank the author for making this toolbox available. See
`gmdh_functions/` for the toolbox files and their original license.

## License

Code written for this repository (the RSM and GMDH driver scripts) is released
under the MIT License (see `LICENSE`). The GMDH toolbox files under
`gmdh_functions` are third-party software by Gints Jekabsons, licensed
under the GNU General Public License v2 (or later)
