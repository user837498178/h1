# Smart-Meter Load Forecasting Demo

This repository illustrates several time-series forecasting models on the **UK Smart Meter CC_LCL-FullData** set.

* Autoformer
* 1-D CNN
* GRU
* LSTM
* SVR

`data_loader.py` offers a flexible API that supports
1. **Whole-population aggregation** (mean or sum) – forecast the overall demand in one go.
2. **Per-household filtering** – train / evaluate an individual `LCLid` only.

---

## 1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or install packages listed in each script
```
A CUDA build of PyTorch will be used automatically if available.

---

## 2. Data
Download: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/

1. Place **CC_LCL-FullData.csv** under `electricity_forecasting/` (or supply an explicit path through `file_path`).  
2. Example format:

```
LCLid,stdorToU,DateTime,KWH/hh (per half hour) 
MAC000002,Std,2012-10-12 00:30:00.0000000, 0 
...
```

---

## 3. One-shot training – aggregated demand
When `household_id` is *not* specified, `load_data()` performs a mean (default) or sum aggregation across all households at every 30-min timestamp, yielding a single univariate series.

```bash
# Autoformer with mean aggregation
python electricity_forecasting/autoformer.py

# Sum aggregation (override in code or interactively)
python - << 'PY'
from electricity_forecasting.autoformer import main
from functools import partial
from electricity_forecasting import data_loader

data_loader.load_data = partial(data_loader.load_data, aggregate="sum")
main()
PY
```
Scripts `cnn.py`, `gru.py`, `lstm.py`, `svr.py` run in the same fashion.

---

## 4. Per-household training

### 4.1 Single household
```bash
python - << 'PY'
from electricity_forecasting.lstm import main
from functools import partial
from electricity_forecasting import data_loader

data_loader.load_data = partial(data_loader.load_data, household_id="MAC000123")
main()
PY
```

### 4.2 Iterate over all households
The snippet below loops through every `LCLid`, trains a model, and logs results. Parallelisation or sampling is recommended for large-scale runs.

```python
from pathlib import Path
import pandas as pd
from electricity_forecasting import data_loader
from electricity_forecasting.lstm import main as lstm_train
from functools import partial

CSV_PATH = Path("electricity_forecasting/CC_LCL-FullData.csv")
ids = pd.read_csv(CSV_PATH, usecols=["LCLid"]).LCLid.unique()
for hid in ids:
    print(f"\n=== Household {hid} ===")
    data_loader.load_data = partial(data_loader.load_data, household_id=hid)
    lstm_train()
```
Run:
```bash
python scripts/train_all_households.py | tee train_all.log
```

---

## 5. Visualisation
Each script saves a figure such as `lstm_predictions.png`, comparing ground truth and forecasts on the test set.

---

## Code at a glance
* `data_loader.py` – data loading & preprocessing
* `hybrid_cnn_transformer.py` – CNN + Transformer load forecaster
* `fullversion.py` – multi-channel predictor + Bayesian uncertainty + SAC scheduling skeleton 