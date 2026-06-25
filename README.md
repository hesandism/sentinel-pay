# SentinelPay

SentinelPay is a production-style machine-learning system that detects fraudulent card 
transactions in real time. Transactions flow through a streaming pipeline, get enriched with 
behavioural features (spending velocity, geographic distance from the last transaction, deviation from a card’s normal spend), and are scored by a gradient-boosted model served behind a low latency API. Every prediction comes with a SHAP explanation of why it was flagged. The system continuously monitors incoming data for drift, and when the model’s reliability decays it automatically retrains, validates, and promotes a new version - the full MLOps loop, not just a notebook. 

## Project Progress

### Phase 1 — EDA & Data Preparation

- [x] Exploratory data analysis (`notebooks/01_eda_time_split.ipynb`)
- [x] Leakage-safe 80/20 chronological train/test split

**Key findings:**

| Metric | Value |
|--------|-------|
| Total transactions | 1,852,394 |
| Overall fraud rate | 0.52% |
| Time span | Jan 2019 – Dec 2020 |
| Train rows | 1,481,915 (Jan 2019 – Aug 2020) |
| Test rows | 370,479 (Aug 2020 – Dec 2020) |

### Upcoming Phases

- [ ] Feature engineering
- [ ] Model training (LightGBM)
- [ ] SHAP explanations & evaluation
- [ ] API serving & drift monitoring

## Project Structure

```
data/
├── fraudTrain.csv          # Raw Kaggle dataset (train)
├── fraudTest.csv           # Raw Kaggle dataset (test)
└── processed/
    ├── train_time_split.csv  # Chronological 80% train split
    └── test_time_split.csv   # Chronological 20% test split
notebooks/
├── load_and_check.ipynb        # Initial sanity checks
└── 01_eda_time_split.ipynb     # EDA & time-based split
```

## Set Up

1. Manually download the train and test datasets from "https://www.kaggle.com/datasets/kartik2112/fraud-detection" and place them in the `data` folder. The datasets are too large to be included in this repository.

```
data
|__ fraudTest.csv
|__ fraudTrain.csv
```
2. Create a virtual environment and install the dependencies:

```
python -m venv .venv
.venv/bin/activate 
pip install -r requirements.txt 
```

