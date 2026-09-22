# ICU Mortality Prediction with GRU Models

This repository implements Gated Recurrent Unit (GRU) architectures to predict in-hospital mortality using multivariate physiological time-series data from ICU stays. It includes pipelines for baseline supervised training, self-supervised pretraining, and fine-tuning.

---

## Dataset

This project uses the **PhysioNet / Computing in Cardiology Challenge 2012** dataset:
* **Source:** [PhysioNet Challenge 2012 (v1.0.0)](https://physionet.org/content/challenge-2012/1.0.0/)

### Extracted Features
The pipeline processes 48-hour records across 8 key physiological measurements:
* Heart Rate (`HR`)
* Systolic Arterial Blood Pressure (`SysABP`)
* Diastolic Arterial Blood Pressure (`DiasABP`)
* Mean Arterial Blood Pressure (`MeanABP`)
* Respiration Rate (`RespRate`)
* Temperature (`Temp`)
* Peripheral Oxygen Saturation (`SpO2`)
* Glucose (`Glucose`)

Missing observations are forward-filled, mean-imputed, and standardized across the cohort.

---

## Directory Setup

Because clinical datasets and binary checkpoint files should not be stored in version control, set up your local workspace as follows:

```text
├── data/                      # Local data folder (git-ignored)
│   ├── set-a/                 # Extracted patient text records
│   │   ├── 132539.txt
│   │   └── ...
│   └── Outcomes-a.txt         # Mortality labels
├── data_utils.py              # Data extraction, imputation, and normalization
├── model.py                   # Baseline GRU mortality classification model
├── pretrain_model.py          # Autoregressive / reconstruction GRU model
├── finetune_model.py          # Fine-tuning GRU model architecture
├── train_baseline.py          # Baseline supervised training pipeline
├── train_finetune.py          # Fine-tuning training pipeline
├── test_data.py               # Preprocessing validation script
├── requirements.txt           # Environment dependencies
└── README.md
