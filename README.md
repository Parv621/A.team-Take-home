# Media performance anomaly detection and attribution

This repository contains a take-home data science exercise for detecting unusual changes in paid-media performance and ranking plausible explanations. The primary deliverable is [`design_proposal.md`](design_proposal.md); the notebooks provide the supporting implementation and evaluation.

## Run end to end

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, install the dependencies, and run the complete workflow:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

`python run_all.py` is the single command for the end-to-end workflow. It regenerates the deterministic synthetic data and classification-flow diagram, then executes the anomaly-detection and attribution notebooks in order. Notebook-generated CSV and PNG outputs are intentionally ignored by Git.

## Project structure

```text
.
|-- design_proposal.md              # Primary design proposal
|-- run_all.py                      # End-to-end entry point
|-- requirements.txt                # Python dependencies
|-- data/
|   |-- README.md                   # Dataset overview and regeneration notes
|   |-- data_dictionary.md          # Field and cause definitions
|   |-- design_decisions.md         # Synthetic-data assumptions
|   |-- generate_synthetic_data.py  # Deterministic data generator
|   |-- brief_metrics.csv           # Brief-level modeling input
|   |-- panel_data_full.csv         # Observable enriched panel
|   `-- ground_truth/               # Evaluation-only labels and event log
|-- anomaly_detection/
|   |-- README.md
|   |-- anomaly_detection.ipynb     # Main notebook for Anomaly detection
|   `-- proposal_anomaly_detection.md
`-- attribution/
    |-- README.md
    |-- attribution.ipynb           # Main notebook for Attribution
    |-- attribution_lib.py
    |-- proposal_attribution.md
    |-- classification_flow.png
    `-- render_classification_flow.py
```

## Time spent

Approx 1.5 working days of effort.

- Data Generation: 3 hours
- Anomaly detection + understanding trends: 4 hours
- Attribution analysis: 4 hours
- Report writing, edits and publishing: 3 hours

## Reproducibility note

The generator uses a fixed random seed. Files under `data/ground_truth/` are used only to construct labels and evaluate recovery; they are not model features.
