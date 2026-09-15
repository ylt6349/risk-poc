#!/bin/bash
python scripts/synthetic_data.py
python scripts/compute_features.py
python scripts/train_model.py
uvicorn scripts.serve_model:app --host 0.0.0.0 --port 8000
