# EASG Classification & Action Prediction

## Project Overview
This repository implements **Egocentric Action Sequence Graph (EASG)** classification and action prediction from video data. It provides data handling, model definitions, training scripts, and evaluation utilities for the task of recognizing actions, objects, and relationships in egocentric video streams.

## High‑Level Architecture
```
src/
├─ data/          # Dataset modules (raw loading, processing, video dataset
│   └─ datasets/  # Concrete dataset implementations
├─ models/        # Model definitions (ViT, auto‑encoders, GNN classifiers)
├─ utils/         # Helper utilities (logging, checkpointing, misc.)
└─ scripts/       # Entry‑point scripts (run_easg, training wrappers)
```

- **src/data** – Loads annotation files, ROI features, and video tensors. It contains `dataset.py`, `dataset_ae.py`, and the `dataset_video` package.
- **src/models** – Core model building blocks (`simple_vit.py`, `autoencoder.py`, `vitDenoise_model.py`, etc.) and the high‑level classifier `EASGClassifier`.
- **src/utils** – Generic utilities such as WandB configuration helpers, checkpoint saving, and auxiliary functions used across the codebase.
- **src/scripts** – Command‑line entry points. `run_easg.py` provides the main data preparation routine used by training and evaluation scripts.

## Current Status
The codebase has been **restructured into a clean package layout** and imports have been updated accordingly. Functionality remains identical, but the repository has **not been fully tested** after the move. Most scripts run, but edge cases (e.g., custom data paths) may still break.

## Known Limitations
- Hard‑coded default paths (e.g., `./src/data/datasets/dataset_video/...`) assume the repository root as the working directory.
- Some scripts still rely on external data directories (`./data/` for ROI features) that are not part of the repo.
- The training and evaluation loops have not been exercised after the restructuring, so runtime errors may surface.

## Training & Evaluation Workflow
1. **Prepare data** – Ensure the annotation files (`verbs.txt`, `objects.txt`, `relationships.txt`) are under `annts_in_new_format/` and ROI/clip features are reachable via the `--data_path` argument.
2. **Run training** – Use one of the training scripts (`train.py`, `train_ae.py`, `train_vae.py`, `train_vitDiffusion.py`). Arguments control model hyper‑parameters, epochs, and WandB logging.
3. **Evaluate** – Execute `eval.py` with a trained model checkpoint. The script loads the dataset, runs inference, and prints recall metrics.
4. **Diffusion models** – `train_vitDiffusion.py` and `vitDenoise_model.py` operate on the video dataset located at `src/data/datasets/dataset_video/`.

## Future Work
- **Comprehensive testing** of all scripts after the restructure.
- Introduce a lightweight configuration system (e.g., Hydra or JSON/YAML) to replace hard‑coded defaults.
- Clean up legacy path assumptions and add unit tests for data loaders.
- Document model architecture details (layer sizes, loss functions) in separate design docs.
- Continuous‑integration pipelines for automated testing.
