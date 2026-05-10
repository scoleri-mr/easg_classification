# src/models

Model definitions for the EASG pipeline.

- **simple_vit.py** – Vision Transformer backbone used in several experiments.
- **autoencoder.py**, **vitDenoise_model.py**, **denoise_model.py** – Auto‑encoder and diffusion model architectures.
- **models.py** – High‑level classifier (`EASGClassifier`) that combines verb, object, and relationship heads.

These modules are imported by the training scripts to instantiate the neural networks. No training logic lives here; they only define the network architectures and loss helpers.
