# src/utils

Utility helpers used across the repository.

- **utils.py** – Functions for loading pre‑computed embeddings, converting triplets to words, and other misc. data helpers.
- **utils_diffusion.py** – Scheduler and beta‑schedule utilities for diffusion experiments.
- **set_wandb_config**, **load_model**, **save_checkpoint** – Wrappers around WandB logging and model checkpointing.

These utilities are intentionally lightweight and have no dependencies on the core model code.
