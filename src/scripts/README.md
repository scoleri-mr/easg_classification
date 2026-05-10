# src/scripts

Command‑line entry points for the project.

- **run_easg.py** – Loads annotations and ROI/clip features, builds `EASGData` objects, and provides a convenient API for training scripts to obtain datasets.
- Additional scripts (e.g., training wrappers) import functions from this module to keep the main training files concise.

Running a script typically looks like:
```bash
python -m src.scripts.run_easg --path_to_annotations annts_in_new_format/ --path_to_data data/
```

The script is designed to be invoked from the repository root so that relative paths resolve correctly.
