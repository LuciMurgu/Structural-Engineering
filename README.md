# BG-EXP-001 — Behavioral geometry on spring networks

Standalone, deterministic six-DOF spring-network experiment. No Virtual Cell Challenge files are read or modified.

## Reproduce

From this directory with Python 3.12 and `uv` (or install `requirements.txt` with pip):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/monograph_checks.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_experiment.py prepare
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_experiment.py develop
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_experiment.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_experiment.py final
PYTHONPATH=src MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/make_report.py
```

`final` refuses changed protocol files or a second final run. For independent reproduction, copy this directory to a new location, remove generated `data/`, `results/`, `manifests/splits.json`, and `manifests/protocol_lock.json`, then run the commands above. The configuration, complete split IDs and hashes, protocol lock, raw pair distances, plots, and final decision are included in the workspace.

The source monograph is *Geometry That Understands Behavior*, version 1.0, 23 October 2026 (original PDF SHA-256 `fcea08ff2ad755782cb783a0023726a8b75870099b0d59296c17bddb9971fef4`). The original PDF and named `behavioral_metric_demo.py` were not retained beside the workspace. `references/Geometry_That_Understands_Behavior_extracted.txt` preserves the extracted monograph text read for this experiment. `scripts/monograph_checks.py` independently reconstructs its five Appendix B demonstrations and is **not** the missing companion script.

See [the prespecified protocol](manifests/PROTOCOL.md) and [the scientific report](reports/EXP_001_SCIENTIFIC_REPORT.md).
