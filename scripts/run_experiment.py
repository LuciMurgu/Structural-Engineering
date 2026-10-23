#!/usr/bin/env python3
"""Run BG-EXP-001 stages. Final evaluation requires a verified protocol lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bg_exp001.evaluation import develop, finalize, freeze, prepare, verify_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "develop", "freeze", "final", "verify-lock"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "exp001.json").read_text())
    if args.stage == "prepare":
        prepare(root, cfg)
        print("Wrote parameter/load arrays and split manifest; no responses calculated.")
    elif args.stage == "develop":
        develop(root, cfg)
        print("Wrote development selection; final test responses were not read.")
    elif args.stage == "freeze":
        freeze(root)
        print("Wrote protocol lock before final test evaluation.")
    elif args.stage == "verify-lock":
        verify_lock(root)
        print("Protocol lock verified.")
    else:
        result = finalize(root, cfg)
        print(json.dumps({
            "decision": result["decision"],
            "selected_simple_baseline": result["selected_simple_baseline"],
            "primary_mean_spearman_difference": result["primary_mean_local_minus_baseline_spearman"],
            "primary_95pct_interval": result["uncertainty"]["local_minus_selected_baseline_spearman_95pct"],
        }, indent=2))


if __name__ == "__main__":
    main()
