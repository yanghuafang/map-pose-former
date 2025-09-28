#!/usr/bin/env python3
"""Train a MapPoseFormer.

    tools/train.py --config configs/synth_base.yaml
    tools/train.py --config configs/synth_base.yaml train.lr=1e-3 model.dim=192

Overrides are ``section.field=value`` and are validated against the dataclasses
in ``mapposeformer/config.py``, so a misspelled field fails immediately instead
of training a run that ignored it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import load_config, parse_overrides
from mapposeformer.engine import Trainer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", help="YAML config; omit for the dataclass defaults"
    )
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    Trainer(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
