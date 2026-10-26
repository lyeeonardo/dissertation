"""True-state-conditioned action diffusion, reserved as a GenAIF oracle reference.

This policy alone receives the 145 physical labels and current inlet. It is not
an information-matched competitor to the public-sensor baselines. The frozen
checkpoint can later be queried with inferred state hypotheses instead.
"""

from __future__ import annotations

import json

import os

from pathlib import Path

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)
