"""Canonical BSM1 PI and sensor repairs, isolated from the pinned upstream source."""

from __future__ import annotations

import copy

STREAMS = (
    "y_in1",
    "y_out1",
    "y_out2",
    "y_out3",
    "y_out4",
    "y_out5",
    "y_out5_r",
    "ys_in",
    "ys_out",
    "ys_eff",
    "ys_tss_internal",
    "sludge_height",
    "qintr",
    "klas",
)

def plant_state(plant):
    return {
        "reactors": [getattr(plant, f"reactor{i}").y0.copy() for i in range(1, 6)],
        "settler": plant.settler.ys0.copy(),
        "streams": {k: copy.deepcopy(getattr(plant, k)) for k in STREAMS},
    }

def restore_plant(plant, state):
    for i, y in enumerate(state["reactors"], 1):
        getattr(plant, f"reactor{i}").y0 = y.copy()
    plant.settler.ys0 = state["settler"].copy()
    for k, v in state["streams"].items():
        setattr(plant, k, copy.deepcopy(v))
