"""Streaming plant integration with commanded hydraulic flows applied consistently.

Reactor/settler equations come unchanged from the pinned backend. Recycle
composition is delayed one inner integration interval; commanded volumes are
immediate. This is an explicitly tested operator-splitting approximation.
"""

import warnings

import numpy as np

from bsm2_python.bsm1_base import BSM1Base

from scipy.integrate import ODEintWarning

from simulator.inputs import constant_input

VOLUMES = np.array([1000.0, 1000.0, 1333.0, 1333.0, 1333.0])

class StreamingPlant(BSM1Base):
    def __init__(self):
        super().__init__(data_in=constant_input(0.01), timestep=1 / 1440, endtime=0.0)
        for i in range(1, 6):
            setattr(self, f"y_out{i}", getattr(self, f"reactor{i}").y0.copy())
        self.ys_tss_internal = self.settler.ys0[70:80].copy()
        self.ys_was = self.ys_out.copy()
        self.ys_out[14] = self.settler.q_r
        self.ys_was[14] = self.settler.q_w
        self.y_out5_r = self.y_out5.copy()
        self.y_out5_r[14] = self.qintr

    def advance(self, influent, dt, day, klas, imlr, ras, was):
        if imlr < 0 or ras < 0 or was < 0 or influent[14] <= was:
            raise ValueError("Infeasible hydraulic command")
        self.klas = np.asarray(klas, dtype=float).copy()
        self.qintr = float(imlr)
        self.settler.q_r, self.settler.q_w = float(ras), float(was)
        self.ys_out[14], self.y_out5_r[14] = ras, imlr
        self.y_in1 = self.combiner.output(influent, self.ys_out, self.y_out5_r)
        previous = self.y_in1
        with warnings.catch_warnings():
            warnings.simplefilter("error", ODEintWarning)
            for i, kla in enumerate(klas, 1):
                reactor = getattr(self, f"reactor{i}")
                reactor.kla = kla
                previous = reactor.output(dt, day, previous)
                setattr(self, f"y_out{i}", previous)
            self.ys_in, self.y_out5_r = self.splitter.output(self.y_out5, (float(influent[14] + ras), float(imlr)))
            self.ys_out, self.ys_was, self.ys_eff, self.sludge_height, self.ys_tss_internal = self.settler.output(
                dt, day, self.ys_in
            )
        states = np.concatenate([*[getattr(self, f"reactor{i}").y0[:14] for i in range(1, 6)], self.settler.ys0])
        if not np.isfinite(states).all() or np.min(states) < -1e-6:
            raise FloatingPointError("Nonfinite or negative physical state")
        if not np.isclose(self.ys_eff[14] + self.ys_was[14], influent[14], atol=1e-7):
            raise FloatingPointError("External flow imbalance")
        return float(np.min(states))
