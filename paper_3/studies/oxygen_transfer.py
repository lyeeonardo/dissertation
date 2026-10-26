from __future__ import annotations

import numpy as np

from bsm1_repro.rewards import concentrations

from bsm1_offline_benchmark.paper_shared_dataset import PaperPublic57Env

DO_SD = .05

def sensor_error(noise_key, relative_minute, seed=17):
    """One held observation realization; no eta, action or policy enters its RNG."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(noise_key), int(relative_minute), 20261016]))
    return float(rng.normal(0., DO_SD))

class TransferEfficiencyEnv(PaperPublic57Env):
    """Native integration plus an isolated physical efficiency and DO instrument.

    The temporary advance wrapper changes the ODE input then restores the
    bookkeeping air command before CanonicalBSM1Env calls rates/evaluator.
    Snapshots retain full native state plus extension and sensor identity.
    """
    def __init__(self, *args, eta=1., root_minute=0, noise_key=0, noisy_do=True, **kwargs):
        if not .55 <= float(eta) <= 1.:
            raise ValueError('This bounded extension supports eta in[.55,1]')
        self.eta = float(eta)
        self.root_minute = int(root_minute)
        self.noise_key = int(noise_key)
        self.noisy_do = bool(noisy_do)
        self.minute_trace = []
        super().__init__(*args, **kwargs)

    def observation(self):
        public = self.clean_observation().copy()
        if self.noisy_do and self.minute >= self.root_minute:
            public[41] += sensor_error(self.noise_key, self.minute-self.root_minute)
        return public

    def step(self, action):
        native_advance = self.plant.advance
        # Instance wrapper is removed in finally; never serialized into a snapshot.
        def transfer_advance(influent, dt, day, klas, imlr, ras, was):
            command = np.asarray(klas, dtype=np.float64).copy()
            effective = command.copy(); effective[4] *= self.eta
            result = native_advance(influent, dt, day, effective, imlr, ras, was)
            self.plant.klas = command
            c = concentrations(self.plant.ys_eff)
            self.minute_trace.append([self.minute+1-self.root_minute, self.plant.y_out5[7],
                                      c['NHx'], c['TN'], command[4], effective[4]])
            return result
        self.plant.advance = transfer_advance
        try:
            return super().step(action)
        finally:
            del self.plant.advance

    def snapshot(self):
        value = super().snapshot()
        value['oxygen_probe'] = dict(eta=self.eta, root_minute=self.root_minute,
            noise_key=self.noise_key, noisy_do=self.noisy_do)
        return value

    def restore(self, snapshot):
        extension = snapshot.get('oxygen_probe')
        if extension:
            for key in ('eta', 'root_minute', 'noise_key', 'noisy_do'):
                setattr(self, key, extension[key])
        self.minute_trace = []
        return super().restore(snapshot)
