"""Predeclared validation-plateau rule shared by all new model fits."""
from dataclasses import asdict, dataclass
import math


@dataclass
class Plateau:
    minimum_updates: int = 10_000
    patience: int = 5
    relative_improvement: float = 0.005
    best_value: float | None = None
    best_update: int | None = None
    reference_value: float | None = None
    bad_checks: int = 0
    last_update: int = -1
    monitoring_started: bool = False

    def observe(self, update: int, value: float) -> dict:
        if update <= self.last_update or update < 0 or not math.isfinite(value):
            raise ValueError('Require increasing updates and a finite validation loss')
        if value < 0:
            raise ValueError('This relative plateau rule requires nonnegative loss')
        self.last_update = update
        is_best = self.best_value is None or value < self.best_value
        if is_best:
            self.best_value, self.best_update = float(value), int(update)
        significant = False
        if update >= self.minimum_updates:
            if not self.monitoring_started:
                self.reference_value = float(value)
                self.bad_checks = 0
                self.monitoring_started = True
            elif value < self.reference_value * (1 - self.relative_improvement):
                self.reference_value = float(value)
                self.bad_checks = 0
                significant = True
            else:
                self.bad_checks += 1
        return dict(is_best=is_best, significant_improvement=significant,
                    stop=self.monitoring_started and self.bad_checks >= self.patience,
                    best_value=self.best_value, best_update=self.best_update,
                    bad_checks=self.bad_checks, reference_value=self.reference_value)

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**state)
