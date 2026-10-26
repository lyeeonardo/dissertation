"""Independent physical accounting. Croll training reward remains conflicting across main and SI."""

import numpy as np

from .plant import VOLUMES

LIMITS = {"COD": 100.0, "BOD5": 10.0, "TSS": 30.0, "TN": 18.0, "NHx": 4.0}

def concentrations(y):
    y = np.asarray(y)
    tkn = y[..., 9] + y[..., 10] + y[..., 11] + 0.08 * (y[..., 4] + y[..., 5]) + 0.06 * (y[..., 2] + y[..., 6])
    return {
        "COD": y[..., :7].sum(axis=-1),
        "BOD5": 0.25 * (y[..., 1] + y[..., 3] + 0.92 * (y[..., 4] + y[..., 5])),
        "TSS": y[..., 13],
        "TN": tkn + y[..., 8],
        "NHx": y[..., 9],
        "TKN": tkn,
        "NOx": y[..., 8],
    }

def quality_rate(y):
    c = concentrations(y)
    return (2 * c["TSS"] + c["COD"] + 30 * c["TKN"] + 10 * c["NOx"] + 2 * c["BOD5"]) * y[..., 14] / 1000

def inventory(plant):
    reactor_mass = sum(getattr(plant, f"y_out{i}")[13] * v for i, v in enumerate(VOLUMES, 1))
    n = int(plant.settler.layer[1])
    settler_mass = np.sum(plant.settler.ys0[7 * n : 8 * n]) * np.prod(plant.settler.dim) / n
    return float((reactor_mass + settler_mass) / 1000)

def rates(plant):
    c = concentrations(plant.ys_eff)
    ae = float(np.dot(VOLUMES, plant.klas) * 8 / 1800)
    pe = float(0.004 * plant.qintr + 0.008 * plant.ys_out[14] + 0.05 * plant.ys_was[14])
    me = float(0.005 * 24 * VOLUMES[plant.klas < 20].sum())
    # PROPOSED: BSM1 non-N terms; replace the nitrogen terms with soluble R5 signals.
    proxy = (
        (2 * c["TSS"] + c["COD"] + 2 * c["BOD5"] + 30 * (plant.y_out5[9] + plant.y_out5[10]) + 10 * plant.y_out5[8])
        * plant.ys_eff[14]
        / 1000
    )
    return {
        "AE_kWh_per_day": ae,
        "PE_kWh_per_day": pe,
        "ME_kWh_per_day": me,
        "EQI_kgPU_per_day": float(quality_rate(plant.ys_eff)),
        "waste_kgSS_per_day": float(plant.ys_was[13] * plant.ys_was[14] / 1000),
        "effluent_kgSS_per_day": float(plant.ys_eff[13] * plant.ys_eff[14] / 1000),
        "training_quality_proxy_per_day": float(proxy),
    }
