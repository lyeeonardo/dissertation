"""Streaming evaluator; true effluent and inventory remain privileged."""

from .rewards import LIMITS, concentrations

class Evaluator:
    def __init__(self, initial_inventory):
        self.initial_inventory = float(initial_inventory)
        self.final_inventory = float(initial_inventory)
        self.days = 0.0
        self.volume = 0.0
        self.integrals = {}
        self.loads = {k: 0.0 for k in LIMITS}
        self.exceedance = {k: 0.0 for k in LIMITS}

    def add(self, effluent, rates, dt, inventory):
        self.days += dt
        self.volume += float(effluent[14] * dt)
        self.final_inventory = float(inventory)
        for k, v in rates.items():
            self.integrals[k] = self.integrals.get(k, 0.0) + float(v * dt)
        c = concentrations(effluent)
        for k, limit in LIMITS.items():
            self.loads[k] += float(c[k] * effluent[14] * dt)
            self.exceedance[k] += dt * bool(c[k] > limit)

    def summary(self):
        if self.days <= 0 or self.volume <= 0:
            return {"days": self.days, "status": "no_valid_intervals"}
        avg = {k: v / self.days for k, v in self.integrals.items()}
        sludge = (self.integrals["waste_kgSS_per_day"] + self.final_inventory - self.initial_inventory) / self.days
        oci = avg["AE_kWh_per_day"] + avg["PE_kWh_per_day"] + avg["ME_kWh_per_day"] + 5 * sludge
        croll_ae = avg["AE_kWh_per_day"] * 9.458 / 8
        croll_oci = croll_ae + avg["PE_kWh_per_day"] + avg["ME_kWh_per_day"] + 5 * avg["waste_kgSS_per_day"]
        return {
            "croll_si_evaluation": {
                "AE_kWh_per_day": croll_ae,
                "SP_kgSS_per_day": avg["waste_kgSS_per_day"],
                "OCI": croll_oci,
                "EQI": avg["EQI_kgPU_per_day"],
                "total_cost": croll_oci + avg["EQI_kgPU_per_day"],
                "source": "Recovered Croll SI Eqs.1-6; AE uses 9.458; SP excludes inventory",
                "mapping": "ASM1 total Kjeldahl N for effluent; SUMO equivalence unverified",
                "physical_backend_DO_saturation_mg_L": 8,
            },
            "days": self.days,
            "flow_weighted_mg_L": {k: v / self.volume for k, v in self.loads.items()},
            "exceedance_days": self.exceedance.copy(),
            "average_rates": avg,
            "inventory_change_kgSS": self.final_inventory - self.initial_inventory,
            "sludge_disposal_kgSS_per_day": sludge,
            "sludge_total_kgSS_per_day": sludge + avg["effluent_kgSS_per_day"],
            "OCI": oci,
            "EQI": avg["EQI_kgPU_per_day"],
            "total_cost": oci + avg["EQI_kgPU_per_day"],
            "metric_convention": "IWA BSM1; right endpoint inner-step rectangle; OCI=AE+PE+ME+5*SP",
            "integrals": self.integrals.copy(),
        }
