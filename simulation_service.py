from typing import Dict, Tuple

import numpy as np
import pandas as pd

from battery_models import BatterySpec

INTERVAL_HOURS = 0.25


def simulate_battery(df: pd.DataFrame, spec: BatterySpec, initial_soc: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    result = df.copy()
    charge_eff = np.sqrt(spec.roundtrip_efficiency)
    discharge_eff = np.sqrt(spec.roundtrip_efficiency)
    soc = max(0.0, min(spec.usable_kwh, initial_soc * spec.usable_kwh))

    soc_trace = []
    charge_trace = []
    discharge_trace = []
    export_trace = []
    import_trace = []

    max_charge_step = spec.max_charge_kw * INTERVAL_HOURS
    max_discharge_step = spec.max_discharge_kw * INTERVAL_HOURS

    for _, row in result.iterrows():
        surplus = float(row["surplus_kwh"])
        deficit = float(row["deficit_kwh"])

        charge_input = min(surplus, max_charge_step, (spec.usable_kwh - soc) / charge_eff if charge_eff > 0 else 0)
        stored = charge_input * charge_eff
        soc += stored

        available_from_battery = min(soc, max_discharge_step / discharge_eff if discharge_eff > 0 else 0)
        discharge_from_soc = min(available_from_battery, deficit / discharge_eff if discharge_eff > 0 else 0)
        discharge_to_load = discharge_from_soc * discharge_eff
        soc -= discharge_from_soc

        remaining_export = max(0.0, surplus - charge_input)
        remaining_import = max(0.0, deficit - discharge_to_load)

        soc_trace.append(soc)
        charge_trace.append(charge_input)
        discharge_trace.append(discharge_to_load)
        export_trace.append(remaining_export)
        import_trace.append(remaining_import)

    result["battery_charge_kwh"] = charge_trace
    result["battery_discharge_kwh"] = discharge_trace
    result["soc_kwh"] = soc_trace
    result["grid_export_after_battery_kwh"] = export_trace
    result["grid_import_after_battery_kwh"] = import_trace
    result["self_consumption_with_battery_kwh"] = result["direct_self_consumption_kwh"] + result["battery_discharge_kwh"]

    metrics = {
        "total_consumption_kwh": result["consumption_kwh"].sum(),
        "total_production_kwh": result["production_kwh"].sum(),
        "battery_charge_kwh": result["battery_charge_kwh"].sum(),
        "battery_discharge_kwh": result["battery_discharge_kwh"].sum(),
        "grid_import_without_battery_kwh": result["deficit_kwh"].sum(),
        "grid_export_without_battery_kwh": result["surplus_kwh"].sum(),
        "grid_import_with_battery_kwh": result["grid_import_after_battery_kwh"].sum(),
        "grid_export_with_battery_kwh": result["grid_export_after_battery_kwh"].sum(),
        "direct_self_consumption_kwh": result["direct_self_consumption_kwh"].sum(),
        "self_consumption_with_battery_kwh": result["self_consumption_with_battery_kwh"].sum(),
        "final_soc_kwh": result["soc_kwh"].iloc[-1] if len(result) else 0.0,
    }
    return result, metrics


def calculate_financials(result_df: pd.DataFrame, metrics: Dict[str, float], battery_price: float) -> Dict[str, float]:
    without_battery_cost = (
        result_df["deficit_kwh"] * result_df["import_price_eur_per_kwh"]
        - result_df["surplus_kwh"] * result_df["export_price_eur_per_kwh"]
    ).sum()
    with_battery_cost = (
        result_df["grid_import_after_battery_kwh"] * result_df["import_price_eur_per_kwh"]
        - result_df["grid_export_after_battery_kwh"] * result_df["export_price_eur_per_kwh"]
    ).sum()
    saving = without_battery_cost - with_battery_cost
    payback_years = battery_price / saving if saving > 0 else np.nan

    self_consumption_ratio_without = metrics["direct_self_consumption_kwh"] / metrics["total_production_kwh"] if metrics["total_production_kwh"] > 0 else np.nan
    self_consumption_ratio_with = metrics["self_consumption_with_battery_kwh"] / metrics["total_production_kwh"] if metrics["total_production_kwh"] > 0 else np.nan
    autarky_without = metrics["direct_self_consumption_kwh"] / metrics["total_consumption_kwh"] if metrics["total_consumption_kwh"] > 0 else np.nan
    autarky_with = metrics["self_consumption_with_battery_kwh"] / metrics["total_consumption_kwh"] if metrics["total_consumption_kwh"] > 0 else np.nan

    return {
        "cost_without_battery_eur": without_battery_cost,
        "cost_with_battery_eur": with_battery_cost,
        "theoretical_saving_eur": saving,
        "simple_payback_years": payback_years,
        "self_consumption_ratio_without": self_consumption_ratio_without,
        "self_consumption_ratio_with": self_consumption_ratio_with,
        "autarky_without": autarky_without,
        "autarky_with": autarky_with,
    }
