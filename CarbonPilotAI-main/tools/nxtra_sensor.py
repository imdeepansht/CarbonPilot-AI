"""
nxtra_sensor.py - hourly data-centre sensor anchored to Nxtra by Airtel (FY 2025-26)
===================================================================================
Reads Airtel_Nxtra_Combined.xlsx and rebuilds a realistic hourly stream:

  * energy MIX (grid / solar / wind / diesel) comes from the Nxtra GRI 302-1 figures
  * grid emission factor is Nxtra's implied factor: Scope 2 / grid MWh (~0.71 tCO2/MWh)
  * diesel CO2 uses the IPCC default for diesel (74.1 kg CO2 per GJ of fuel energy)
  * the annual figures are scaled down to a site size you choose (--mean-kw)

Hourly shapes (solar bell curve, wind noise, mild diurnal load) are SIMULATED; only the
annual totals and shares are real. Nxtra does not disclose PUE, and its reported total
energy is facility energy, so no PUE multiplier is applied here.

Usage:
    python nxtra_sensor.py Airtel_Nxtra_Combined.xlsx
    python nxtra_sensor.py Airtel_Nxtra_Combined.xlsx --mean-kw 1000 --hours 48
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

GJ_PER_MWH = 3.6
DIESEL_KG_CO2_PER_GJ = 74.1  # IPCC default for diesel fuel (not from the workbook)


def read_metrics(path):
    m = pd.read_excel(path, sheet_name="Nxtra Metrics", header=2)
    m = m.dropna(subset=["Metric", "Value"])
    m["Term"] = m["Term"].astype(str).str.strip()
    m["Metric"] = m["Metric"].astype(str).str.strip()
    m["Value"] = pd.to_numeric(m["Value"], errors="coerce")
    lookup = {(r["Term"], r["Metric"]): float(r["Value"]) for _, r in m.dropna(subset=["Value"]).iterrows()}

    def get(term, metric):
        return lookup[(term, metric)]

    return {
        "grid_gj": get("Non-renewable", "Electricity (grid)"),
        "diesel_gj": get("Diesel", "Diesel consumption"),
        "solar_gj": get("Renewable", "Solar"),
        "wind_gj": get("Renewable", "Wind"),
        "total_gj": get("Total", "Total energy consumption"),
        "scope1_t": get("Scope 1", "Gross Scope 1 emissions"),
        "scope2_t": get("Scope 2", "Location-based Scope 2 emissions"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--mean-kw", type=float, default=1000.0,
                    help="average facility power of the simulated site (kW)")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--out", default="nxtra_sensor.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = read_metrics(a.xlsx)
    total = d["grid_gj"] + d["diesel_gj"] + d["solar_gj"] + d["wind_gj"]
    share = {k: d[f"{k}_gj"] / total for k in ("grid", "diesel", "solar", "wind")}
    grid_mwh = d["grid_gj"] / GJ_PER_MWH
    ef_g_per_kwh = d["scope2_t"] / grid_mwh * 1000.0  # tCO2/MWh == kg/kWh -> x1000 = g/kWh
    fleet_mean_mw = (total / GJ_PER_MWH) / 8760.0

    rng = np.random.default_rng(a.seed)
    start = datetime.now().replace(minute=0, second=0, microsecond=0)

    # solar bell curve (6:00-19:00) normalised so its daily mean equals the real solar share
    hrs = np.arange(24)
    bell = np.where((hrs >= 6) & (hrs <= 19), np.sin(np.pi * (hrs - 6) / 13), 0.0).clip(0)
    solar_peak = share["solar"] * a.mean_kw / bell.mean()

    rows = []
    for h in range(a.hours):
        ts = start + timedelta(hours=h)
        load = a.mean_kw * (1 + 0.04 * np.sin(2 * np.pi * (ts.hour - 9) / 24)) * (1 + rng.normal(0, 0.01))
        solar = solar_peak * bell[ts.hour] * max(0.0, 1 + rng.normal(0, 0.08))
        wind = share["wind"] * a.mean_kw * max(0.0, 1 + rng.normal(0, 0.4))
        diesel = share["diesel"] * a.mean_kw * max(0.0, 1 + rng.normal(0, 0.05))
        solar = min(solar, load)
        wind = min(wind, load - solar)
        diesel = min(diesel, load - solar - wind)
        grid = load - solar - wind - diesel
        ef = ef_g_per_kwh * (1 + rng.normal(0, 0.02))
        co2 = grid * ef / 1000.0 + diesel * 3.6 / 1000.0 * DIESEL_KG_CO2_PER_GJ  # kWh -> GJ = x0.0036
        rows.append({
            "timestamp": ts.isoformat(),
            "sensor_id": "DC-01",
            "sector": "data_centre",
            "source_dataset": "Nxtra by Airtel FY2025-26",
            "grid_region": "India (CEA, implied)",
            "facility_power_kw": round(load, 2),
            "grid_kw": round(grid, 2),
            "solar_kw": round(solar, 2),
            "wind_kw": round(wind, 2),
            "diesel_kw": round(diesel, 2),
            "grid_intensity_g_per_kwh": round(ef, 1),
            "energy_kwh": round(load, 2),
            "co2_kg": round(co2, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    print("From the workbook (annual, FY 2025-26):")
    print(f"  fleet average load   : {fleet_mean_mw:,.1f} MW")
    print(f"  mix                  : " + ", ".join(f"{k} {v*100:.1f}%" for k, v in share.items()))
    print(f"  implied grid factor  : {ef_g_per_kwh:,.0f} gCO2/kWh  (Scope 2 {d['scope2_t']:,.0f} t / {grid_mwh:,.0f} MWh)")
    print(f"Simulated site ({a.mean_kw:,.0f} kW mean), {len(df)} h:")
    e = {k: df[f'{k}_kw'].sum() for k in ("grid", "solar", "wind", "diesel")}
    tot = sum(e.values())
    print("  resulting mix        : " + ", ".join(f"{k} {v/tot*100:.1f}%" for k, v in e.items()))
    print(f"  total CO2            : {df['co2_kg'].sum():,.0f} kg")


if __name__ == "__main__":
    main()
