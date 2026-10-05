"""
infra_sensor.py - build the Infrastructure sensor from the india_dc_* dataset
=============================================================================
Reads india_dc_telemetry_hourly.csv (+ india_dc_facilities.csv for context) and writes
an hourly sensor CSV in the same layout the CarbonPilot backend loads
(data/sensors/infra.csv).

What it does
  1. picks one facility (default: the one with the most grid-outage hours)
  2. picks the window of --hours consecutive hours containing the most outage hours
  3. scales the site down so its MEAN facility power equals --mean-kw (1000 kW matches
     the Tech and AI sensors); every power and CO2 column is scaled by the same factor
  4. splits power into grid_kw / diesel_kw from the grid_available and dg_running flags
  5. keeps the cooling / UPS / PUE breakdown and the recorded outage flags as extra columns

Notes on the source data
  * It is SYNTHETIC (fictional operators, simulated weather) per its own data dictionary.
  * Its co2_kg uses one flat 0.71 kg/kWh factor, including diesel hours. That is kept as-is,
    so grid intensity is flat at 710 g/kWh unless you pass --grid-ef.

Usage:
    python infra_sensor.py --telemetry india_dc_telemetry_hourly.csv --facilities india_dc_facilities.csv
    python infra_sensor.py --list
    python infra_sensor.py --facility IN-DC-003 --hours 72 --mean-kw 2000 --grid-ef 900
"""
import argparse

import pandas as pd


def pick_window(d, hours):
    """Start index (on a day boundary where possible) of the window with most outage hours."""
    n = len(d)
    if n <= hours:
        return 0
    starts = [i for i in range(0, n - hours + 1) if d["timestamp"].iloc[i].hour == 0] or list(range(0, n - hours + 1))
    return max(starts, key=lambda i: (int((d["grid_available"].iloc[i:i + hours] == 0).sum()), -i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default="india_dc_telemetry_hourly.csv")
    ap.add_argument("--facilities", default="india_dc_facilities.csv")
    ap.add_argument("--facility", help="facility_id (default: most grid-outage hours)")
    ap.add_argument("--hours", type=int, default=96, help="window length in hours")
    ap.add_argument("--mean-kw", type=float, default=1000.0, help="target mean facility power after scaling")
    ap.add_argument("--grid-ef", type=float, default=None,
                    help="override grid intensity in g/kWh (default: keep the file's 710)")
    ap.add_argument("--list", action="store_true", help="list facilities in the telemetry and exit")
    ap.add_argument("--out", default="infra.csv")
    a = ap.parse_args()

    t = pd.read_csv(a.telemetry, parse_dates=["timestamp"])
    fac = pd.read_csv(a.facilities).set_index("facility_id")

    summary = t.groupby("facility_id").agg(
        city=("city", "first"), mean_kw=("total_facility_kw", "mean"),
        outage_h=("grid_available", lambda x: int((x == 0).sum())))
    summary["class"] = fac["facility_class"].reindex(summary.index)
    summary["renewable_pct"] = fac["renewable_share_pct"].reindex(summary.index)
    if a.list:
        print(summary.round(0).to_string())
        return

    fid = a.facility or summary["outage_h"].idxmax()
    if fid not in summary.index:
        raise SystemExit(f"{fid} not in telemetry. Use --list to see options.")
    d = t[t["facility_id"] == fid].sort_values("timestamp").reset_index(drop=True)
    i0 = pick_window(d, a.hours)
    w = d.iloc[i0:i0 + a.hours].reset_index(drop=True)

    k = a.mean_kw / w["total_facility_kw"].mean()
    ef = (w["co2_kg"] / w["total_facility_kw"] * 1000.0) if a.grid_ef is None else pd.Series(a.grid_ef, index=w.index)
    total = w["total_facility_kw"] * k
    grid_up = w["grid_available"] == 1
    dg_on = w["dg_running"] == 1

    out = pd.DataFrame({
        "timestamp": w["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "sensor_id": "INFRA-01",
        "sector": "infrastructure",
        "source_dataset": "india_dc synthetic telemetry",
        "facility_id": fid,
        "city": w["city"],
        "facility_class": fac.loc[fid, "facility_class"],
        "facility_power_kw": total.round(2),
        "it_power_kw": (w["it_load_kw"] * k).round(2),
        "cooling_kw": (w["cooling_kw"] * k).round(2),
        "ups_loss_kw": (w["ups_loss_kw"] * k).round(2),
        "misc_overhead_kw": (w["misc_overhead_kw"] * k).round(2),
        "pue": w["pue"],
        "grid_kw": total.where(grid_up, 0.0).round(2),
        "diesel_kw": total.where(dg_on, 0.0).round(2),
        "grid_intensity_g_per_kwh": ef.round(1),
        "energy_kwh": total.round(2),
        "co2_kg": (total * ef / 1000.0).round(2),
        "grid_available": w["grid_available"],
        "dg_running": w["dg_running"],
        "outside_temp_c": w["outside_temp_c"],
    })
    out.to_csv(a.out, index=False)

    print(f"Facility   : {fid} ({w['city'].iloc[0]}, {fac.loc[fid, 'facility_class']}, "
          f"{fac.loc[fid, 'renewable_share_pct']}% renewable per facilities file)")
    print(f"Window     : {out['timestamp'].iloc[0]} to {out['timestamp'].iloc[-1]} ({len(out)} h)")
    print(f"Scaling    : x{k:.5f} -> mean {out['facility_power_kw'].mean():,.0f} kW, peak {out['facility_power_kw'].max():,.0f} kW")
    print(f"PUE        : {w['pue'].mean():.2f} mean (cooling is {100 * w['cooling_kw'].sum() / w['total_facility_kw'].sum():.0f}% of facility power)")
    print(f"Grid       : {out['grid_intensity_g_per_kwh'].mean():,.0f} g CO2/kWh")
    print(f"Outages    : {int((~grid_up).sum())} h with grid down, diesel running {int(dg_on.sum())} h")
    print(f"Total CO2  : {out['co2_kg'].sum():,.0f} kg over {len(out)} h")


if __name__ == "__main__":
    main()
