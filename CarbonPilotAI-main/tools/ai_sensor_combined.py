"""
ai_sensor_combined.py - calibrate the training and inference AI sensor streams into one site
=============================================================================================
Inputs : ai_sensor.csv            (training cluster, near-flat load)
         ai_sensor_inference.csv  (inference serving, daily traffic curve)

Calibration steps
  1. Align on timestamp (the two files start an hour apart, so only overlapping hours are kept).
  2. Use ONE grid-intensity series for both (each file had its own independent noise).
  3. Normalise each stream to its own mean, then weight by --train-share so the shape of each
     is kept but the split between training and inference is explicit.
  4. Rescale the combined IT load so its PEAK equals --peak-kw.
  5. Recompute facility power, energy and CO2 from the shared PUE and grid series.

Usage:
    python ai_sensor_combined.py ai_sensor.csv ai_sensor_inference.csv
    python ai_sensor_combined.py ai_sensor.csv ai_sensor_inference.csv --peak-kw 1000 --train-share 0.6
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("training_csv")
    ap.add_argument("inference_csv")
    ap.add_argument("--peak-kw", type=float, default=1000.0,
                    help="peak combined IT load in kW (1000 matches the planner's default DC capacity)")
    ap.add_argument("--train-share", type=float, default=0.6,
                    help="share of average IT load that is training (assumption, 0-1)")
    ap.add_argument("--pue", type=float, default=1.5)
    ap.add_argument("--out", default="ai_sensor_combined.csv")
    a = ap.parse_args()

    t = pd.read_csv(a.training_csv, parse_dates=["timestamp"])
    i = pd.read_csv(a.inference_csv, parse_dates=["timestamp"])

    m = t.merge(i, on="timestamp", suffixes=("_train", "_inf"), how="inner")
    if m.empty:
        raise SystemExit("The two files share no timestamps.")

    # 3. shape-preserving weights
    w_t, w_i = a.train_share, 1 - a.train_share
    shape_t = m["it_power_kw_train"] / m["it_power_kw_train"].mean()
    shape_i = m["it_power_kw_inf"] / m["it_power_kw_inf"].mean()
    combined = w_t * shape_t + w_i * shape_i

    # 4. peak calibration
    scale = a.peak_kw / combined.max()
    it_kw = combined * scale
    train_kw = w_t * shape_t * scale
    inf_kw = w_i * shape_i * scale

    # 2/5. shared grid series (from the training file), recompute everything downstream
    grid = m["grid_intensity_g_per_kwh_train"]
    fac_kw = it_kw * a.pue

    out = pd.DataFrame({
        "timestamp": m["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "sensor_id": "AI-01",
        "sector": "ai",
        "model": "training + inference (" + m["model_train"] + " | " + m["model_inf"] + ")",
        "grid_region": m["grid_region_train"],
        "training_it_kw": train_kw.round(2),
        "inference_it_kw": inf_kw.round(2),
        "it_power_kw": it_kw.round(2),
        "pue": a.pue,
        "facility_power_kw": fac_kw.round(2),
        "grid_intensity_g_per_kwh": grid.round(1),
        "energy_kwh": fac_kw.round(2),
        "co2_kg": (fac_kw * grid / 1000).round(2),
    })
    out.to_csv(a.out, index=False)

    print(f"Aligned rows     : {len(out)} (of {len(t)} / {len(i)})")
    print(f"Raw means (IT kW): training {t['it_power_kw'].mean():,.0f}, inference {i['it_power_kw'].mean():,.0f}")
    print(f"Scale factor     : x{scale:.3f}  -> peak {out['it_power_kw'].max():,.0f} kW, mean {out['it_power_kw'].mean():,.0f} kW")
    print(f"Split (mean kW)  : training {out['training_it_kw'].mean():,.0f} / inference {out['inference_it_kw'].mean():,.0f}")
    print(f"Total CO2        : {out['co2_kg'].sum():,.0f} kg over {len(out)} h")


if __name__ == "__main__":
    main()
