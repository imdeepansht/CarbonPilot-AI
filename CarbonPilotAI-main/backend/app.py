"""
CarbonPilot AI — Energy Source Switch Planner & Carbon-Water Impact Reporter
=============================================================
Single-file backend. Every model described in Table 3 of the technical
report lives in this file as its own clearly-marked section, orchestrated
by one pipeline function and served over one FastAPI app.

    MODEL 1  LoadForecastModel        -> Fourier-feature regression (Prophet/SARIMA-lite)
    MODEL 2  SolarForecastModel       -> weather-feature regression (cloud cover / irradiance)
    MODEL 3  SourceOptimizer          -> linear program over grid/solar/battery/diesel (PuLP)
    MODEL 4  ImpactCalculator         -> deterministic CO2 + water footprint engine
    MODEL 5  ESGNarrativeGenerator    -> LLM (Claude) narrative writer, template fallback

Run:
    pip install -r requirements.txt
    python app.py
    # -> http://localhost:8000  (docs at /docs)
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover
    PULP_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:  # pragma: no cover
    APSCHEDULER_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    REPORTLAB_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANTHROPIC_SDK_AVAILABLE = False


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.jsonl"
LATEST_FILE = DATA_DIR / "latest.json"
SENSOR_DIR = DATA_DIR / "sensors"
SENSOR_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Emission / water factors. Sources noted inline; these are published
# reference figures used for a deterministic (non-ML) calculation, exactly
# as scoped in the report ("not ML, but essential to the pipeline").
GRID_EF_BASE_G_PER_KWH = 713.0       # CEA CO2 Baseline Database, India grid average, order-of-magnitude
DIESEL_EF_KG_PER_L = 2.68            # IPCC default emission factor, diesel combustion
DIESEL_GENSET_KWH_PER_L = 3.0        # typical diesel genset output per litre (~0.33 L/kWh)
COOLING_WUE_L_PER_KWH = 1.8          # The Green Grid WUE benchmark, evaporative cooling
GRID_UPSTREAM_WATER_L_PER_KWH = 2.0  # thermal generation cooling-water withdrawal, industry order-of-magnitude
DIESEL_WATER_L_PER_KWH = 0.05        # radiator top-up, small vs. thermal grid generation

UNSERVED_PENALTY_INR_PER_KWH = 5000.0  # makes the LP report unserved load instead of becoming infeasible

DEFAULT_INTERVAL_SECONDS = int(os.environ.get("DEMO_INTERVAL_SECONDS", "3600"))


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class PlanRequest(BaseModel):
    city: str = Field(default="Pune, India")
    lat: float = Field(default=18.5204)
    lon: float = Field(default=73.8567)
    hours: int = Field(default=24, ge=6, le=48)
    dc_capacity_kw: float = Field(default=1000.0, gt=0)
    solar_capacity_kw: float = Field(default=400.0, ge=0)
    battery_capacity_kwh: float = Field(default=500.0, ge=0)
    battery_max_rate_kw: float = Field(default=150.0, ge=0)
    diesel_max_kw: float = Field(default=600.0, ge=0)
    grid_price_per_kwh: float = Field(default=8.0, gt=0)
    diesel_price_per_l: float = Field(default=95.0, gt=0)
    carbon_priority: float = Field(
        default=0.5, ge=0, le=1,
        description="0 = pure lowest-cost, 1 = pure lowest-carbon. Sets the LP's shadow carbon price."
    )
    simulate_outage: bool = Field(default=False)
    outage_start_hour: int = Field(default=14)
    outage_end_hour: int = Field(default=17)
    # --- sensor mode: when sensor_id is set, the sensor's readings replace the
    # synthetic load forecast and the diurnal grid-intensity model.
    sensor_id: Optional[str] = Field(default=None, description="tech | ai | infra. None = built-in forecast model.")
    sensor_load_scale: float = Field(default=1.0, gt=0, le=10, description="Multiplier on the sensor's power readings.")
    sensor_grid_multiplier: float = Field(default=1.0, gt=0, le=5, description="Multiplier on the sensor's grid intensity.")
    sensor_noise_pct: float = Field(default=0.0, ge=0, le=50, description="Gaussian noise (std, % of load) added to the sensor.")
    sensor_seed: int = Field(default=42)


class HourlyRecord(BaseModel):
    hour_index: int
    timestamp: str
    load_kw: float
    solar_available_kw: float
    grid_carbon_intensity_g_per_kwh: float
    grid_price_per_kwh: float
    grid_kw: float
    solar_kw: float
    battery_discharge_kw: float
    battery_charge_kw: float
    diesel_kw: float
    battery_soc_kwh: float
    unserved_kw: float = 0.0
    cost_inr: float
    co2_kg: float
    water_l: float
    grid_outage: bool


class PlanResponse(BaseModel):
    generated_at: str
    city: str
    lat: float
    lon: float
    weather_source: str
    optimizer_engine: str
    hourly: list[HourlyRecord]
    totals: dict
    baseline_totals: dict
    savings: dict
    mix_pct: dict
    sensor: Optional[dict] = None


class ESGNarrativeResponse(BaseModel):
    narrative: str
    generated_with: str


# ============================================================================
# LOCAL STORE  (JSON-file stand-in for Firebase/MongoDB Atlas — same
# schema-flexible document pattern, zero external service needed for a demo)
# ============================================================================

class LocalStore:
    def __init__(self):
        self._lock = threading.Lock()

    def save_plan(self, plan: dict) -> None:
        with self._lock:
            LATEST_FILE.write_text(json.dumps(plan))
            with HISTORY_FILE.open("a") as f:
                f.write(json.dumps({
                    "generated_at": plan["generated_at"],
                    "city": plan["city"],
                    "totals": plan["totals"],
                    "baseline_totals": plan["baseline_totals"],
                    "savings": plan["savings"],
                }) + "\n")

    def load_latest(self) -> Optional[dict]:
        if not LATEST_FILE.exists():
            return None
        return json.loads(LATEST_FILE.read_text())

    def load_history(self, limit: int = 30) -> list[dict]:
        if not HISTORY_FILE.exists():
            return []
        lines = HISTORY_FILE.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-limit:]]


store = LocalStore()
_last_params: Optional[PlanRequest] = None


# ============================================================================
# EXTERNAL DATA FEEDS  (Stage 1 of the report's system workflow)
# ============================================================================

def fetch_weather_forecast(lat: float, lon: float, hours: int) -> tuple[pd.DataFrame, str]:
    """Pull hourly cloud cover + shortwave radiation from Open-Meteo (free,
    no API key). Falls back to a synthetic clear-sky model if the network
    call fails, so the pipeline always produces a demoable result."""
    if httpx is not None:
        try:
            resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "cloudcover,shortwave_radiation,temperature_2m",
                    "forecast_days": 2,
                    "timezone": "auto",
                },
                timeout=6.0,
            )
            resp.raise_for_status()
            data = resp.json()["hourly"]
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["time"]),
                "cloud_cover_pct": data["cloudcover"],
                "shortwave_radiation": data["shortwave_radiation"],
                "temperature_c": data["temperature_2m"],
            })
            now = datetime.now()
            df = df[df["timestamp"] >= now - timedelta(hours=1)].reset_index(drop=True)
            if len(df) >= hours:
                return df.iloc[:hours].reset_index(drop=True), "open-meteo.com (live)"
        except Exception:
            pass  # fall through to synthetic feed

    return _synthetic_weather(hours), "synthetic clear-sky model (offline fallback)"


def _synthetic_weather(hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    rows = []
    cloud_regime = rng.uniform(10, 40)  # slowly-drifting cloud base for the window
    for h in range(hours):
        ts = start + timedelta(hours=h)
        hour_of_day = ts.hour
        # bell-shaped clear-sky irradiance, zero before 6am / after 18:30
        daylight = max(0.0, math.sin(math.pi * (hour_of_day - 6) / 13)) if 6 <= hour_of_day <= 19 else 0.0
        ghi = 950 * daylight
        cloud_regime += rng.normal(0, 4)
        cloud_regime = min(max(cloud_regime, 5), 85)
        rows.append({
            "timestamp": ts,
            "cloud_cover_pct": cloud_regime,
            "shortwave_radiation": ghi,
            "temperature_c": 24 + 6 * daylight + rng.normal(0, 0.5),
        })
    return pd.DataFrame(rows)


def generate_synthetic_load_history(capacity_kw: float, days: int = 21, seed: int = 42) -> pd.DataFrame:
    """A data-centre load profile is much flatter than an office building's:
    it never goes to zero, but still tracks business hours and weekly
    rhythm as batch jobs / user traffic ebb and flow. Used only to *train*
    the forecasting model — clearly a labelled synthetic substitute for
    real DC telemetry, per the report's data-sources section."""
    rng = np.random.default_rng(seed)
    hours = days * 24
    t = np.arange(hours)
    hour_of_day = t % 24
    day_of_week = (t // 24) % 7

    base = 0.80  # DC floor load never drops much — always-on servers
    diurnal_swing = 0.20 * np.sin((hour_of_day - 8) / 24 * 2 * np.pi) ** 2
    diurnal_swing *= np.where((hour_of_day >= 8) & (hour_of_day <= 22), 1.0, 0.3)
    weekend_dip = np.where(day_of_week >= 5, -0.04, 0.0)
    noise = rng.normal(0, 0.018, size=hours)

    load_fraction = base + diurnal_swing + weekend_dip + noise
    load_fraction = np.clip(load_fraction, 0.55, 1.05)
    load_kw = load_fraction * capacity_kw

    start = datetime.now() - timedelta(hours=hours)
    timestamps = [start + timedelta(hours=int(h)) for h in t]
    return pd.DataFrame({"timestamp": timestamps, "hour_index": t, "load_kw": load_kw})


# ============================================================================
# SENSORS  (three simulated carbon sensors, one per industry domain)
# Each sensor is an hourly CSV. The planner can run on a sensor's readings
# instead of its built-in load forecast. Minimum CSV: a power column
# (facility_power_kw / power_kw / load_kw / it_power_kw). Optional:
# timestamp, grid_intensity_g_per_kwh, co2_kg, grid_kw, solar_kw, wind_kw, diesel_kw.
# ============================================================================

SENSOR_REGISTRY = {
    "tech": {
        "label": "Tech - data centre",
        "domain": "Tech",
        "file": "tech_nxtra.csv",
        "blurb": "Data-centre stream anchored to Nxtra by Airtel FY 2025-26: real energy mix and grid factor, scaled to a 1 MW site.",
    },
    "ai": {
        "label": "AI - training + inference",
        "domain": "AI",
        "file": "ai_combined.csv",
        "blurb": "AI cluster running training and inference together on a coal-heavy grid: the high-emission case.",
    },
    "infra": {
        "label": "Infrastructure",
        "domain": "Infrastructure",
        "file": "infra.csv",
        "blurb": "Data-centre facility telemetry (synthetic India dataset, Kolkata hyperscale site): power chain, cooling, PUE and real grid-outage events.",
    },
}

# --- Built-in default data for the three sensors -----------------------
# On startup, any missing sensor file is restored: first from a copy sitting next
# to app.py or in the working directory, otherwise from the embedded defaults
# below. So the Tech and AI sensors work even if data/sensors/ was not copied
# over. A file you upload through "Update CSV" is never overwritten.
_DEFAULT_SENSOR_CSV = {
    "tech_nxtra.csv": """timestamp,sensor_id,sector,source_dataset,grid_region,facility_power_kw,grid_kw,solar_kw,wind_kw,diesel_kw,grid_intensity_g_per_kwh,energy_kwh,co2_kg
2026-10-03T05:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",968.3,844.22,0.0,61.24,62.84,682.3,968.3,592.77
2026-10-03T06:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",959.06,857.95,0.0,41.15,59.96,697.9,959.06,614.75
2026-10-03T07:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",988.62,683.68,193.2,48.35,63.4,716.6,988.62,506.86
2026-10-03T08:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",981.14,525.85,363.61,29.04,62.65,709.3,981.14,389.69
2026-10-03T09:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",998.15,391.95,476.52,70.14,59.55,703.9,998.15,291.78
2026-10-03T10:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1006.79,239.45,652.1,53.99,61.25,716.1,1006.79,187.82
2026-10-03T11:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1041.84,259.32,687.5,37.45,57.57,718.7,1041.84,201.74
2026-10-03T12:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1039.89,203.5,747.58,31.27,57.54,719.2,1039.89,161.72
2026-10-03T13:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1042.33,159.82,787.24,34.56,60.71,711.7,1042.33,129.93
2026-10-03T14:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1040.91,167.39,760.15,51.32,62.05,711.0,1040.91,135.56
2026-10-03T15:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1043.01,307.25,657.05,19.65,59.05,703.3,1043.01,231.85
2026-10-03T16:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1032.0,406.44,492.88,75.27,57.42,723.7,1032.0,309.48
2026-10-03T17:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1017.23,561.56,343.73,50.17,61.77,720.1,1017.23,420.86
2026-10-03T18:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1036.44,758.66,176.8,38.39,62.59,707.3,1036.44,553.28
2026-10-03T19:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1006.99,915.7,0.0,29.78,61.51,712.0,1006.99,668.41
2026-10-03T20:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1017.33,905.35,0.0,50.09,61.89,705.6,1017.33,655.33
2026-10-03T21:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1004.57,905.44,0.0,40.26,58.87,693.0,1004.57,643.19
2026-10-03T22:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",994.47,885.67,0.0,47.34,61.46,716.3,994.47,650.84
2026-10-03T23:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",986.52,887.62,0.0,39.13,59.77,686.0,986.52,624.89
2026-10-04T00:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",957.65,868.13,0.0,28.31,61.21,697.1,957.65,621.54
2026-10-04T01:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",961.71,859.09,0.0,40.39,62.23,696.7,961.71,615.16
2026-10-04T02:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",959.39,856.14,0.0,40.72,62.54,685.5,959.39,603.54
2026-10-04T03:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",964.17,872.59,0.0,35.91,55.67,711.0,964.17,635.28
2026-10-04T04:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",956.27,843.94,0.0,47.52,64.82,706.6,956.27,613.62
2026-10-04T05:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",955.48,840.14,0.0,51.25,64.09,721.9,955.48,623.56
2026-10-04T06:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",975.18,892.38,0.0,24.71,58.09,696.8,975.18,637.35
2026-10-04T07:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",976.18,695.91,161.85,59.07,59.35,689.1,976.18,495.4
2026-10-04T08:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",979.6,488.65,362.05,62.89,66.01,751.4,979.6,384.77
2026-10-04T09:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1004.14,472.32,464.07,6.93,60.82,698.5,1004.14,346.12
2026-10-04T10:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1006.16,303.66,594.84,44.45,63.21,712.2,1006.16,233.14
2026-10-04T11:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1018.38,292.54,651.73,15.55,58.55,709.2,1018.38,223.1
2026-10-04T12:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1046.46,160.01,762.32,65.62,58.52,693.2,1046.46,126.52
2026-10-04T13:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1024.66,169.22,710.68,87.21,57.55,721.9,1024.66,137.51
2026-10-04T14:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1029.26,151.79,763.57,54.36,59.54,709.4,1029.26,123.57
2026-10-04T15:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1033.19,290.54,647.78,38.53,56.34,691.9,1033.19,216.04
2026-10-04T16:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1040.43,363.02,567.64,50.12,59.66,714.1,1040.43,275.13
2026-10-04T17:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1048.15,586.07,359.39,39.36,63.33,716.1,1048.15,436.57
2026-10-04T18:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1044.08,779.59,184.55,24.03,55.91,733.4,1044.08,586.7
2026-10-04T19:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1037.58,933.3,0.0,39.88,64.4,694.3,1037.58,665.15
2026-10-04T20:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1001.31,901.65,0.0,39.67,60.0,707.7,1001.31,654.08
2026-10-04T21:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",1003.38,892.62,0.0,48.81,61.95,680.9,1003.38,624.3
2026-10-04T22:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",989.17,907.65,0.0,24.14,57.38,705.3,989.17,655.43
2026-10-04T23:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",988.98,882.73,0.0,47.68,58.56,705.3,988.98,638.26
2026-10-05T00:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",981.46,849.61,0.0,72.3,59.55,700.1,981.46,610.71
2026-10-05T01:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",963.2,856.01,0.0,50.43,56.76,711.3,963.2,624.01
2026-10-05T02:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",963.56,823.64,0.0,82.47,57.45,705.9,963.56,596.75
2026-10-05T03:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",945.95,829.27,0.0,53.05,63.63,699.6,945.95,597.17
2026-10-05T04:00:00,DC-01,data_centre,Nxtra by Airtel FY2025-26,"India (CEA, implied)",955.07,854.21,0.0,44.04,56.83,702.5,955.07,615.23
""",
    "ai_combined.csv": """timestamp,sensor_id,sector,model,grid_region,training_it_kw,inference_it_kw,it_power_kw,pue,facility_power_kw,grid_intensity_g_per_kwh,energy_kwh,co2_kg
2026-10-03T11:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,513.06,414.12,927.18,1.5,1390.77,1028.0,1390.77,1429.71
2026-10-03T12:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,499.8,453.33,953.12,1.5,1429.69,982.7,1429.69,1404.95
2026-10-03T13:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,509.89,441.22,951.11,1.5,1426.67,1002.6,1426.67,1430.38
2026-10-03T14:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,507.67,486.42,994.09,1.5,1491.14,991.8,1491.14,1478.91
2026-10-03T15:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,509.84,489.74,999.58,1.5,1499.37,1024.7,1499.37,1536.41
2026-10-03T16:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,502.65,497.35,1000.0,1.5,1500.0,1031.7,1500.0,1547.55
2026-10-03T17:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,501.07,469.58,970.65,1.5,1455.98,991.7,1455.98,1443.9
2026-10-03T18:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,496.72,449.56,946.29,1.5,1419.43,989.6,1419.43,1404.67
2026-10-03T19:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,495.37,414.9,910.27,1.5,1365.4,1008.0,1365.4,1376.33
2026-10-03T20:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,486.57,381.76,868.33,1.5,1302.5,995.3,1302.5,1296.38
2026-10-03T21:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,490.32,328.91,819.23,1.5,1228.84,1005.9,1228.84,1236.09
2026-10-03T22:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,479.98,300.15,780.12,1.5,1170.19,1001.9,1170.19,1172.41
2026-10-03T23:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,483.11,247.89,731.0,1.5,1096.51,1016.4,1096.51,1114.49
2026-10-04T00:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,482.03,221.62,703.65,1.5,1055.48,1017.7,1055.48,1074.16
2026-10-04T01:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,490.85,195.22,686.07,1.5,1029.1,1000.8,1029.1,1029.92
2026-10-04T02:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,479.57,188.29,667.86,1.5,1001.79,992.6,1001.79,994.37
2026-10-04T03:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,487.39,168.86,656.25,1.5,984.37,1031.8,984.37,1015.67
2026-10-04T04:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,486.92,180.19,667.11,1.5,1000.67,992.0,1000.67,992.66
2026-10-04T05:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,487.0,192.17,679.18,1.5,1018.77,1022.1,1018.77,1041.28
2026-10-04T06:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,498.58,212.74,711.32,1.5,1066.98,1020.0,1066.98,1088.32
2026-10-04T07:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,495.42,256.72,752.14,1.5,1128.21,1013.7,1128.21,1143.67
2026-10-04T08:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,502.91,283.75,786.65,1.5,1179.98,1013.4,1179.98,1195.79
2026-10-04T09:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,509.8,331.9,841.7,1.5,1262.55,1013.5,1262.55,1279.59
2026-10-04T10:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,511.21,381.68,892.89,1.5,1339.33,1010.4,1339.33,1353.26
2026-10-04T11:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,510.71,418.72,929.43,1.5,1394.15,1021.7,1394.15,1424.4
2026-10-04T12:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,502.32,447.19,949.51,1.5,1424.26,1002.5,1424.26,1427.82
2026-10-04T13:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,506.85,448.17,955.01,1.5,1432.52,996.1,1432.52,1426.93
2026-10-04T14:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,506.36,477.72,984.08,1.5,1476.12,1039.2,1476.12,1533.99
2026-10-04T15:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,501.02,485.94,986.96,1.5,1480.45,1028.5,1480.45,1522.64
2026-10-04T16:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,493.87,471.98,965.84,1.5,1448.77,1002.2,1448.77,1451.95
2026-10-04T17:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,499.55,444.99,944.54,1.5,1416.81,1020.8,1416.81,1446.28
2026-10-04T18:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,498.42,445.51,943.93,1.5,1415.89,1025.0,1415.89,1451.29
2026-10-04T19:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,489.35,419.12,908.47,1.5,1362.7,999.7,1362.7,1362.29
2026-10-04T20:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,491.66,368.06,859.72,1.5,1289.58,1005.1,1289.58,1296.16
2026-10-04T21:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,478.22,339.25,817.47,1.5,1226.21,986.1,1226.21,1209.16
2026-10-04T22:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,477.61,278.44,756.05,1.5,1134.08,1019.0,1134.08,1155.62
2026-10-04T23:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,481.24,244.19,725.43,1.5,1088.15,1022.9,1088.15,1113.07
2026-10-05T00:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,478.0,219.07,697.07,1.5,1045.61,1012.2,1045.61,1058.36
2026-10-05T01:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,483.57,190.36,673.93,1.5,1010.89,1002.8,1010.89,1013.72
2026-10-05T02:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,484.24,180.24,664.48,1.5,996.73,995.6,996.73,992.34
2026-10-05T03:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,482.64,173.85,656.49,1.5,984.73,1001.3,984.73,986.01
2026-10-05T04:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,481.64,174.99,656.64,1.5,984.96,1018.8,984.96,1003.48
2026-10-05T05:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,488.75,185.91,674.66,1.5,1011.99,1009.3,1011.99,1021.41
2026-10-05T06:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,497.28,215.07,712.34,1.5,1068.52,1018.0,1068.52,1087.75
2026-10-05T07:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,502.06,254.74,756.8,1.5,1135.2,1007.0,1135.2,1143.14
2026-10-05T08:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,500.19,295.31,795.5,1.5,1193.25,1007.4,1193.25,1202.08
2026-10-05T09:00:00,AI-01,ai,training + inference (GPT-6 Astra | camel-ai/CAMEL-13B-Combined-Data),Azure / South Africa West,496.87,326.54,823.41,1.5,1235.12,979.8,1235.12,1210.17
""",
    "infra.csv": """timestamp,sensor_id,sector,source_dataset,facility_id,city,facility_class,facility_power_kw,it_power_kw,cooling_kw,ups_loss_kw,misc_overhead_kw,pue,grid_kw,diesel_kw,grid_intensity_g_per_kwh,energy_kwh,co2_kg,grid_available,dg_running,outside_temp_c
2026-04-17T00:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,982.91,693.07,249.61,34.91,5.32,1.418,982.91,0.0,710.0,982.91,697.87,1,0,29.9
2026-04-17T01:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,982.21,693.07,247.93,34.91,6.3,1.417,982.21,0.0,710.0,982.21,697.37,1,0,28.7
2026-04-17T02:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,979.67,693.07,245.32,34.91,6.37,1.414,979.67,0.0,710.0,979.67,695.57,1,0,27.2
2026-04-17T03:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,979.34,693.07,245.76,34.91,5.6,1.413,979.34,0.0,710.0,979.34,695.33,1,0,28.6
2026-04-17T04:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,994.22,693.07,261.12,34.91,5.12,1.435,994.22,0.0,710.0,994.22,705.9,1,0,29.1
2026-04-17T05:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,983.33,693.07,249.6,34.91,5.76,1.419,983.33,0.0,710.0,983.33,698.17,1,0,29.3
2026-04-17T06:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,981.61,693.07,248.14,34.91,5.49,1.416,981.61,0.0,710.0,981.61,696.93,1,0,29.7
2026-04-17T07:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,992.22,693.07,257.53,34.91,6.71,1.432,992.22,0.0,710.0,992.22,704.47,1,0,31.8
2026-04-17T08:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1004.99,693.07,271.05,34.91,5.96,1.45,1004.99,0.0,710.0,1004.99,713.55,1,0,32.0
2026-04-17T09:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,996.39,693.07,263.13,34.91,5.28,1.438,996.39,0.0,710.0,996.39,707.42,1,0,33.4
2026-04-17T10:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1004.92,693.07,271.73,34.91,5.21,1.45,1004.92,0.0,710.0,1004.92,713.48,1,0,34.8
2026-04-17T11:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1011.65,693.07,279.89,34.91,3.79,1.46,1011.65,0.0,710.0,1011.65,718.28,1,0,35.7
2026-04-17T12:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1005.52,693.07,272.48,34.91,5.06,1.451,1005.52,0.0,710.0,1005.52,713.92,1,0,36.6
2026-04-17T13:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1003.8,693.07,270.96,34.91,4.85,1.448,1003.8,0.0,710.0,1003.8,712.7,1,0,36.2
2026-04-17T14:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1022.66,693.07,289.14,34.91,5.54,1.476,1022.66,0.0,710.0,1022.66,726.08,1,0,38.5
2026-04-17T15:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1017.23,693.07,283.21,34.91,6.04,1.468,1017.23,0.0,710.0,1017.23,722.24,1,0,38.1
2026-04-17T16:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1009.26,693.07,277.12,34.91,4.15,1.456,1009.26,0.0,710.0,1009.26,716.57,1,0,37.6
2026-04-17T17:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1007.94,693.07,274.25,34.91,5.71,1.454,1007.94,0.0,710.0,1007.94,715.63,1,0,36.7
2026-04-17T18:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1001.49,693.07,267.83,34.91,5.68,1.445,1001.49,0.0,710.0,1001.49,711.05,1,0,35.2
2026-04-17T19:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1007.34,693.07,274.42,34.91,4.94,1.453,1007.34,0.0,710.0,1007.34,715.2,1,0,35.6
2026-04-17T20:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1006.7,693.07,272.78,34.91,5.93,1.453,1006.7,0.0,710.0,1006.7,714.74,1,0,34.3
2026-04-17T21:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1004.44,693.07,270.16,34.91,6.3,1.449,1004.44,0.0,710.0,1004.44,713.16,1,0,33.8
2026-04-17T22:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,996.05,693.07,262.56,34.91,5.51,1.437,996.05,0.0,710.0,996.05,707.21,1,0,32.7
2026-04-17T23:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,993.67,693.07,260.27,34.91,5.41,1.434,993.67,0.0,710.0,993.67,705.51,1,0,31.3
2026-04-18T00:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,993.86,693.07,258.5,34.91,7.38,1.434,993.86,0.0,710.0,993.86,705.64,1,0,29.8
2026-04-18T01:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,998.39,693.07,262.96,34.91,7.45,1.441,998.39,0.0,710.0,998.39,708.86,1,0,30.2
2026-04-18T02:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,984.32,693.07,250.02,34.91,6.32,1.42,984.32,0.0,710.0,984.32,698.87,1,0,29.2
2026-04-18T03:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,984.27,693.07,250.57,34.91,5.72,1.42,984.27,0.0,710.0,984.27,698.85,1,0,28.6
2026-04-18T04:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,984.83,693.07,251.54,34.91,5.3,1.421,984.83,0.0,710.0,984.83,699.24,1,0,28.8
2026-04-18T05:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,990.89,693.07,255.91,34.91,7.0,1.43,990.89,0.0,710.0,990.89,703.54,1,0,29.0
2026-04-18T06:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,989.01,693.07,256.35,34.91,4.69,1.427,989.01,0.0,710.0,989.01,702.19,1,0,29.9
2026-04-18T07:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,995.12,693.07,260.66,34.91,6.48,1.436,995.12,0.0,710.0,995.12,706.53,1,0,31.0
2026-04-18T08:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,988.14,693.07,253.28,34.91,6.88,1.426,988.14,0.0,710.0,988.14,701.58,1,0,31.4
2026-04-18T09:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1006.81,693.07,272.5,34.91,6.33,1.453,1006.81,0.0,710.0,1006.81,714.83,1,0,33.9
2026-04-18T10:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1001.84,693.07,269.43,34.91,4.43,1.446,1001.84,0.0,710.0,1001.84,711.31,1,0,35.1
2026-04-18T11:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1005.64,693.07,272.25,34.91,5.42,1.451,1005.64,0.0,710.0,1005.64,714.0,1,0,34.8
2026-04-18T12:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1007.52,693.07,274.07,34.91,5.47,1.454,1007.52,0.0,710.0,1007.52,715.35,1,0,36.7
2026-04-18T13:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1012.04,693.07,279.41,34.91,4.66,1.46,1012.04,0.0,710.0,1012.04,718.54,1,0,36.8
2026-04-18T14:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1004.86,693.07,271.01,34.91,5.88,1.45,1004.86,0.0,710.0,1004.86,713.46,1,0,37.2
2026-04-18T15:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1012.82,693.07,279.48,34.91,5.35,1.461,1012.82,0.0,710.0,1012.82,719.11,1,0,38.9
2026-04-18T16:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1016.34,693.07,282.52,34.91,5.84,1.466,1016.34,0.0,710.0,1016.34,721.61,1,0,38.2
2026-04-18T17:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1021.15,693.07,286.77,34.91,6.4,1.473,1021.15,0.0,710.0,1021.15,725.02,1,0,37.7
2026-04-18T18:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1006.35,693.07,273.51,34.91,4.86,1.452,1006.35,0.0,710.0,1006.35,714.5,1,0,36.6
2026-04-18T19:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1002.36,693.07,269.32,34.91,5.06,1.446,1002.36,0.0,710.0,1002.36,711.68,1,0,35.6
2026-04-18T20:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,997.12,693.07,264.14,34.91,5.0,1.439,997.12,0.0,710.0,997.12,707.95,1,0,34.1
2026-04-18T21:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,999.84,693.07,265.37,34.91,6.49,1.443,999.84,0.0,710.0,999.84,709.88,1,0,34.5
2026-04-18T22:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1002.74,693.07,270.05,34.91,4.72,1.447,1002.74,0.0,710.0,1002.74,711.94,1,0,32.4
2026-04-18T23:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,993.89,693.07,259.9,34.91,6.0,1.434,993.89,0.0,710.0,993.89,705.67,1,0,31.9
2026-04-19T00:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,988.09,693.07,254.76,34.91,5.35,1.426,988.09,0.0,710.0,988.09,701.54,1,0,30.5
2026-04-19T01:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,986.04,693.07,252.64,34.91,5.41,1.423,986.04,0.0,710.0,986.04,700.08,1,0,29.2
2026-04-19T02:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,990.53,693.07,257.12,34.91,5.43,1.429,990.53,0.0,710.0,990.53,703.28,1,0,29.6
2026-04-19T03:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,982.81,693.07,249.06,34.91,5.77,1.418,982.81,0.0,710.0,982.81,697.78,1,0,28.8
2026-04-19T04:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,979.92,693.07,246.2,34.91,5.75,1.414,0.0,979.92,710.0,979.92,695.74,0,1,28.6
2026-04-19T05:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,992.58,693.07,258.04,34.91,6.56,1.432,0.0,992.58,710.0,992.58,704.73,0,1,30.3
2026-04-19T06:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,992.56,693.07,259.12,34.91,5.46,1.432,992.56,0.0,710.0,992.56,704.71,1,0,30.2
2026-04-19T07:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,989.39,693.07,255.21,34.91,6.2,1.428,989.39,0.0,710.0,989.39,702.47,1,0,31.2
2026-04-19T08:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,989.18,693.07,254.69,34.91,6.52,1.427,989.18,0.0,710.0,989.18,702.32,1,0,32.1
2026-04-19T09:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,995.38,693.07,262.73,34.91,4.66,1.436,995.38,0.0,710.0,995.38,706.71,1,0,33.5
2026-04-19T10:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1003.77,693.07,270.02,34.91,5.77,1.448,1003.77,0.0,710.0,1003.77,712.68,1,0,35.7
2026-04-19T11:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1008.54,693.07,273.35,34.91,7.21,1.455,1008.54,0.0,710.0,1008.54,716.07,1,0,35.6
2026-04-19T12:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1013.6,693.07,279.69,34.91,5.93,1.462,1013.6,0.0,710.0,1013.6,719.65,1,0,37.0
2026-04-19T13:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1024.01,693.07,290.9,34.91,5.13,1.477,1024.01,0.0,710.0,1024.01,727.04,1,0,37.8
2026-04-19T14:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1015.16,693.07,281.66,34.91,5.52,1.465,1015.16,0.0,710.0,1015.16,720.76,1,0,38.3
2026-04-19T15:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1020.34,693.07,286.7,34.91,5.67,1.472,1020.34,0.0,710.0,1020.34,724.45,1,0,38.6
2026-04-19T16:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1015.31,693.07,281.34,34.91,5.99,1.465,1015.31,0.0,710.0,1015.31,720.87,1,0,37.9
2026-04-19T17:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1014.01,693.07,281.16,34.91,4.88,1.463,1014.01,0.0,710.0,1014.01,719.96,1,0,38.1
2026-04-19T18:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1004.67,693.07,271.41,34.91,5.28,1.45,1004.67,0.0,710.0,1004.67,713.31,1,0,36.0
2026-04-19T19:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1012.43,693.07,278.77,34.91,5.68,1.461,1012.43,0.0,710.0,1012.43,718.83,1,0,36.1
2026-04-19T20:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1012.33,693.07,277.74,34.91,6.61,1.461,1012.33,0.0,710.0,1012.33,718.76,1,0,35.6
2026-04-19T21:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1000.37,693.07,267.07,34.91,5.32,1.443,1000.37,0.0,710.0,1000.37,710.27,1,0,33.8
2026-04-19T22:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1003.67,693.07,269.63,34.91,6.07,1.448,1003.67,0.0,710.0,1003.67,712.61,1,0,33.0
2026-04-19T23:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,995.31,693.07,260.66,34.91,6.67,1.436,995.31,0.0,710.0,995.31,706.66,1,0,31.9
2026-04-20T00:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,993.96,693.07,261.71,34.91,4.27,1.434,993.96,0.0,710.0,993.96,705.71,1,0,30.3
2026-04-20T01:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,995.84,693.07,263.11,34.91,4.75,1.437,995.84,0.0,710.0,995.84,707.06,1,0,30.9
2026-04-20T02:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,981.68,693.07,248.4,34.91,5.3,1.416,981.68,0.0,710.0,981.68,697.0,1,0,29.1
2026-04-20T03:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,997.7,693.07,264.72,34.91,5.0,1.44,997.7,0.0,710.0,997.7,708.36,1,0,28.8
2026-04-20T04:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,993.82,693.07,260.76,34.91,5.07,1.434,993.82,0.0,710.0,993.82,705.6,1,0,29.4
2026-04-20T05:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,981.24,693.07,247.88,34.91,5.37,1.416,981.24,0.0,710.0,981.24,696.67,1,0,29.2
2026-04-20T06:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,989.07,693.07,256.51,34.91,4.58,1.427,989.07,0.0,710.0,989.07,702.23,1,0,29.6
2026-04-20T07:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,988.27,693.07,253.44,34.91,6.85,1.426,988.27,0.0,710.0,988.27,701.67,1,0,30.9
2026-04-20T08:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,996.46,693.07,262.1,34.91,6.38,1.438,996.46,0.0,710.0,996.46,707.49,1,0,33.2
2026-04-20T09:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,996.46,693.07,262.66,34.91,5.82,1.438,996.46,0.0,710.0,996.46,707.49,1,0,32.9
2026-04-20T10:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1001.9,693.07,270.04,34.91,3.88,1.446,1001.9,0.0,710.0,1001.9,711.36,1,0,34.6
2026-04-20T11:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1006.08,693.07,272.34,34.91,5.76,1.452,1006.08,0.0,710.0,1006.08,714.31,1,0,36.1
2026-04-20T12:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1004.1,693.07,269.51,34.91,6.61,1.449,1004.1,0.0,710.0,1004.1,712.9,1,0,36.7
2026-04-20T13:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1009.98,693.07,276.26,34.91,5.74,1.457,1009.98,0.0,710.0,1009.98,717.09,1,0,37.4
2026-04-20T14:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1022.6,693.07,288.9,34.91,5.72,1.475,1022.6,0.0,710.0,1022.6,726.04,1,0,38.8
2026-04-20T15:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1008.15,693.07,274.52,34.91,5.65,1.455,1008.15,0.0,710.0,1008.15,715.79,1,0,38.1
2026-04-20T16:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1013.86,693.07,281.11,34.91,4.77,1.463,1013.86,0.0,710.0,1013.86,719.85,1,0,38.1
2026-04-20T17:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1016.02,693.07,282.02,34.91,6.02,1.466,1016.02,0.0,710.0,1016.02,721.37,1,0,37.6
2026-04-20T18:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1006.94,693.07,274.04,34.91,4.92,1.453,1006.94,0.0,710.0,1006.94,714.92,1,0,37.0
2026-04-20T19:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,1009.41,693.07,276.32,34.91,5.1,1.456,1009.41,0.0,710.0,1009.41,716.68,1,0,35.9
2026-04-20T20:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,997.25,693.07,263.75,34.91,5.52,1.439,0.0,997.25,710.0,997.25,708.05,0,1,34.4
2026-04-20T21:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,997.51,693.07,264.01,34.91,5.52,1.439,0.0,997.51,710.0,997.51,708.23,0,1,33.0
2026-04-20T22:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,993.51,693.07,258.38,34.91,7.15,1.433,0.0,993.51,710.0,993.51,705.38,0,1,33.5
2026-04-20T23:00:00,INFRA-01,infrastructure,india_dc synthetic telemetry,IN-DC-060,Kolkata,Hyperscale,998.5,693.07,264.31,34.91,6.22,1.441,0.0,998.5,710.0,998.5,708.94,0,1,31.8
""",
}


def _seed_sensor_files() -> None:
    for fname, text in _DEFAULT_SENSOR_CSV.items():
        target = SENSOR_DIR / fname
        if target.exists():
            continue
        for candidate in (Path(__file__).parent / fname, Path.cwd() / fname):
            if candidate.exists() and candidate.resolve() != target.resolve():
                target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
                break
        else:
            target.write_text(text, encoding="utf-8")


_seed_sensor_files()


_LOAD_COLS = ["facility_power_kw", "power_kw", "load_kw", "it_power_kw"]
_EF_COLS = ["grid_intensity_g_per_kwh", "grid_ef_g_per_kwh", "grid_intensity"]
_sensor_cache: dict = {}


def _normalise_sensor_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in raw.columns}
    load_col = next((cols[c] for c in _LOAD_COLS if c in cols), None)
    if load_col is None:
        raise ValueError("The CSV needs a power column named one of: " + ", ".join(_LOAD_COLS))

    df = pd.DataFrame()
    if "timestamp" in cols:
        df["timestamp"] = pd.to_datetime(raw[cols["timestamp"]], errors="coerce")
    else:
        base = datetime.now().replace(minute=0, second=0, microsecond=0)
        df["timestamp"] = [base + timedelta(hours=i) for i in range(len(raw))]
    df["load_kw"] = pd.to_numeric(raw[load_col], errors="coerce")

    ef_col = next((cols[c] for c in _EF_COLS if c in cols), None)
    if ef_col:
        df["grid_ef"] = pd.to_numeric(raw[ef_col], errors="coerce").fillna(GRID_EF_BASE_G_PER_KWH)
    else:
        df["grid_ef"] = GRID_EF_BASE_G_PER_KWH

    if "co2_kg" in cols:
        df["co2_kg"] = pd.to_numeric(raw[cols["co2_kg"]], errors="coerce")
    else:
        df["co2_kg"] = df["load_kw"] * df["grid_ef"] / 1000.0

    for src in ("grid", "solar", "wind", "diesel"):
        c = cols.get(f"{src}_kw")
        df[f"{src}_kw"] = pd.to_numeric(raw[c], errors="coerce") if c else np.nan

    df = df.dropna(subset=["timestamp", "load_kw", "co2_kg"])
    df = df[df["load_kw"] > 0].sort_values("timestamp").reset_index(drop=True)
    if len(df) < 6:
        raise ValueError("The CSV needs at least 6 valid hourly rows.")

    # Split recorded CO2 into the part that scales with grid intensity and the rest
    # (e.g. diesel), so the grid-intensity slider only moves the grid part.
    grid_co2 = np.where(df["grid_kw"].notna(), df["grid_kw"].fillna(0) * df["grid_ef"] / 1000.0, df["co2_kg"])
    df["grid_co2_kg"] = np.minimum(grid_co2, df["co2_kg"])
    df["other_co2_kg"] = (df["co2_kg"] - df["grid_co2_kg"]).clip(lower=0)
    return df


def load_sensor_df(sensor_id: str) -> pd.DataFrame:
    meta = SENSOR_REGISTRY.get(sensor_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Unknown sensor '{sensor_id}'.")
    path = SENSOR_DIR / meta["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"The {meta['label']} sensor has no data yet. Upload a CSV for it first.")
    mtime = path.stat().st_mtime
    cached = _sensor_cache.get(sensor_id)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        df = _normalise_sensor_df(pd.read_csv(path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _sensor_cache[sensor_id] = (mtime, df)
    return df


def sensor_meta(sensor_id: str) -> dict:
    meta = SENSOR_REGISTRY[sensor_id]
    out = {"id": sensor_id, "label": meta["label"], "domain": meta["domain"],
           "blurb": meta["blurb"], "file": meta["file"], "ready": False}
    if not (SENSOR_DIR / meta["file"]).exists():
        return out
    try:
        df = load_sensor_df(sensor_id)
    except HTTPException as e:
        out["error"] = e.detail
        return out
    out.update({
        "ready": True,
        "rows": int(len(df)),
        "mean_kw": round(float(df["load_kw"].mean()), 1),
        "peak_kw": round(float(df["load_kw"].max()), 1),
        "mean_ef": round(float(df["grid_ef"].mean()), 1),
        "recorded_co2_kg": round(float(df["co2_kg"].sum()), 1),
    })
    if df["solar_kw"].notna().any():
        out["solar_peak_kw"] = round(float(df["solar_kw"].max()), 1)
    if df["grid_kw"].notna().all():
        tot = sum(float(df[f"{k}_kw"].fillna(0).sum()) for k in ("grid", "solar", "wind", "diesel")) or 1.0
        out["mix_pct"] = {k: round(100 * float(df[f"{k}_kw"].fillna(0).sum()) / tot, 1)
                          for k in ("grid", "solar", "wind", "diesel")}
    return out


def sensor_horizon(req: PlanRequest, hours: int):
    """Slice the sensor to the planning horizon. The series is rotated so its
    hour-of-day lines up with the planner's timestamps (tariff bands, outage
    window and solar all depend on the clock hour), then the sensor's load /
    grid-intensity configuration is applied."""
    df = load_sensor_df(req.sensor_id)
    n = len(df)
    start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    timestamps = [start + timedelta(hours=h) for h in range(hours)]

    hod = df["timestamp"].dt.hour.to_numpy()
    matches = np.where(hod == start.hour)[0]
    i0 = int(matches[0]) if len(matches) else 0
    idx = []
    for k in range(hours):
        j = i0 + k
        if j >= n:
            j = (n - 24 + ((j - n) % 24)) if n >= 24 else (j % n)
        idx.append(j)
    seg = df.iloc[idx].reset_index(drop=True)

    base_load = seg["load_kw"].to_numpy()
    load = base_load * req.sensor_load_scale
    if req.sensor_noise_pct > 0:
        rng = np.random.default_rng(req.sensor_seed)
        load = load * (1 + rng.normal(0, req.sensor_noise_pct / 100.0, size=hours))
        load = np.clip(load, 0.05 * float(np.max(base_load)), None)
    ef = seg["grid_ef"].to_numpy() * req.sensor_grid_multiplier

    factor = load / base_load
    recorded = float(((seg["grid_co2_kg"].to_numpy() * req.sensor_grid_multiplier
                       + seg["other_co2_kg"].to_numpy()) * factor).sum())
    meta = SENSOR_REGISTRY[req.sensor_id]
    info = {
        "id": req.sensor_id,
        "label": meta["label"],
        "domain": meta["domain"],
        "rows_used": int(hours),
        "recorded_co2_kg": round(recorded, 1),
        "load_scale": req.sensor_load_scale,
        "grid_multiplier": req.sensor_grid_multiplier,
        "noise_pct": req.sensor_noise_pct,
    }
    return timestamps, load, ef, info


# ============================================================================
# MODEL 1 — LOAD FORECASTING
# Report spec: "Facebook Prophet or a simple SARIMA model ... both handle
# daily/weekly seasonality with minimal tuning." Implemented here as a
# Fourier-feature linear regression: same seasonality-capture idea, zero
# heavy dependencies, trains in milliseconds — a deliberate lightweight
# stand-in that is swappable for Prophet/SARIMA/LSTM later (Table 3,
# "stretch / future upgrade").
# ============================================================================

class LoadForecastModel:
    N_HARMONICS_DAILY = 3
    N_HARMONICS_WEEKLY = 2

    def __init__(self):
        self.model = LinearRegression()

    def _features(self, t: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(t, dtype=float)]
        for k in range(1, self.N_HARMONICS_DAILY + 1):
            cols.append(np.sin(2 * np.pi * k * t / 24))
            cols.append(np.cos(2 * np.pi * k * t / 24))
        for k in range(1, self.N_HARMONICS_WEEKLY + 1):
            cols.append(np.sin(2 * np.pi * k * t / (24 * 7)))
            cols.append(np.cos(2 * np.pi * k * t / (24 * 7)))
        return np.column_stack(cols)

    def fit_and_predict(self, capacity_kw: float, hours: int) -> pd.DataFrame:
        history = generate_synthetic_load_history(capacity_kw)
        t_hist = history["hour_index"].to_numpy()
        X_hist = self._features(t_hist)
        y_hist = history["load_kw"].to_numpy()
        self.model.fit(X_hist, y_hist)

        t_future = np.arange(t_hist[-1] + 1, t_hist[-1] + 1 + hours)
        X_future = self._features(t_future)
        forecast = self.model.predict(X_future)
        forecast = np.clip(forecast, 0.5 * capacity_kw, 1.1 * capacity_kw)

        start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        timestamps = [start + timedelta(hours=h) for h in range(hours)]
        return pd.DataFrame({"timestamp": timestamps, "load_kw": forecast})


# ============================================================================
# MODEL 2 — RENEWABLE (SOLAR) AVAILABILITY FORECASTING
# Report spec: "Regression model (linear regression or XGBoost) using
# weather-API features such as cloud cover and irradiance proxy."
# Implemented as a scikit-learn LinearRegression trained on a small
# physics-motivated synthetic dataset, then applied to real (or
# fallback-synthetic) weather features for the forecast window.
# ============================================================================

class SolarForecastModel:
    def __init__(self, panel_performance_ratio: float = 0.80):
        self.performance_ratio = panel_performance_ratio
        self.model = LinearRegression()
        self._train()

    def _train(self):
        rng = np.random.default_rng(11)
        n = 500
        ghi_norm = rng.uniform(0, 1, n)          # shortwave radiation / 1000 W/m^2
        cloud = rng.uniform(0, 1, n)              # cloud cover fraction
        output_fraction = np.clip(
            ghi_norm * (1 - 0.75 * cloud) + rng.normal(0, 0.02, n), 0, 1
        )
        X = np.column_stack([ghi_norm, cloud])
        self.model.fit(X, output_fraction)

    def predict(self, weather: pd.DataFrame, solar_capacity_kw: float) -> np.ndarray:
        ghi_norm = np.clip(weather["shortwave_radiation"].to_numpy() / 1000.0, 0, 1.3)
        cloud = np.clip(weather["cloud_cover_pct"].to_numpy() / 100.0, 0, 1)
        X = np.column_stack([ghi_norm, cloud])
        output_fraction = np.clip(self.model.predict(X), 0, 1)
        return output_fraction * solar_capacity_kw * self.performance_ratio


# ============================================================================
# MODEL 3 — SOURCE OPTIMISATION ENGINE
# Report spec: "Rule-based multi-objective optimisation using SciPy or
# PuLP (linear programming) over forecasted load, availability, and carbon
# intensity." Implemented as a genuine linear program (PuLP + CBC) coupling
# all hours of the horizon through the battery's state of charge, with a
# tunable carbon shadow-price so the same engine can be pushed from
# cost-optimal to carbon-optimal (the "carbon_priority" slider on the
# dashboard). Falls back to a simple greedy rule-based allocator if PuLP is
# unavailable, matching the report's named fallback approach.
# ============================================================================

def grid_carbon_intensity_series(timestamps: list[datetime]) -> np.ndarray:
    """More coal-heavy baseload at night, cleaner mix midday when solar
    feeds the wider grid — a simplified diurnal curve around the CEA
    baseline average."""
    hours = np.array([ts.hour for ts in timestamps])
    return GRID_EF_BASE_G_PER_KWH * (1 + 0.14 * np.cos(2 * np.pi * (hours - 3) / 24))


def grid_tariff_series(timestamps: list[datetime], base_price: float) -> np.ndarray:
    """Simple time-of-use tariff: peak / off-peak / normal bands."""
    hours = np.array([ts.hour for ts in timestamps])
    price = np.full(len(hours), base_price)
    peak = ((hours >= 9) & (hours < 12)) | ((hours >= 18) & (hours < 22))
    offpeak = (hours >= 22) | (hours < 6)
    price[peak] = base_price * 1.3
    price[offpeak] = base_price * 0.8
    return price


def _in_outage(hour: int, start: int, end: int) -> bool:
    """Outage window [start, end) in clock hours; wraps past midnight (e.g. 22 to 2)."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def optimize_sources(
    load_kw: np.ndarray,
    solar_avail_kw: np.ndarray,
    grid_ef: np.ndarray,
    grid_price: np.ndarray,
    timestamps: list[datetime],
    req: PlanRequest,
) -> tuple[pd.DataFrame, str]:
    H = len(load_kw)
    outage = np.zeros(H, dtype=bool)
    if req.simulate_outage:
        for i, ts in enumerate(timestamps):
            if _in_outage(ts.hour, req.outage_start_hour, req.outage_end_hour):
                outage[i] = True

    diesel_price_per_kwh = req.diesel_price_per_l / DIESEL_GENSET_KWH_PER_L
    diesel_ef_per_kwh = DIESEL_EF_KG_PER_L / DIESEL_GENSET_KWH_PER_L  # kg CO2 per kWh generated
    # Shadow carbon price: at priority=0 -> ~0 INR/kg (pure cost); at
    # priority=1 -> a strong ~35 INR/kg that dominates the objective.
    carbon_price_per_kg = 35.0 * (req.carbon_priority ** 1.5)

    battery_eff = 0.9
    soc_min = 0.10 * req.battery_capacity_kwh
    soc_max = 0.95 * req.battery_capacity_kwh
    soc_init = 0.5 * req.battery_capacity_kwh

    if PULP_AVAILABLE and req.battery_capacity_kwh >= 0:
        try:
            prob = pulp.LpProblem("dc_source_plan", pulp.LpMinimize)
            idx = range(H)
            grid = pulp.LpVariable.dicts("grid", idx, lowBound=0)
            solar = pulp.LpVariable.dicts("solar", idx, lowBound=0)
            batt_c = pulp.LpVariable.dicts("batt_charge", idx, lowBound=0, upBound=req.battery_max_rate_kw)
            batt_d = pulp.LpVariable.dicts("batt_discharge", idx, lowBound=0, upBound=req.battery_max_rate_kw)
            diesel = pulp.LpVariable.dicts("diesel", idx, lowBound=0, upBound=req.diesel_max_kw)
            unserved = pulp.LpVariable.dicts("unserved", idx, lowBound=0)
            soc = pulp.LpVariable.dicts("soc", range(H + 1), lowBound=soc_min, upBound=soc_max)

            prob += soc[0] == soc_init
            for h in idx:
                prob += solar[h] <= float(solar_avail_kw[h])
                if outage[h]:
                    prob += grid[h] == 0
                else:
                    prob += diesel[h] == 0  # diesel is backup-only, reserved for outage hours
                prob += (solar[h] + grid[h] + batt_d[h] + diesel[h] + unserved[h] - batt_c[h]) == float(load_kw[h])
                prob += soc[h + 1] == soc[h] + battery_eff * batt_c[h] - batt_d[h] * (1.0 / battery_eff)

            cost_term = pulp.lpSum(
                grid_price[h] * grid[h] + diesel_price_per_kwh * diesel[h] for h in idx
            )
            carbon_term = pulp.lpSum(
                carbon_price_per_kg * (
                    (grid_ef[h] / 1000.0) * grid[h] + diesel_ef_per_kwh * diesel[h]
                ) for h in idx
            )
            prob += cost_term + carbon_term + UNSERVED_PENALTY_INR_PER_KWH * pulp.lpSum(unserved[h] for h in idx)

            solver = pulp.PULP_CBC_CMD(msg=False)
            prob.solve(solver)

            if pulp.LpStatus[prob.status] == "Optimal":
                rows = []
                for h in idx:
                    rows.append({
                        "grid_kw": max(0.0, grid[h].value() or 0.0),
                        "solar_kw": max(0.0, solar[h].value() or 0.0),
                        "battery_discharge_kw": max(0.0, batt_d[h].value() or 0.0),
                        "battery_charge_kw": max(0.0, batt_c[h].value() or 0.0),
                        "diesel_kw": max(0.0, diesel[h].value() or 0.0),
                        "unserved_kw": max(0.0, unserved[h].value() or 0.0),
                        "battery_soc_kwh": max(0.0, soc[h + 1].value() or 0.0),
                        "grid_outage": bool(outage[h]),
                    })
                return pd.DataFrame(rows), "linear program (PuLP / CBC)"
        except Exception:
            pass  # fall through to greedy allocator

    return _greedy_allocate(load_kw, solar_avail_kw, grid_ef, grid_price, outage, req), "rule-based greedy allocator (fallback)"


def _greedy_allocate(load_kw, solar_avail_kw, grid_ef, grid_price, outage, req: PlanRequest) -> pd.DataFrame:
    """Fallback hour-by-hour rule-based allocator: solar first, then battery
    when it's cheap/clean to do so, then grid, then diesel only on outage."""
    H = len(load_kw)
    soc = 0.5 * req.battery_capacity_kwh
    soc_min = 0.10 * req.battery_capacity_kwh
    soc_max = 0.95 * req.battery_capacity_kwh
    battery_eff = 0.9
    price_median = float(np.median(grid_price))
    ef_median = float(np.median(grid_ef))
    rows = []
    for h in range(H):
        remaining = float(load_kw[h])
        solar_use = min(remaining, float(solar_avail_kw[h]))
        remaining -= solar_use
        batt_discharge = 0.0
        batt_charge = 0.0
        diesel_use = 0.0
        grid_use = 0.0

        expensive_or_dirty = (grid_price[h] > price_median) or (grid_ef[h] > ef_median)
        if remaining > 0 and not outage[h] and expensive_or_dirty and soc > soc_min:
            batt_discharge = min(remaining, req.battery_max_rate_kw, (soc - soc_min) * battery_eff)
            remaining -= batt_discharge
            soc -= batt_discharge / battery_eff

        unserved = 0.0
        if outage[h]:
            diesel_use = min(remaining, req.diesel_max_kw)
            remaining -= diesel_use
            unserved = max(0.0, remaining)
        else:
            grid_use = remaining
            remaining = 0.0

        surplus_solar = max(0.0, float(solar_avail_kw[h]) - solar_use)
        if surplus_solar > 0 and soc < soc_max:
            batt_charge = min(surplus_solar, req.battery_max_rate_kw, (soc_max - soc) / battery_eff)
            soc += batt_charge * battery_eff

        rows.append({
            "grid_kw": grid_use, "solar_kw": solar_use,
            "battery_discharge_kw": batt_discharge, "battery_charge_kw": batt_charge,
            "diesel_kw": diesel_use, "unserved_kw": unserved, "battery_soc_kwh": soc, "grid_outage": bool(outage[h]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# MODEL 4 — CARBON & WATER IMPACT CALCULATOR
# Report spec: deterministic calculation using published IPCC/CPCB emission
# factors and industry WUE benchmarks. Not ML — but essential to the
# pipeline, and reused to build a "business as usual" baseline for savings.
# ============================================================================

def compute_impact_row(row: pd.Series, grid_price: float, diesel_price_per_kwh: float) -> dict:
    co2_kg = (
        row["grid_kw"] * (row["grid_carbon_intensity_g_per_kwh"] / 1000.0)
        + row["diesel_kw"] * DIESEL_EF_KG_PER_L / DIESEL_GENSET_KWH_PER_L
    )
    water_l = (
        row["load_kw"] * COOLING_WUE_L_PER_KWH
        + row["grid_kw"] * GRID_UPSTREAM_WATER_L_PER_KWH
        + row["diesel_kw"] * DIESEL_WATER_L_PER_KWH
    )
    cost_inr = row["grid_kw"] * grid_price + row["diesel_kw"] * diesel_price_per_kwh
    return {"co2_kg": co2_kg, "water_l": water_l, "cost_inr": cost_inr}


def compute_baseline(load_kw: np.ndarray, grid_ef: np.ndarray, req: PlanRequest) -> dict:
    """100% grid, no solar/battery/optimisation — the counterfactual the
    dashboard's 'savings' figures are measured against."""
    flat_price = req.grid_price_per_kwh
    co2 = float(np.sum(load_kw * (grid_ef / 1000.0)))
    water = float(np.sum(load_kw * (COOLING_WUE_L_PER_KWH + GRID_UPSTREAM_WATER_L_PER_KWH)))
    cost = float(np.sum(load_kw * flat_price))
    return {"co2_kg": co2, "water_l": water, "cost_inr": cost}


# ============================================================================
# MODEL 5 — ESG NARRATIVE GENERATOR
# Report spec: "A structured prompt to an LLM API (e.g. Claude) that turns
# the computed figures into a plain-language ESG summary paragraph."
# Falls back to a deterministic template if no ANTHROPIC_API_KEY is set, so
# the demo still works offline / without credentials.
# ============================================================================

def generate_esg_narrative(plan: dict) -> ESGNarrativeResponse:
    totals = plan["totals"]
    savings = plan["savings"]
    mix = plan["mix_pct"]

    sensor_line = (f"Data source: {plan['sensor']['label']} sensor stream.\n" if plan.get("sensor") else "")
    prompt = (
        "You are drafting one paragraph for a data centre's ESG disclosure summary, "
        "aimed at non-technical stakeholders (board members, compliance officers). "
        "Use the figures below faithfully, plain language, no bullet points, "
        "120-170 words, confident but not promotional.\n\n"
        f"City: {plan['city']}. Planning horizon: {len(plan['hourly'])} hours.\n"
        f"{sensor_line}"
        f"Energy mix: grid {mix['grid_pct']:.1f}%, solar {mix['solar_pct']:.1f}%, "
        f"battery {mix['battery_pct']:.1f}%, diesel {mix['diesel_pct']:.1f}%.\n"
        f"Total CO2 emitted: {totals['co2_kg']:.1f} kg. "
        f"CO2 avoided vs. all-grid baseline: {savings['co2_kg']:.1f} kg "
        f"({savings['co2_pct']:.1f}% reduction).\n"
        f"Total water footprint: {totals['water_l']:.0f} L. "
        f"Water impact change vs. baseline: {savings['water_l']:.0f} L.\n"
        f"Total energy cost: Rs {totals['cost_inr']:.0f}. "
        f"Cost saved vs. baseline: Rs {savings['cost_inr']:.0f} "
        f"({savings['cost_pct']:.1f}% reduction)."
    )

    if ANTHROPIC_API_KEY and ANTHROPIC_SDK_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in resp.content if block.type == "text").strip()
            if text:
                return ESGNarrativeResponse(narrative=text, generated_with=ANTHROPIC_MODEL)
        except Exception:
            pass  # fall through to template

    template = (
        f"Over the {len(plan['hourly'])}-hour planning window for {plan['city']}, the platform's "
        f"optimiser drew {mix['solar_pct']:.0f}% of energy from solar, {mix['battery_pct']:.0f}% from "
        f"battery storage, {mix['grid_pct']:.0f}% from the grid, and {mix['diesel_pct']:.0f}% from diesel "
        f"backup. Compared with a business-as-usual, all-grid baseline, this switching plan avoided an "
        f"estimated {savings['co2_kg']:.1f} kg of CO2 ({savings['co2_pct']:.1f}% lower) and reduced total "
        f"energy cost by roughly Rs {savings['cost_inr']:.0f} ({savings['cost_pct']:.1f}%). Water footprint "
        f"moved by {savings['water_l']:.0f} L against baseline over the same period. These figures are "
        f"generated from forecast and simulated load data for demonstration purposes and should be "
        f"validated against metered site data before being used in a formal ESG disclosure."
    )
    return ESGNarrativeResponse(narrative=template, generated_with="template fallback (no ANTHROPIC_API_KEY set)")


# ============================================================================
# ORCHESTRATION PIPELINE  (Stages 1-3b-4 of the report's system workflow)
# ============================================================================

def run_pipeline(req: PlanRequest) -> dict:
    weather, weather_source = fetch_weather_forecast(req.lat, req.lon, req.hours)

    sensor_info = None
    if req.sensor_id:
        timestamps, load_kw, grid_ef, sensor_info = sensor_horizon(req, req.hours)
    else:
        load_model = LoadForecastModel()
        load_df = load_model.fit_and_predict(req.dc_capacity_kw, req.hours)
        timestamps = load_df["timestamp"].tolist()
        load_kw = load_df["load_kw"].to_numpy()
        grid_ef = grid_carbon_intensity_series(timestamps)

    solar_model = SolarForecastModel()
    solar_kw = solar_model.predict(weather.iloc[:req.hours], req.solar_capacity_kw)

    grid_price = grid_tariff_series(timestamps, req.grid_price_per_kwh)

    alloc_df, engine_name = optimize_sources(load_kw, solar_kw, grid_ef, grid_price, timestamps, req)

    hourly_records = []
    totals = {"co2_kg": 0.0, "water_l": 0.0, "cost_inr": 0.0,
              "grid_kwh": 0.0, "solar_kwh": 0.0, "battery_kwh": 0.0, "diesel_kwh": 0.0,
              "unserved_kwh": 0.0, "outage_hours": 0}
    diesel_price_per_kwh = req.diesel_price_per_l / DIESEL_GENSET_KWH_PER_L

    for h in range(req.hours):
        row = {
            "load_kw": load_kw[h],
            "grid_kw": alloc_df["grid_kw"].iloc[h],
            "solar_kw": alloc_df["solar_kw"].iloc[h],
            "diesel_kw": alloc_df["diesel_kw"].iloc[h],
            "grid_carbon_intensity_g_per_kwh": grid_ef[h],
        }
        impact = compute_impact_row(pd.Series(row), grid_price[h], diesel_price_per_kwh)
        rec = HourlyRecord(
            hour_index=h,
            timestamp=timestamps[h].isoformat(),
            load_kw=round(float(load_kw[h]), 2),
            solar_available_kw=round(float(solar_kw[h]), 2),
            grid_carbon_intensity_g_per_kwh=round(float(grid_ef[h]), 1),
            grid_price_per_kwh=round(float(grid_price[h]), 2),
            grid_kw=round(float(alloc_df["grid_kw"].iloc[h]), 2),
            solar_kw=round(float(alloc_df["solar_kw"].iloc[h]), 2),
            battery_discharge_kw=round(float(alloc_df["battery_discharge_kw"].iloc[h]), 2),
            battery_charge_kw=round(float(alloc_df["battery_charge_kw"].iloc[h]), 2),
            diesel_kw=round(float(alloc_df["diesel_kw"].iloc[h]), 2),
            battery_soc_kwh=round(float(alloc_df["battery_soc_kwh"].iloc[h]), 2),
            unserved_kw=round(float(alloc_df["unserved_kw"].iloc[h]), 2),
            cost_inr=round(float(impact["cost_inr"]), 2),
            co2_kg=round(float(impact["co2_kg"]), 3),
            water_l=round(float(impact["water_l"]), 2),
            grid_outage=bool(alloc_df["grid_outage"].iloc[h]),
        )
        hourly_records.append(rec)
        totals["co2_kg"] += impact["co2_kg"]
        totals["water_l"] += impact["water_l"]
        totals["cost_inr"] += impact["cost_inr"]
        totals["grid_kwh"] += alloc_df["grid_kw"].iloc[h]
        totals["solar_kwh"] += alloc_df["solar_kw"].iloc[h]
        totals["battery_kwh"] += alloc_df["battery_discharge_kw"].iloc[h]
        totals["diesel_kwh"] += alloc_df["diesel_kw"].iloc[h]
        totals["unserved_kwh"] += alloc_df["unserved_kw"].iloc[h]
        totals["outage_hours"] += int(bool(alloc_df["grid_outage"].iloc[h]))

    baseline_totals = compute_baseline(load_kw, grid_ef, req)

    def pct_saved(base, actual):
        return 0.0 if base == 0 else 100.0 * (base - actual) / base

    savings = {
        "co2_kg": baseline_totals["co2_kg"] - totals["co2_kg"],
        "co2_pct": pct_saved(baseline_totals["co2_kg"], totals["co2_kg"]),
        "water_l": baseline_totals["water_l"] - totals["water_l"],
        "water_pct": pct_saved(baseline_totals["water_l"], totals["water_l"]),
        "cost_inr": baseline_totals["cost_inr"] - totals["cost_inr"],
        "cost_pct": pct_saved(baseline_totals["cost_inr"], totals["cost_inr"]),
        "diesel_liters": totals["diesel_kwh"] / DIESEL_GENSET_KWH_PER_L,
    }

    total_energy = sum(totals[k] for k in ("grid_kwh", "solar_kwh", "battery_kwh", "diesel_kwh")) or 1.0
    mix_pct = {
        "grid_pct": 100 * totals["grid_kwh"] / total_energy,
        "solar_pct": 100 * totals["solar_kwh"] / total_energy,
        "battery_pct": 100 * totals["battery_kwh"] / total_energy,
        "diesel_pct": 100 * totals["diesel_kwh"] / total_energy,
    }

    plan = {
        "generated_at": datetime.now().isoformat(),
        "city": req.city,
        "lat": req.lat,
        "lon": req.lon,
        "weather_source": weather_source,
        "optimizer_engine": engine_name,
        "hourly": [r.model_dump() for r in hourly_records],
        "totals": totals,
        "baseline_totals": baseline_totals,
        "savings": savings,
        "mix_pct": mix_pct,
        "sensor": sensor_info,
    }
    store.save_plan(plan)
    return plan


# ============================================================================
# SCHEDULER  (Stage: hourly forecast-and-recommend cycle, per stack table)
# ============================================================================

scheduler = None
if APSCHEDULER_AVAILABLE:
    scheduler = BackgroundScheduler()

    def _scheduled_run():
        global _last_params
        if _last_params is not None:
            try:
                run_pipeline(_last_params)
            except Exception as e:
                print(f"[scheduler] pipeline run failed: {e}")

    scheduler.add_job(_scheduled_run, "interval", seconds=DEFAULT_INTERVAL_SECONDS, id="hourly_cycle")


# ============================================================================
# FASTAPI APP
# ============================================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    if scheduler is not None and not scheduler.running:
        scheduler.start()
    yield
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="CarbonPilot AI — Energy Source Switch Planner & Carbon-Water Impact Reporter",
    description="Hackathon MVP backend — Green AI Datacenters",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pulp_available": PULP_AVAILABLE,
        "reportlab_available": REPORTLAB_AVAILABLE,
        "anthropic_configured": bool(ANTHROPIC_API_KEY and ANTHROPIC_SDK_AVAILABLE),
        "scheduler_running": bool(scheduler and scheduler.running),
        "sensors_ready": [s for s in SENSOR_REGISTRY if (SENSOR_DIR / SENSOR_REGISTRY[s]["file"]).exists()],
    }


@app.get("/api/config/defaults")
def config_defaults():
    return PlanRequest().model_dump()


@app.get("/api/sensors")
def list_sensors():
    return [sensor_meta(sid) for sid in SENSOR_REGISTRY]


@app.get("/api/sensors/{sensor_id}/data")
def sensor_data(sensor_id: str, limit: int = 96):
    df = load_sensor_df(sensor_id)
    rows = [{
        "timestamp": r["timestamp"].isoformat(),
        "load_kw": round(float(r["load_kw"]), 2),
        "grid_ef": round(float(r["grid_ef"]), 1),
        "co2_kg": round(float(r["co2_kg"]), 2),
        "grid_co2_kg": round(float(r["grid_co2_kg"]), 2),
        "other_co2_kg": round(float(r["other_co2_kg"]), 2),
    } for _, r in df.head(max(1, min(limit, 500))).iterrows()]
    return {"sensor": sensor_meta(sensor_id), "rows": rows}


class SensorUpload(BaseModel):
    csv: str = Field(max_length=2_000_000)


@app.post("/api/sensors/{sensor_id}/upload")
def upload_sensor(sensor_id: str, body: SensorUpload):
    if sensor_id not in SENSOR_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown sensor '{sensor_id}'.")
    try:
        _normalise_sensor_df(pd.read_csv(io.StringIO(body.csv)))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read that file as a CSV.")
    (SENSOR_DIR / SENSOR_REGISTRY[sensor_id]["file"]).write_text(body.csv, encoding="utf-8")
    _sensor_cache.pop(sensor_id, None)
    return sensor_meta(sensor_id)


@app.post("/api/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest):
    global _last_params
    if req.sensor_id and req.sensor_id not in SENSOR_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown sensor '{req.sensor_id}'.")
    _last_params = req
    plan = run_pipeline(req)
    return plan


@app.get("/api/plan/latest", response_model=PlanResponse)
def latest_plan():
    plan = store.load_latest()
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan has been generated yet. POST /api/plan first.")
    return plan


@app.get("/api/history")
def history(limit: int = 30):
    return store.load_history(limit)


@app.post("/api/report/esg", response_model=ESGNarrativeResponse)
def esg_report():
    plan = store.load_latest()
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan has been generated yet. POST /api/plan first.")
    return generate_esg_narrative(plan)


@app.get("/api/report/pdf")
def esg_report_pdf():
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=500, detail="reportlab is not installed on the server.")
    plan = store.load_latest()
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan has been generated yet. POST /api/plan first.")
    narrative = generate_esg_narrative(plan)

    tmp_path = Path(tempfile.gettempdir()) / f"esg_report_{int(datetime.now().timestamp())}.pdf"
    _build_pdf(plan, narrative, tmp_path)
    return FileResponse(tmp_path, filename="esg_impact_report.pdf", media_type="application/pdf")


def _build_pdf(plan: dict, narrative: ESGNarrativeResponse, out_path: Path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=colors.HexColor("#1B3A2B"))
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], leading=15)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    elements = [
        Paragraph("ESG Impact Report", title_style),
        Paragraph("CarbonPilot AI — Energy Source Switch Planner &amp; Carbon-Water Impact Reporter", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph(f"Location: {plan['city']}  |  Generated: {plan['generated_at']}  |  "
                  f"Horizon: {len(plan['hourly'])}h  |  Engine: {plan['optimizer_engine']}"
                  + (f"  |  Sensor: {plan['sensor']['label']}" if plan.get("sensor") else ""), styles["Italic"]),
        Spacer(1, 6 * mm),
        Paragraph("Executive Summary", styles["Heading2"]),
        Paragraph(narrative.narrative, body_style),
        Spacer(1, 6 * mm),
        Paragraph("Key Figures", styles["Heading2"]),
    ]

    totals, baseline, savings, mix = plan["totals"], plan["baseline_totals"], plan["savings"], plan["mix_pct"]
    table_data = [
        ["Metric", "Optimised Plan", "All-Grid Baseline", "Savings"],
        ["CO2 emissions (kg)", f"{totals['co2_kg']:.1f}", f"{baseline['co2_kg']:.1f}", f"{savings['co2_kg']:.1f} ({savings['co2_pct']:.1f}%)"],
        ["Water footprint (L)", f"{totals['water_l']:.0f}", f"{baseline['water_l']:.0f}", f"{savings['water_l']:.0f} ({savings['water_pct']:.1f}%)"],
        ["Energy cost (Rs)", f"{totals['cost_inr']:.0f}", f"{baseline['cost_inr']:.0f}", f"{savings['cost_inr']:.0f} ({savings['cost_pct']:.1f}%)"],
        ["Diesel used (L)", f"{savings['diesel_liters']:.1f}", "-", "-"],
    ]
    tbl = Table(table_data, colWidths=[45 * mm, 38 * mm, 38 * mm, 45 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B3A2B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F3")]),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph("Energy Mix Over Planning Horizon", styles["Heading2"]))
    mix_data = [
        ["Grid", "Solar", "Battery", "Diesel"],
        [f"{mix['grid_pct']:.1f}%", f"{mix['solar_pct']:.1f}%", f"{mix['battery_pct']:.1f}%", f"{mix['diesel_pct']:.1f}%"],
    ]
    mix_tbl = Table(mix_data, colWidths=[41.5 * mm] * 4)
    mix_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0EA")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(mix_tbl)
    elements.append(Spacer(1, 8 * mm))
    elements.append(Paragraph(
        "Figures are generated from a forecast + simulation pipeline for demonstration purposes "
        "(hackathon MVP scope) and should be validated against metered site data before use in a "
        "formal ESG disclosure.", styles["Italic"]))

    doc.build(elements)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
