# CarbonPilot AI — Energy Source Switch Planner & Carbon-Water Impact Reporter

A working prototype of the hackathon MVP described in the technical report and TIH
proposal: an AI planner that recommends the lowest-cost/lowest-carbon energy source
(grid / solar / battery / diesel) for a data centre hour-by-hour, then turns that
plan into a carbon & water impact report.

```
dc-planner/
├── backend/
│   ├── app.py            <- single-file backend: all 5 models + API
│   ├── requirements.txt
│   └── data/             <- created automatically (local JSON "database")
├── frontend/
│   └── index.html        <- single-file UI (HTML + CSS + JS, no build step)
└── README.md
```

## 1. Run the backend

Requires Python 3.10+.

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API starts at **http://localhost:8000**. Interactive API docs (Swagger UI) are
available at http://localhost:8000/docs — useful for poking at the five models
directly.

### Optional: real ESG narratives via Claude

Without any setup, the ESG narrative endpoint uses a deterministic template so the
demo works fully offline. To have Claude write the narrative instead, set an API
key before starting the server:

```bash
cp .env.example .env
# edit .env and paste your key
export $(cat .env | xargs)   # or use python-dotenv / your shell's env loading
python app.py
```

`ANTHROPIC_MODEL` defaults to `claude-sonnet-5`; change it in `.env` if you want a
different model.

### Optional: live weather

The solar forecasting model calls [Open-Meteo](https://open-meteo.com) (free, no
API key) for real cloud cover / irradiance. If there's no network access, it falls
back automatically to a synthetic clear-sky model — the pipeline still runs end to
end either way. Check `weather_source` in the `/api/plan` response to see which one
was used.

## 2. Open the frontend

No build step — it's a single static HTML file.

```bash
cd frontend
python -m http.server 5500
# then open http://localhost:5500 in your browser
```

(or just double-click `index.html` to open it directly in a browser — CORS is
already enabled on the backend for this).

If your backend isn't running on `localhost:8000`, change the **Backend API base
URL** field under "Advanced" in the dashboard's config panel.

## 3. Using it

1. Pick a site, planning horizon, and drag the cost ↔ carbon priority slider.
2. Optionally simulate a grid outage — this is the only condition under which the
   optimiser will call on diesel backup.
3. Click **Run the planner**. This hits the real backend: it forecasts load and
   solar for the horizon, solves a linear program over the whole window (coupled
   through battery state of charge), and returns the hourly plan plus its carbon,
   water, and cost footprint versus an all-grid baseline.
4. Scroll to **ESG impact report** and click **Generate ESG narrative**, then
   **Download PDF report** for a shareable one-pager.

## Sensors (Tech, AI, Infrastructure)

The **Sensors** section of the page lets you pick one of three hourly sensor streams, configure it, simulate it live, and run the planner on it instead of the built-in load forecast.

| Sensor | Data file | Source |
|---|---|---|
| Tech | `data/sensors/tech_nxtra.csv` | Nxtra by Airtel FY 2025-26 mix and grid factor, scaled to 1 MW |
| AI | `data/sensors/ai_combined.csv` | Training + inference cluster, high-emission grid |
| Infrastructure | `data/sensors/infra.csv` | Synthetic India data-centre telemetry (Kolkata hyperscale site, 96 h with 6 grid-outage hours), built by `infra_sensor.py` |

- **Configure:** load scale, grid carbon intensity multiplier, sensor noise. "Size site assets to this sensor" fills solar, battery and diesel capacity from the sensor's peak load.
- **Simulate:** streams the readings live with noise; shows load, CO2 per hour and cumulative CO2.
- **Grid outage:** the toggle in the dashboard turns the grid off for the chosen hours (wrapping past midnight is supported, e.g. 22 to 2). Diesel is only allowed in those hours. Outage hours are outlined in the timeline, and any load the site cannot serve is reported as unserved instead of being hidden.
- **Run planner:** sends `sensor_id` plus the configuration to `POST /api/plan`. The sensor's readings replace the load forecast and the diurnal grid-intensity model. The result compares the plan against the all-grid baseline and the CO2 the sensor recorded.
- **CSV format for a new sensor:** one power column named `facility_power_kw`, `power_kw` or `load_kw`; optional `timestamp`, `grid_intensity_g_per_kwh`, `co2_kg`, `grid_kw`, `solar_kw`, `wind_kw`, `diesel_kw`. At least 6 hourly rows.
- **API:** `GET /api/sensors`, `GET /api/sensors/{id}/data`, `POST /api/sensors/{id}/upload` (JSON body `{"csv": "..."}`).

Note: `requirements.txt` pins `pulp<3.0`. In testing, PuLP 4.0 found no CBC solver, which makes the planner silently fall back to the greedy allocator.

## What's simulated vs. real

Per the report's own scoping (Section 7, Data Sources):

| Data | Status |
|---|---|
| Weather / solar irradiance | Real (Open-Meteo), synthetic fallback if offline |
| Data-centre load | Synthetic diurnal profile (real DC telemetry not available for a hackathon team — clearly labelled per the report) |
| Grid carbon intensity | Simplified diurnal model around the published CEA baseline average (swap in Electricity Maps / WattTime / Grid-India for production) |
| Emission & water factors | Published IPCC / CEA figures and industry WUE benchmarks |
| Source optimisation | A genuine linear program (PuLP + CBC), not a mock |

## Models implemented (see `app.py` section headers)

1. **Load forecasting** — Fourier-feature linear regression (daily + weekly
   seasonality), a lightweight stand-in for Prophet/SARIMA per the report's
   stretch-goal note.
2. **Solar forecasting** — regression on weather features (cloud cover,
   irradiance).
3. **Source optimisation** — linear program across the full horizon, coupling
   every hour through battery SOC; falls back to a rule-based greedy allocator if
   PuLP is unavailable.
4. **Carbon & water calculator** — deterministic, using published emission/water
   factors.
5. **ESG narrative generator** — Claude API call with a template fallback.

## Post-hackathon roadmap (not in this MVP, per the report's Section 9)

Cooling optimisation, full IoT/BMS integration, multi-site fleet dashboard, carbon
credit calculation, and a reinforcement-learning switching policy are deliberately
out of scope here.
