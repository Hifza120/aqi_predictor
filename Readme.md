# 🌫️ Lahore AQI Predictor

An end-to-end ML system that predicts the Air Quality Index (AQI) for Lahore, Pakistan over the next 24, 48, and 72 hours — using real-time weather and pollution data, a feature store, and an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```
Weather & Pollution APIs
        │
        ▼
┌─────────────────────┐
│  Feature Pipeline   │  ← fetch_data.py / historic_backfill.py
│  feature_engineering│
└────────┬────────────┘
         │ features
         ▼
┌─────────────────────┐
│   Feast Feature     │  ← feature_store_pipeline.py
│      Store          │  (SQLite — offline + online)
└────────┬────────────┘
         │ features + targets
         ▼
┌─────────────────────┐
│  Training Pipeline  │  ← model_training.py
│  RandomForest       │
│  XGBoost / LightGBM │
│  Ridge / ExtraTrees │
└────────┬────────────┘
         │ best model per horizon
         ▼
┌─────────────────────┐
│  Streamlit Dashboard│  ← app/dashboard.py
│  Live AQI + Forecast│
│  EDA + SHAP plots   │
└─────────────────────┘
```

---

## 📁 Project Structure

```
aqi_predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     ← runs every hour
│       └── training_pipeline.yml    ← runs every day at 02:00 UTC
├── app/
│   └── dashboard.py                 ← Streamlit dashboard
├── pipelines/
│   ├── fetch_data.py                ← live AQI + weather fetch → Feast
│   ├── historic_backfill.py         ← historical weather (Open-Meteo archive)
│   ├── historic_airquality.py       ← historical AQ (Open-Meteo AQ API)
│   ├── feature_engineering.py       ← merges + engineers all features
│   ├── feature_store_pipeline.py    ← Feast setup + materialize
│   ├── model_training.py            ← trains + evaluates + saves models
│   ├── predict.py                   ← runs 24h/48h/72h forecasts
│   ├── eda.py                       ← generates 10 EDA plots
│   ├── test.py                      ← pipeline health checks
│   ├── features.py                  ← Feast schema definitions
│   ├── feature_repo/                ← Feast registry + SQLite stores
│   └── models/                      ← saved .pkl models + SHAP plots
├── requirements.txt
└── README.md
```

---

## 🔌 Data Sources

| Source | What it provides |
|--------|-----------------|
| [AQICN API](https://aqicn.org/api/) | Real-time AQI, PM2.5, PM10, NO₂ |
| [Open-Meteo Forecast](https://open-meteo.com/) | Live weather (temperature, humidity, wind, etc.) |
| [Open-Meteo Air Quality](https://air-quality-api.open-meteo.com/) | Live + historical pollutants |
| [Open-Meteo Archive](https://archive-api.open-meteo.com/) | Historical weather (2022–2026) |

---

## ⚙️ Engineered Features

| Category | Features |
|----------|----------|
| Pollutants | PM2.5, PM10, NO₂, SO₂, CO, O₃ |
| Weather | Temperature, Humidity, Wind Speed, Precipitation, Cloud Cover, Boundary Layer |
| Time (cyclical) | sin/cos of hour, day, month, day-of-week |
| Lag features | PM2.5 lag 1h/24h, AQI lag 1h/24h |
| Rolling features | PM2.5 rolling mean 6h/24h, AQI rolling mean 6h/24h |
| Targets | AQI at +24h, +48h, +72h |

---

## 🤖 Models

Five models trained and compared per horizon:

- Ridge Regression
- Random Forest
- XGBoost
- Extra Trees
- LightGBM

Best model per horizon selected by R² score and saved to `pipelines/models/`.

| Horizon | Winner | Features used |
|---------|--------|--------------|
| 24h | RandomForest | All features incl. short-term lags |
| 48h | RandomForest | Base + 6h rolling (no 1h lags) |
| 72h | XGBoost | Base features only (24h+ lags) |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup `.env`
```bash
cp pipelines/.env.template pipelines/.env
# Add your AQICN_TOKEN
```

### 3. Run feature store pipeline (first time setup)
```bash
cd pipelines
python feature_store_pipeline.py
```

### 4. Train models
```bash
python model_training.py
```

### 5. Run dashboard
```bash
streamlit run app/dashboard.py
```

### 6. Fetch live data (push to online store)
```bash
python fetch_data.py
```

### 7. Run pipeline health check
```bash
python test.py
```

---

## 📊 Dashboard Pages

| Page | Content |
|------|---------|
| 🏠 Live Dashboard | Current AQI, 3-day forecast, 7-day history, hazard alerts |
| 📈 Historical EDA | Distribution, seasonal patterns, pollutant trends, weather correlations |
| 🔬 Model Insights | SHAP feature importance, correlation matrix, model summary |
| ⚙️ Data Explorer | Filter by date/AQI, interactive charts, CSV download |

---

## ⚡ Automated Pipelines (GitHub Actions)

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| `feature_pipeline.yml` | Every hour | Fetches live data → pushes to Feast online store |
| `training_pipeline.yml` | Daily 02:00 UTC | Backfills features → retrains all models → saves new pkls |

### Required GitHub Secret
```
AQICN_TOKEN = your_token_here
```
> Settings → Secrets and variables → Actions → New repository secret

---

## 📍 Station Info

- **Location:** Lahore, Punjab, Pakistan
- **Coordinates:** 31.55°N, 74.34°E
- **Station ID:** `lahore_main`
- **Data range:** June 2022 – present

---

## 🛠️ Tech Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `LightGBM` `SHAP` `Feast` `Streamlit` `Plotly` `GitHub Actions`