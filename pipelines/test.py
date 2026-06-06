"""
test.py
───────
Fix #9 — feature view names now match features.py: aqi_raw, aqi_time, aqi_lag, aqi_targets
"""

import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from feast import FeatureStore

ROOT         = Path(__file__).resolve().parent
FEATURE_REPO = ROOT / "feature_repo"
PARQUET      = FEATURE_REPO / "data" / "aqi_features.parquet"

passed = 0
failed = 0

def ok(msg):
    global passed; passed += 1
    print(f"  ✅ {msg}")

def fail(msg):
    global failed; failed += 1
    print(f"  ❌ {msg}")

# ── 1. Connect ────────────────────────────────────────────────────────────────
print("=" * 52)
print("1. Connecting to Feast...")
print("=" * 52)
try:
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    ok("Connected (local SQLite)")
except Exception as e:
    fail(f"Connection failed: {e}")
    sys.exit(1)

# ── 2. Parquet exists ─────────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("2. Checking offline data (parquet)...")
print("=" * 52)
if not PARQUET.exists():
    fail("Parquet not found — run feature_store_pipeline.py first")
    sys.exit(1)

df = pd.read_parquet(PARQUET)
ok(f"Rows: {len(df):,}  Cols: {len(df.columns)}")
ok(f"Range: {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")

# Check wind_speed + boundary are present (fix #4 / #5)
for col in ["wind_speed", "boundary"]:
    if col in df.columns:
        ok(f"Column '{col}' present ✓")
    else:
        fail(f"Column '{col}' missing — re-run feature_store_pipeline.py")

print(df.head(2).to_string())

# ── 3. Online store lookup ────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("3. Online store lookup (~1-5ms expected)...")
print("=" * 52)
try:
    t0 = time.time()
    result = store.get_online_features(
        features=[
            "aqi_raw:pm25",
            "aqi_raw:pm10",
            "aqi_raw:aqi",
            "aqi_raw:no2",
            "aqi_raw:wind_speed",    # fix #5 — now in schema
            "aqi_raw:boundary",      # fix #5 — now in schema
            "aqi_lag:aqi_lag1",
            "aqi_lag:pm25_roll_mean_24",
            "aqi_time:hour",
            "aqi_time:sin_hour",
        ],
        entity_rows=[{"station_id": "lahore_main"}],
    ).to_dict()
    latency_ms = (time.time() - t0) * 1000
    ok(f"Online fetch in {latency_ms:.1f}ms")
    for k, v in result.items():
        print(f"     {k:<35} {v[0]}")
except Exception as e:
    fail(f"Online store error (run materialize first): {e}")

# ── 4. Historical feature pull ────────────────────────────────────────────────
print("\n" + "=" * 52)
print("4. Historical feature pull (training simulation)...")
print("=" * 52)
try:
    entity_df = pd.DataFrame({
        "station_id":      ["lahore_main"] * 5,
        "event_timestamp": pd.date_range(
            end=datetime.now(timezone.utc),
            periods=5,
            freq="1h"
        )
    })

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "aqi_raw:pm25", "aqi_raw:aqi",
            "aqi_raw:wind_speed", "aqi_raw:boundary",
            "aqi_lag:aqi_lag24", "aqi_lag:pm25_roll_mean_24",
            "aqi_targets:aqi_target_24h",
        ],
    ).to_df()

    ok(f"Historical pull: {len(training_df)} rows")
    print(training_df.to_string())
except Exception as e:
    fail(f"Historical pull: {e}")

# ── 5. Models exist ───────────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("5. Checking saved models...")
print("=" * 52)
MODELS_DIR = ROOT / "models"
for horizon in ["24h", "48h", "72h"]:
    model_path = MODELS_DIR / f"best_model_{horizon}.pkl"
    feat_path  = MODELS_DIR / f"features_{horizon}.pkl"
    if model_path.exists() and feat_path.exists():
        ok(f"best_model_{horizon}.pkl + features_{horizon}.pkl ✓")
    elif model_path.exists():
        fail(f"best_model_{horizon}.pkl exists but features_{horizon}.pkl missing — retrain")
    else:
        fail(f"best_model_{horizon}.pkl missing — run model_training.py")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 52)
print(f"  PASSED: {passed}   FAILED: {failed}")
print("=" * 52)
if failed == 0:
    print("✅ All checks passed — pipeline is healthy!")
else:
    print("⚠️  Some checks failed — see above.")
