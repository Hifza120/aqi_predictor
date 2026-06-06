
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

THIS_DIR     = Path(__file__).resolve().parent
FEATURE_REPO = THIS_DIR / "feature_repo"
DATA_DIR     = FEATURE_REPO / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(FEATURE_REPO))  

import features as feat_module
from feast import FeatureStore

print("=" * 55)
print("STEP 1: Running feature engineering...")
print("=" * 55)

import feature_engineering
df_features = feature_engineering.build()

parquet_path = DATA_DIR / "aqi_features.parquet"
df_features.to_parquet(parquet_path, index=False)
print(f" Saved {len(df_features):,} rows → {parquet_path}")

print("\n" + "=" * 55)
print("STEP 2: Setting up feature_repo...")
print("=" * 55)

yaml_path = FEATURE_REPO / "feature_store.yaml"
yaml_path.write_text("""project: aqi_lahore
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
""")
print("feature_store.yaml written")

features_src = THIS_DIR / "features.py"
features_dst = FEATURE_REPO / "features.py"

if features_src.exists():
    shutil.copy(features_src, features_dst)
    print("features.py copied to feature_repo/")
elif features_dst.exists():
    print("features.py already in feature_repo/ — skipping copy")
else:
    print("features.py not found in pipelines/ or feature_repo/")
    sys.exit(1)

print("\n" + "=" * 55)
print("STEP 3: Registering features (feast apply)...")
print("=" * 55)

store = FeatureStore(repo_path=str(FEATURE_REPO))

store.apply([
    feat_module.station,
    feat_module.aqi_file_source,
    feat_module.aqi_push_source,
    feat_module.raw_features,
    feat_module.time_features,
    feat_module.lag_features,
    feat_module.target_features,
])
print(" feast apply done")

print("\n" + "=" * 55)
print("STEP 4: Materializing to SQLite online store...")
print("=" * 55)

end_dt   = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=4 * 365)

print(f"Materializing {start_dt.date()} → {end_dt.date()} ...")
store.materialize(start_date=start_dt, end_date=end_dt)

print("\n Feature store pipeline complete!")
print(f"   Parquet  : {parquet_path}")
print(f"   Registry : {FEATURE_REPO / 'data' / 'registry.db'}")
print(f"   Online DB: {FEATURE_REPO / 'data' / 'online_store.db'}")