import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from feast import FeatureStore

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import shap
import matplotlib.pyplot as plt



ROOT = Path(__file__).resolve().parent

FEATURE_REPO = ROOT / "feature_repo"
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)



print("Connecting to Feast feature store...")

store = FeatureStore(repo_path=str(FEATURE_REPO))

print(" Connected (local SQLite — no cluster needed)")

parquet_path = FEATURE_REPO / "data" / "aqi_features.parquet"

df = pd.read_parquet(parquet_path)

print(
    f"\nData loaded: {df.shape[0]:,} rows × {df.shape[1]} cols"
)

BASE_FEATURES = [

    "temperature",
    "humidity",
    "precipitation",
    "cloud_cover",
    "wind_speed",
    "boundary",

    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",

    # Time
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",

    "sin_hour",
    "cos_hour",

    "sin_month",
    "cos_month",

    "sin_day",
    "cos_day",

    "sin_dow",
    "cos_dow",

    # Long-term lag features
    "pm25_lag24",
    "pm25_roll_mean_24",

    "aqi_lag24",
    "aqi_roll_mean_24",
]


FEATURES_24H = BASE_FEATURES + [

    "pm25_lag1",
    "pm25_roll_mean_6",

    "aqi_lag1",
    "aqi_change_rate",
    "aqi_roll_mean_6",
]


FEATURES_48H = BASE_FEATURES + [

    "pm25_roll_mean_6",
    "aqi_roll_mean_6",
]


FEATURES_72H = BASE_FEATURES


HORIZON_FEATURES = {
    "24h": FEATURES_24H,
    "48h": FEATURES_48H,
    "72h": FEATURES_72H
}


TARGETS = {
    "24h": "aqi_target_24h",
    "48h": "aqi_target_48h",
    "72h": "aqi_target_72h"
}



best_models = {}

for horizon, target_col in TARGETS.items():

    features = HORIZON_FEATURES[horizon]
    features = [f for f in features if f in df.columns]

    print("\n" + "=" * 52)
    print(f"  TARGET : {horizon} ({len(features)} features)")
    print("=" * 52)

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False
    )

    models = {
        "Ridge":
            Ridge(),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=100,
                random_state=42,
                verbosity=0
            ),

        "ExtraTrees":
            ExtraTreesRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMRegressor(
                n_estimators=100,
                random_state=42,
                verbose=-1
            )
    }

    best_r2 = -999999
    best_name = None
    best_model = None

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(y_test, preds)
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        r2 = r2_score(
            y_test,
            preds
        )

        print(
            f"  {model_name:<15}"
            f" RMSE={rmse:.2f}"
            f" MAE={mae:.2f}"
            f" R²={r2:.3f}"
        )

        if r2 > best_r2:
            best_r2 = r2
            best_name = model_name
            best_model = model

    print(
        f"\n  🏆 Winner: {best_name}"
        f" (R²={best_r2:.3f})"
    )

    best_models[horizon] = (
        best_name,
        best_model,
        features
    )



print("\n" + "=" * 52)
print("Saving models + generating SHAP plots...")
print("=" * 52)

for horizon, (model_name, model, features) in best_models.items():

    model_path = MODELS_DIR / f"best_model_{horizon}.pkl"

    joblib.dump(model, model_path)

    feature_path = MODELS_DIR / f"features_{horizon}.pkl"

    joblib.dump(features, feature_path)

    print(
        f"\nSaved: {model_path.name}"
        f" ({model_name})"
    )

    print(f"   Generating SHAP for {horizon}...")

    X = df[features]
    y = df[f"aqi_target_{horizon}"]

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False
    )

    try:

        if model_name == "Ridge":

            background = shap.sample(
                X_test,
                min(100, len(X_test)),
                random_state=42
            )

            explainer = shap.LinearExplainer(
                model,
                background
            )

            shap_values = explainer.shap_values(X_test)

        else:

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            if (
                hasattr(shap_values, "ndim")
                and shap_values.ndim == 3
            ):
                shap_values = shap_values[:, :, 0]

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=features,
            show=False
        )

        plt.title(
            f"SHAP Feature Importance — {horizon}"
        )

        shap_path = MODELS_DIR / f"shap_{horizon}.png"

        plt.savefig(
            shap_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f" Saved: {shap_path.name}"
        )

    except Exception as e:

        print(
            f"   ⚠️ SHAP failed ({e})"
        )

        continue

print(
    f"\nTraining complete! All files saved to:"
    f" {MODELS_DIR}"
)

print("\nSummary:")

for horizon, (model_name, _, _) in best_models.items():

    print(
        f"  {horizon}  →  {model_name}"
    )