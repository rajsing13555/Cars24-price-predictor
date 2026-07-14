"""
train_model.py
----------------
Advanced modeling pipeline (v2):
1. Load + clean sold-listings data
2. Feature engineering
3. Train baseline models (Linear Regression, Random Forest, XGBoost)
4. Hyperparameter-tune the best baseline with RandomizedSearchCV
5. Evaluate tuned model
6. SHAP explainability — global feature importance + a single-prediction
   waterfall explanation (the kind of artifact you'd actually show an
   HR / hiring-manager round: "here's WHY the model priced this car this way")
7. Save the final tuned pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import joblib
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------
df = pd.read_csv("data/car_listings.csv")
df = df[df["status"] == "Sold"].copy()  # only sold listings have a real transacted price

df["mileage_kmpl"] = df["mileage_kmpl"].fillna(df["mileage_kmpl"].median())
df["owner_count"] = df["owner_count"].fillna(df["owner_count"].mode()[0])
df["insurance_valid"] = df["insurance_valid"].fillna("No")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)

feature_cols = ["brand", "city", "car_age", "km_driven", "fuel_type", "transmission",
                 "owner_count", "mileage_kmpl", "engine_cc", "insurance_valid", "km_per_year"]
target_col = "sold_price"  # NOTE: predicting actual transacted price, not just asking price

X = df[feature_cols]
y = df[target_col]

categorical_features = ["brand", "city", "fuel_type", "transmission", "insurance_valid"]
numeric_features = ["car_age", "km_driven", "owner_count", "mileage_kmpl", "engine_cc", "km_per_year"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 3. BASELINE MODELS
# ---------------------------------------------------------------------------
baseline_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost (default)": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1),
}

results = {}
for name, model in baseline_models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "MAE": round(mean_absolute_error(y_test, preds), 2),
        "R2": round(r2_score(y_test, preds), 4),
    }
    print(f"{name:22s} | RMSE: {results[name]['RMSE']:>10,.0f} | R2: {results[name]['R2']:.4f}")

# ---------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (RandomizedSearchCV on XGBoost)
# ---------------------------------------------------------------------------
print("\nRunning RandomizedSearchCV on XGBoost (this may take a minute)...")

param_dist = {
    "model__n_estimators": [200, 300, 400, 600],
    "model__max_depth": [4, 5, 6, 8, 10],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
}

tuning_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(random_state=42, n_jobs=-1)),
])

search = RandomizedSearchCV(
    tuning_pipe, param_distributions=param_dist, n_iter=25, cv=4,
    scoring="r2", random_state=42, n_jobs=-1, verbose=0,
)
search.fit(X_train, y_train)

best_pipe = search.best_estimator_
preds = best_pipe.predict(X_test)

tuned_results = {
    "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
    "MAE": round(mean_absolute_error(y_test, preds), 2),
    "R2": round(r2_score(y_test, preds), 4),
}
results["XGBoost (tuned)"] = tuned_results

print(f"\nBest params: {search.best_params_}")
print(f"{'XGBoost (tuned)':22s} | RMSE: {tuned_results['RMSE']:>10,.0f} | R2: {tuned_results['R2']:.4f}")

with open("visuals/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("src/best_params.json", "w") as f:
    json.dump(search.best_params_, f, indent=2)

# ---------------------------------------------------------------------------
# 5. MODEL COMPARISON PLOT
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).T.astype(float)
plt.figure(figsize=(9, 5))
colors = ["#7F77DD" if "tuned" not in k else "#1D9E75" for k in results_df.index]
plt.bar(results_df.index, results_df["R2"], color=colors)
plt.title("Model Comparison — R² Score (tuned XGBoost highlighted)")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("visuals/07_model_comparison_r2.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. SHAP EXPLAINABILITY
# ---------------------------------------------------------------------------
print("\nComputing SHAP values...")

# transform a sample of test data for SHAP (raw XGBoost needs transformed input)
X_test_transformed = best_pipe.named_steps["preprocessor"].transform(X_test)
ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = cat_names + numeric_features

explainer = shap.TreeExplainer(best_pipe.named_steps["model"])
sample_idx = np.random.RandomState(42).choice(X_test_transformed.shape[0], size=min(500, X_test_transformed.shape[0]), replace=False)
X_sample = X_test_transformed[sample_idx]
if hasattr(X_sample, "toarray"):
    X_sample = X_sample.toarray()

shap_values = explainer.shap_values(X_sample)

# Global feature importance (SHAP summary as bar chart)
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, plot_type="bar", show=False, max_display=12)
plt.title("SHAP Global Feature Importance — What Drives Price Predictions")
plt.tight_layout()
plt.savefig("visuals/10_shap_feature_importance.png")
plt.close()

# SHAP beeswarm (shows direction of effect, not just magnitude)
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, show=False, max_display=12)
plt.title("SHAP Summary — Direction & Magnitude of Feature Effects")
plt.tight_layout()
plt.savefig("visuals/11_shap_beeswarm.png")
plt.close()

# Single-prediction waterfall — explain ONE car's predicted price
single_explainer = shap.Explainer(best_pipe.named_steps["model"], feature_names=all_feature_names)
single_shap = single_explainer(X_sample[:1])
plt.figure(figsize=(9, 6))
shap.plots.waterfall(single_shap[0], show=False, max_display=12)
plt.tight_layout()
plt.savefig("visuals/12_shap_single_prediction_waterfall.png")
plt.close()

print("SHAP plots saved to visuals/")

# ---------------------------------------------------------------------------
# 7. SAVE FINAL MODEL
# ---------------------------------------------------------------------------
joblib.dump(best_pipe, "src/best_model.pkl")
print("\nFinal tuned model saved to src/best_model.pkl")
print(f"\nFINAL RESULT: R² = {tuned_results['R2']:.4f}  |  RMSE = Rs.{tuned_results['RMSE']:,.0f}  |  MAE = Rs.{tuned_results['MAE']:,.0f}")
