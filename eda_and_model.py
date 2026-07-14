"""
eda_and_model.py
-----------------
End-to-end pipeline:
1. Load data
2. Clean missing values
3. Exploratory Data Analysis (EDA) with saved plots
4. Feature engineering
5. Train/test split
6. Train 3 models: Linear Regression, Random Forest, XGBoost
7. Evaluate with RMSE, MAE, R2
8. Save comparison plot + best model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("data/car_data.csv")
print("Shape:", df.shape)
print(df.isna().sum())

# ---------------------------------------------------------------------------
# 2. CLEAN MISSING VALUES
# ---------------------------------------------------------------------------
df["mileage_kmpl"] = df["mileage_kmpl"].fillna(df["mileage_kmpl"].median())
df["owner_count"] = df["owner_count"].fillna(df["owner_count"].mode()[0])
df["insurance_valid"] = df["insurance_valid"].fillna("No")

# ---------------------------------------------------------------------------
# 3. EDA — save each plot as a PNG for README / dashboard use
# ---------------------------------------------------------------------------

# 3a. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], bins=40, kde=True, color="#1f77b4")
plt.title("Distribution of Used Car Selling Price")
plt.xlabel("Selling Price (INR)")
plt.tight_layout()
plt.savefig("visuals/01_price_distribution.png")
plt.close()

# 3b. Price vs Car Age
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="car_age", y="selling_price", alpha=0.3, color="#d62728")
plt.title("Selling Price vs Car Age (Depreciation Effect)")
plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price (INR)")
plt.tight_layout()
plt.savefig("visuals/02_price_vs_age.png")
plt.close()

# 3c. Average Price by Brand
plt.figure(figsize=(9, 5))
brand_avg = df.groupby("brand")["selling_price"].mean().sort_values(ascending=False)
sns.barplot(x=brand_avg.values, y=brand_avg.index, palette="viridis")
plt.title("Average Selling Price by Brand")
plt.xlabel("Average Selling Price (INR)")
plt.tight_layout()
plt.savefig("visuals/03_avg_price_by_brand.png")
plt.close()

# 3d. Average Price by City
plt.figure(figsize=(9, 5))
city_avg = df.groupby("city")["selling_price"].mean().sort_values(ascending=False)
sns.barplot(x=city_avg.values, y=city_avg.index, palette="mako")
plt.title("Average Selling Price by City")
plt.xlabel("Average Selling Price (INR)")
plt.tight_layout()
plt.savefig("visuals/04_avg_price_by_city.png")
plt.close()

# 3e. Correlation heatmap (numeric features)
plt.figure(figsize=(8, 6))
numeric_cols = ["car_age", "km_driven", "owner_count", "mileage_kmpl", "engine_cc", "selling_price"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("visuals/05_correlation_heatmap.png")
plt.close()

# 3f. Price by Fuel Type & Transmission
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="fuel_type", y="selling_price", hue="transmission")
plt.title("Selling Price by Fuel Type & Transmission")
plt.tight_layout()
plt.savefig("visuals/06_price_by_fuel_transmission.png")
plt.close()

print("EDA plots saved to visuals/")

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["price_per_cc"] = df["selling_price"] / df["engine_cc"]  # for reference/analysis only, not used as model input
df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)

feature_cols = ["brand", "city", "car_age", "km_driven", "fuel_type",
                 "transmission", "owner_count", "mileage_kmpl", "engine_cc",
                 "insurance_valid", "km_per_year"]
target_col = "selling_price"

X = df[feature_cols]
y = df[target_col]

categorical_features = ["brand", "city", "fuel_type", "transmission", "insurance_valid"]
numeric_features = ["car_age", "km_driven", "owner_count", "mileage_kmpl", "engine_cc", "km_per_year"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

# ---------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 6. TRAIN MULTIPLE MODELS
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1),
}

results = {}
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}
    trained_pipelines[name] = pipe
    print(f"{name:20s} | RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | R2: {r2:.4f}")

# ---------------------------------------------------------------------------
# 7. SAVE RESULTS + COMPARISON PLOT
# ---------------------------------------------------------------------------
with open("visuals/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

results_df = pd.DataFrame(results).T
plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df["R2"], palette="crest")
plt.title("Model Comparison — R² Score (higher is better)")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("visuals/07_model_comparison_r2.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df["RMSE"], palette="flare")
plt.title("Model Comparison — RMSE (lower is better)")
plt.ylabel("RMSE (INR)")
plt.tight_layout()
plt.savefig("visuals/08_model_comparison_rmse.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. SAVE BEST MODEL (based on R2)
# ---------------------------------------------------------------------------
best_model_name = results_df["R2"].idxmax()
best_pipeline = trained_pipelines[best_model_name]
joblib.dump(best_pipeline, "src/best_model.pkl")
print(f"\nBest model: {best_model_name} -> saved to src/best_model.pkl")

# Feature importance plot (for tree-based best model)
if best_model_name in ["Random Forest", "XGBoost"]:
    ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = cat_feature_names + numeric_features

    importances = best_pipeline.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top 15 Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.savefig("visuals/09_feature_importance.png")
    plt.close()

print("\nAll done. Check visuals/ folder for charts and src/best_model.pkl for the trained model.")
