"""
test_pipeline.py
------------------
Basic QA/testing layer — the kind of checks a real ML pipeline needs
before it's trusted in production. Run with: pytest tests/ -v
"""

import pandas as pd
import joblib
import pytest
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data", "car_listings.csv"))


@pytest.fixture(scope="module")
def model():
    return joblib.load(os.path.join(BASE_DIR, "src", "best_model.pkl"))


# ---------------------------------------------------------------------------
# DATA QUALITY TESTS
# ---------------------------------------------------------------------------

def test_data_not_empty(raw_data):
    assert len(raw_data) > 0


def test_no_negative_prices(raw_data):
    assert (raw_data["asking_price"] > 0).all()


def test_sold_price_not_greater_than_asking(raw_data):
    sold = raw_data.dropna(subset=["sold_price"])
    assert (sold["sold_price"] <= sold["asking_price"]).all(), \
        "Found sold_price greater than asking_price — pricing logic bug"


def test_car_age_is_reasonable(raw_data):
    assert raw_data["car_age"].between(0, 30).all()


def test_missing_value_rate_within_expected_bounds(raw_data):
    # we intentionally inject ~2% missingness; fail if it balloons unexpectedly
    for col in ["mileage_kmpl", "owner_count", "insurance_valid"]:
        missing_rate = raw_data[col].isna().mean()
        assert missing_rate < 0.05, f"{col} missing rate too high: {missing_rate:.2%}"


def test_no_duplicate_listing_ids(raw_data):
    assert raw_data["listing_id"].is_unique


# ---------------------------------------------------------------------------
# MODEL SANITY TESTS
# ---------------------------------------------------------------------------

def test_model_loads(model):
    assert model is not None


def test_model_predicts_positive_price(model):
    sample = pd.DataFrame([{
        "brand": "Hyundai", "city": "Bengaluru", "car_age": 4, "km_driven": 48000,
        "fuel_type": "Petrol", "transmission": "Manual", "owner_count": 1,
        "mileage_kmpl": 18.5, "engine_cc": 1197, "insurance_valid": "Yes",
        "km_per_year": 48000 / 4,
    }])
    pred = model.predict(sample)[0]
    assert pred > 0


def test_older_car_predicts_lower_price_than_newer(model):
    """Sanity check: holding everything else equal, an older car should be
    predicted cheaper than a newer one. This catches silent feature-encoding
    bugs that a pure accuracy metric might miss."""
    base = {
        "brand": "Maruti Suzuki", "city": "Delhi NCR", "km_driven": 40000,
        "fuel_type": "Petrol", "transmission": "Manual", "owner_count": 1,
        "mileage_kmpl": 18.0, "engine_cc": 1197, "insurance_valid": "Yes",
    }
    newer = pd.DataFrame([{**base, "car_age": 2, "km_per_year": 40000 / 2}])
    older = pd.DataFrame([{**base, "car_age": 10, "km_per_year": 40000 / 10}])

    price_newer = model.predict(newer)[0]
    price_older = model.predict(older)[0]

    assert price_newer > price_older


def test_automatic_transmission_commands_premium(model):
    """Sanity check: automatic transmission should predict a higher price
    than manual, all else equal — matches the EDA finding."""
    base = {
        "brand": "Honda", "city": "Mumbai", "car_age": 3, "km_driven": 30000,
        "fuel_type": "Petrol", "owner_count": 1, "mileage_kmpl": 17.0,
        "engine_cc": 1197, "insurance_valid": "Yes", "km_per_year": 10000,
    }
    manual = pd.DataFrame([{**base, "transmission": "Manual"}])
    automatic = pd.DataFrame([{**base, "transmission": "Automatic"}])

    assert model.predict(automatic)[0] > model.predict(manual)[0]
