"""
generate_data.py (v2)
----------------------
Generates a realistic used-car MARKETPLACE dataset - not just static listings,
but a full listing-to-sale funnel, which is what makes the SQL analysis and
dashboard meaningful (days-to-sell, sell-through rate, price negotiation gap,
monthly trends - the metrics a real CARS24 ops/BI team would track).

Two tables are produced (a mini star-schema):
  - car_listings.csv   : one row per car listed (dimension + listing facts)
  - dim_city.csv       : city reference data (region, tier) for JOIN practice
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N = 8000

brand_model_map = {
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Dzire", "Alto"],
    "Hyundai":       ["i20", "Creta", "Venue", "Grand i10", "Verna"],
    "Tata":          ["Nexon", "Tiago", "Altroz", "Punch", "Harrier"],
    "Honda":         ["City", "Amaze", "Jazz", "WR-V"],
    "Toyota":        ["Innova", "Fortuner", "Glanza", "Urban Cruiser"],
    "Mahindra":      ["XUV700", "Scorpio", "Bolero", "XUV300"],
    "Kia":           ["Seltos", "Sonet", "Carens"],
    "Ford":          ["EcoSport", "Figo"],
    "Volkswagen":    ["Polo", "Vento"],
    "Renault":       ["Kwid", "Triber"],
}
brand_base_price_lakh = {"Maruti Suzuki": 7.0, "Hyundai": 9.0, "Tata": 9.5, "Honda": 10.5,
                          "Toyota": 15.0, "Mahindra": 13.0, "Kia": 12.5, "Ford": 9.0,
                          "Volkswagen": 10.0, "Renault": 7.5}

city_meta = {
    "Delhi NCR":   ("North", 1, 1.05), "Mumbai": ("West", 1, 1.10),
    "Bengaluru":   ("South", 1, 1.08), "Pune": ("West", 1, 1.03),
    "Hyderabad":   ("South", 1, 1.02), "Chennai": ("South", 1, 1.00),
    "Kolkata":     ("East", 1, 0.95),  "Ahmedabad": ("West", 2, 0.97),
    "Jaipur":      ("North", 2, 0.93), "Chandigarh": ("North", 2, 0.96),
}
cities = list(city_meta.keys())

fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
fuel_weights = [0.55, 0.30, 0.12, 0.03]
fuel_price_adj = {"Petrol": 1.00, "Diesel": 1.05, "CNG": 0.90, "Electric": 1.25}
transmissions = ["Manual", "Automatic"]
transmission_weights = [0.72, 0.28]

start_date = datetime(2024, 1, 1)
rows = []

for i in range(N):
    brand = np.random.choice(list(brand_model_map.keys()))
    model = np.random.choice(brand_model_map[brand])
    city = np.random.choice(cities)
    region, tier, city_mult = city_meta[city]
    fuel = np.random.choice(fuel_types, p=fuel_weights)
    transmission = np.random.choice(transmissions, p=transmission_weights)

    year = np.random.randint(2012, 2025)
    car_age = 2025 - year
    km_driven = max(500, int(np.random.normal(loc=car_age * 12000, scale=8000)))
    owner_count = np.random.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05])
    mileage_kmpl = max(4, round(np.random.normal(18, 4), 1) if fuel != "Electric" else round(np.random.normal(6, 1), 1))
    engine_cc = int(np.random.choice([800, 1000, 1197, 1199, 1498, 1997, 2000, 2200],
                                      p=[0.08, 0.15, 0.20, 0.15, 0.20, 0.10, 0.07, 0.05]))
    insurance_valid = np.random.choice(["Yes", "No"], p=[0.78, 0.22])

    base = brand_base_price_lakh[brand] * 100000
    depreciation_factor = max(0.15, (0.88 ** car_age))
    price = base * depreciation_factor
    price *= (1 - min(0.35, km_driven / 300000))
    price *= (1 - 0.05 * (owner_count - 1))
    if transmission == "Automatic":
        price *= 1.12
    price *= fuel_price_adj[fuel]
    price *= city_mult
    if insurance_valid == "Yes":
        price *= 1.03
    price *= (1 + (engine_cc - 1000) / 20000)
    price *= np.random.normal(1.0, 0.08)
    asking_price = max(50000, round(price, -3))

    listing_date = start_date + timedelta(days=int(np.random.randint(0, 545)))

    base_days = 12 + car_age * 2.2 - (city_mult - 1) * 20
    days_to_sell = max(1, int(np.random.exponential(scale=max(3, base_days))))
    is_sold = listing_date + timedelta(days=days_to_sell) <= datetime(2025, 6, 30)

    negotiation_pct = np.random.uniform(0.0, 0.08)
    sold_price = int(asking_price * (1 - negotiation_pct)) if is_sold else None

    rows.append({
        "listing_id": f"C24-{10000+i}",
        "brand": brand, "model": model, "city": city,
        "year": year, "car_age": car_age, "km_driven": km_driven,
        "fuel_type": fuel, "transmission": transmission, "owner_count": owner_count,
        "mileage_kmpl": mileage_kmpl, "engine_cc": engine_cc,
        "insurance_valid": insurance_valid,
        "listing_date": listing_date.strftime("%Y-%m-%d"),
        "asking_price": asking_price,
        "status": "Sold" if is_sold else "Active",
        "days_to_sell": days_to_sell if is_sold else None,
        "sold_price": sold_price,
    })

df = pd.DataFrame(rows)

for col in ["mileage_kmpl", "insurance_valid", "owner_count"]:
    mask = np.random.rand(len(df)) < 0.02
    df.loc[mask, col] = np.nan

df.to_csv("data/car_listings.csv", index=False)

dim_city = pd.DataFrame([{"city": c, "region": v[0], "tier": v[1]} for c, v in city_meta.items()])
dim_city.to_csv("data/dim_city.csv", index=False)

compat = df.rename(columns={"asking_price": "selling_price"}).drop(
    columns=["listing_date", "status", "days_to_sell", "sold_price"])
compat.to_csv("data/car_data.csv", index=False)

print(f"Generated {len(df)} listings -> data/car_listings.csv")
print(f"Sold: {(df['status']=='Sold').sum()}  |  Active: {(df['status']=='Active').sum()}")
print(dim_city)
