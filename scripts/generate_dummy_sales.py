import pandas as pd
import numpy as np
import random
from faker import Faker
from pathlib import Path

fake = Faker()

# ======= Dimensions =======
categories = ["Jackets", "Footwear", "Backpacks", "Trousers", "Shirts"]
brands = ["RegattaX", "NorthPeak", "SummitGear", "TrailPro", "UrbanClimb"]
genders = ["Men", "Women", "Unisex"]

regions = ["North", "South", "East", "West"]
# generate 20 stores across regions
stores = []
for i in range(1, 21):
    region = random.choice(regions)
    stores.append({
        "StoreID": f"S{i:03d}",
        "StoreName": f"{region} Outlet {i}",
        "Region": region
    })

promo_types = ["Seasonal", "Clearance", "Flash", "Bundle"]

# ======= Config =======
n_rows = 6000
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

def seasonal_multiplier(category: str, month: int) -> float:
    """Bump volume/value by category seasonality."""
    mult = 1.0
    # Jackets & Trousers stronger in colder months
    if category in ["Jackets", "Trousers"] and month in [10, 11, 12, 1, 2]:
        mult *= np.random.uniform(1.1, 1.4)
    # Backpacks & Shirts stronger in summer
    if category in ["Backpacks", "Shirts"] and month in [6, 7, 8]:
        mult *= np.random.uniform(1.05, 1.3)
    return mult

def promo_probability(date: pd.Timestamp, category: str) -> float:
    """Return probability of a promo on this date for this category."""
    p = 0.08  # base 8%
    # Weekends slightly higher
    if date.weekday() >= 5:
        p += 0.05
    # End of month bump
    if date.day >= 26:
        p += 0.04
    # Seasonal promos: winter for Jackets/Trousers, summer for Shirts/Backpacks
    if category in ["Jackets", "Trousers"] and date.month in [10, 11, 12, 1]:
        p += 0.05
    if category in ["Shirts", "Backpacks"] and date.month in [6, 7, 8]:
        p += 0.05
    # Cap between 0 and 0.35
    return min(max(p, 0.0), 0.35)

def promo_discount_pct(promo_type: str) -> float:
    """Discount percent by promo type."""
    if promo_type == "Seasonal":
        return np.random.uniform(0.10, 0.25)
    if promo_type == "Clearance":
        return np.random.uniform(0.25, 0.50)
    if promo_type == "Flash":
        return np.random.uniform(0.15, 0.35)
    if promo_type == "Bundle":
        return np.random.uniform(0.10, 0.30)
    return 0.0

rows = []
for i in range(n_rows):
    dt = random.choice(dates)
    month = dt.month

    category = random.choice(categories)
    brand = random.choice(brands)
    gender = random.choice(genders)

    product = f"{brand} {category} {fake.word().capitalize()}"

    store = random.choice(stores)
    store_id = store["StoreID"]
    store_name = store["StoreName"]
    region = store["Region"]

    # Units and pricing
    units = np.random.randint(1, 5)
    base_price = np.random.randint(20, 200)
    list_price = round(base_price * np.random.uniform(0.9, 1.15), 2)

    # Seasonality on demand/realised net sales
    season_mult = seasonal_multiplier(category, month)

    # Promo decision
    is_promo = np.random.rand() < promo_probability(dt, category)
    promo_type = random.choice(promo_types) if is_promo else None
    discount_pct = promo_discount_pct(promo_type) if is_promo else 0.0

    # Apply discount to get net price; modest uplift in units during promo
    net_unit_price = round(list_price * (1 - discount_pct), 2)
    units_effective = units
    if is_promo:
        # uplift units during promo
        units_effective = int(round(units * np.random.uniform(1.05, 1.3)))
        units_effective = max(units_effective, 1)

    # Sales (after discount) with seasonality
    net_sales = round(units_effective * net_unit_price * season_mult, 2)

    # Margin – assume margin based on list price, then discount erodes margin
    margin_pct_base = float(np.random.choice([0.35, 0.4, 0.45, 0.5]))
    # simple model: discount reduces effective margin linearly
    effective_margin_pct = max(margin_pct_base - discount_pct * 0.6, 0.05)
    margin_value = round(net_sales * effective_margin_pct, 2)

    year = dt.year
    # Budget based on planned growth; use net sales as the baseline heuristic
    planned_growth = np.random.uniform(1.03, 1.08)
    if year == 2024:
        budget = round(net_sales * np.random.uniform(0.92, 1.08), 2)
    else:
        # approx last-year baseline by removing a small random variation
        budget = round(net_sales / np.random.uniform(0.96, 1.04) * planned_growth, 2)

    rows.append([
        f"INV{100000+i}",
        dt.date().isoformat(),
        product,
        category,
        gender,
        brand,
        store_id,
        store_name,
        region,
        units_effective,          # UnitsSold (after promo uplift)
        list_price,               # pre-discount
        discount_pct,             # 0–0.5
        net_unit_price,           # after discount
        round(units_effective * list_price, 2),  # GrossSales (before discount)
        net_sales,                # NetSales (after discount)
        effective_margin_pct,     # MarginPct (effective)
        margin_value,             # MarginValue
        bool(is_promo),
        promo_type,
        budget,
        year
    ])

df = pd.DataFrame(rows, columns=[
    "InvoiceID","Date","Product","Category","Gender","Brand",
    "StoreID","StoreName","Region",
    "UnitsSold","ListPrice","DiscountPct","NetUnitPrice",
    "GrossSales","NetSales","MarginPct","MarginValue",
    "IsPromo","PromoType",
    "Budget","Year"
])

Path("data").mkdir(exist_ok=True, parents=True)
df.to_csv("data/dummy_retail_sales.csv", index=False)
print("✅ Generated data/dummy_retail_sales.csv with", len(df), "rows")
