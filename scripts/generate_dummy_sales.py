import pandas as pd
import numpy as np
import random
from faker import Faker
from pathlib import Path

from datetime import timedelta, date

# --- ADD HERE: Retail week helpers (Sun–Sat, retail year starts Feb 1) ---
def _retail_year_for(d: date) -> int:
    feb1_current = date(d.year, 2, 1)
    return d.year if d >= feb1_current else d.year - 1

def _week1_bounds(retail_year: int):
    start = date(retail_year, 2, 1)
    end = start + timedelta(days=(5 - start.weekday()) % 7)  # first Saturday on/after Feb 1
    return start, end

def retail_week_info(d: date):
    ry = _retail_year_for(d)
    w1_start, w1_end = _week1_bounds(ry)

    if d < w1_start:
        ry -= 1
        w1_start, w1_end = _week1_bounds(ry)

    if w1_start <= d <= w1_end:
        week_no = 1
        wk_start = w1_start
        wk_end = w1_end
    else:
        first_full_week_start = w1_end + timedelta(days=1)  # Sunday
        days_since = (d - first_full_week_start).days
        blocks = days_since // 7
        week_no = 2 + blocks
        wk_start = first_full_week_start + timedelta(days=blocks * 7)
        wk_end = wk_start + timedelta(days=6)

    return {
        "RetailYear": ry,
        "RetailWeek": week_no,
        "RetailWeekStart": wk_start.isoformat(),
        "RetailWeekEnd": wk_end.isoformat(),
        "RetailWeekLabel": f"RY{ry}-W{week_no:02d}",
    }
# --- end helpers ---

fake = Faker()

# ======= Dimensions =======
categories = ["Jackets", "Footwear", "Backpacks", "Trousers", "Shirts"]
brands = ["RegattaX", "NorthPeak", "SummitGear", "TrailPro", "UrbanClimb"]
genders = ["Men", "Women", "Unisex"]

# --- NEW: Category weighting, price bands, and unit multipliers ---
category_weights = {
    "Jackets": 0.30,
    "Footwear": 0.20,
    "Backpacks": 0.15,
    "Trousers": 0.20,
    "Shirts": 0.15,
}

# Base price bands (exclusive upper bound, to match np.random.randint behavior)
price_ranges = {
    "Jackets": (80, 200),     # 80–199
    "Footwear": (60, 150),    # 60–149
    "Backpacks": (30, 100),   # 30–99
    "Trousers": (40, 120),    # 40–119
    "Shirts": (20, 70),       # 20–69
}

# Relative unit volume factors (higher for cheaper, faster-moving categories)
unit_factors = {
    "Jackets": 1.0,
    "Footwear": 1.2,
    "Backpacks": 1.5,
    "Trousers": 1.1,
    "Shirts": 1.6,
}
# --- end NEW ---

# Rough bounding boxes for UK regions
region_coords = {
    "North": {"lat": (53.5, 55.0), "lon": (-3.0, -1.0)},   # Manchester/Leeds/Newcastle
    "South": {"lat": (50.8, 51.6), "lon": (-0.5, 0.5)},    # London/Brighton
    "East":  {"lat": (52.3, 53.5), "lon": (0.2, 1.5)},     # Norwich/Cambridge
    "West":  {"lat": (51.2, 52.5), "lon": (-3.5, -2.0)},   # Bristol/Cardiff
}

# generate 20 stores across regions with fixed lat/long
stores = []
for i in range(1, 21):
    region = random.choice(list(region_coords.keys()))
    lat_range = region_coords[region]["lat"]
    lon_range = region_coords[region]["lon"]

    stores.append({
        "StoreID": f"S{i:03d}",
        "StoreName": f"{region} Outlet {i}",
        "Region": region,
        "Latitude": round(random.uniform(*lat_range), 6),
        "Longitude": round(random.uniform(*lon_range), 6)
    })

promo_types = ["Seasonal", "Clearance", "Flash", "Bundle"]

# ======= Config =======
n_rows = 6000
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

def seasonal_multiplier(category: str, month: int) -> float:
    """Bump volume/value by category seasonality."""
    mult = 1.0
    if category in ["Jackets", "Trousers"] and month in [10, 11, 12, 1, 2]:
        mult *= np.random.uniform(1.1, 1.4)
    if category in ["Backpacks", "Shirts"] and month in [6, 7, 8]:
        mult *= np.random.uniform(1.05, 1.3)
    return mult

def promo_probability(date: pd.Timestamp, category: str) -> float:
    """Return probability of a promo on this date for this category."""
    p = 0.08
    if date.weekday() >= 5:
        p += 0.05
    if date.day >= 26:
        p += 0.04
    if category in ["Jackets", "Trousers"] and date.month in [10, 11, 12, 1]:
        p += 0.05
    if category in ["Shirts", "Backpacks"] and date.month in [6, 7, 8]:
        p += 0.05
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

    # --- UPDATED: weighted category choice ---
    category = random.choices(
        population=categories,
        weights=[category_weights[c] for c in categories],
        k=1
    )[0]

    brand = random.choice(brands)
    gender = random.choice(genders)
    product = f"{brand} {category} {fake.word().capitalize()}"

    store = random.choice(stores)
    store_id = store["StoreID"]
    store_name = store["StoreName"]
    region = store["Region"]
    latitude = store["Latitude"]
    longitude = store["Longitude"]

    # --- UPDATED: category-based pricing and units ---
    # Base units 1–4 then scaled by category factor (at least 1)
    base_units = np.random.randint(1, 5)
    units = max(int(round(base_units * unit_factors[category])), 1)

    # Category price bands (np.random.randint upper bound is exclusive)
    pr_lo, pr_hi = price_ranges[category]
    base_price = np.random.randint(pr_lo, pr_hi)
    list_price = round(base_price * np.random.uniform(0.9, 1.15), 2)
    # --- end UPDATED ---

    # Seasonality
    season_mult = seasonal_multiplier(category, month)

    # Promo decision
    is_promo = np.random.rand() < promo_probability(dt, category)
    promo_type = random.choice(promo_types) if is_promo else None
    discount_pct = promo_discount_pct(promo_type) if is_promo else 0.0

    net_unit_price = round(list_price * (1 - discount_pct), 2)
    units_effective = units
    if is_promo:
        units_effective = int(round(units * np.random.uniform(1.05, 1.3)))
        units_effective = max(units_effective, 1)

    net_sales = round(units_effective * net_unit_price * season_mult, 2)

    margin_pct_base = float(np.random.choice([0.35, 0.4, 0.45, 0.5]))
    effective_margin_pct = max(margin_pct_base - discount_pct * 0.6, 0.05)
    margin_value = round(net_sales * effective_margin_pct, 2)

    year = dt.year
    planned_growth = np.random.uniform(1.03, 1.08)
    if year == 2024:
        budget = round(net_sales * np.random.uniform(0.92, 1.08), 2)
    else:
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
        latitude,
        longitude,
        units_effective,
        list_price,
        discount_pct,
        net_unit_price,
        round(units_effective * list_price, 2),
        net_sales,
        effective_margin_pct,
        margin_value,
        bool(is_promo),
        promo_type,
        budget,
        year
    ])

df = pd.DataFrame(rows, columns=[
    "InvoiceID","Date","Product","Category","Gender","Brand",
    "StoreID","StoreName","Region","Latitude","Longitude",
    "UnitsSold","ListPrice","DiscountPct","NetUnitPrice",
    "GrossSales","NetSales","MarginPct","MarginValue",
    "IsPromo","PromoType",
    "Budget","Year"
])

# --- ADD HERE: derive retail-week columns ---
wk = df["Date"].apply(lambda s: retail_week_info(pd.to_datetime(s).date()))
wk_df = pd.DataFrame(list(wk))
df = pd.concat([df, wk_df], axis=1)
# --- end add ---

Path("data").mkdir(exist_ok=True, parents=True)
df.to_csv("data/dummy_retail_sales.csv", index=False)
print("✅ Generated data/dummy_retail_sales.csv with", len(df), "rows")
