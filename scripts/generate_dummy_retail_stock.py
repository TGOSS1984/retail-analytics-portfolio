# scripts/dummy_retail_stock.py
import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import date

# Reproducibility while you iterate
random.seed(42)
np.random.seed(42)

SALES_PATH = Path("data") / "dummy_retail_sales.csv"
STOCK_PATH = Path("data") / "dummy_retail_stock.csv"

def category_stock_range(category: str):
    """
    Rough ranges for stock-on-hand by category.
    Adjust to taste if you want fatter/slimmer inventories.
    """
    ranges = {
        "Jackets":    (5, 60),
        "Footwear":   (10, 120),
        "Backpacks":  (5, 80),
        "Trousers":   (10, 150),
        "Shirts":     (15, 200),
    }
    return ranges.get(category, (10, 100))

def main():
    if not SALES_PATH.exists():
        raise FileNotFoundError(
            f"Sales file not found at {SALES_PATH}. "
            "Run scripts/generate_dummy_sales.py first."
        )

    df_sales = pd.read_csv(SALES_PATH)

    # Ensure required columns exist
    needed = ["StoreID", "StoreName", "Region", "Product", "Category", "Gender", "Brand"]
    missing = [c for c in needed if c not in df_sales.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in sales CSV: {missing}. "
            "Regenerate the sales file with the latest script."
        )

    # --- TRUE UNIQUE: one row per StoreID + Product (keys for the relationship)
    # Keep first-seen descriptive fields to avoid duplicates on the 'one' side
    base = (
        df_sales.groupby(["StoreID", "Product"], as_index=False)
        .agg(StoreName=("StoreName", "first"),
             Region=("Region", "first"),
             Category=("Category", "first"),
             Gender=("Gender", "first"),
             Brand=("Brand", "first"))
    )

    # Optional: write a composite key so model is simpler in Power BI
    base["StoreProductKey"] = base["StoreID"] + "||" + base["Product"]

    # Generate stock figures (snapshot "as of" today)
    as_of = date.today().isoformat()
    on_hand_list, on_order_list, reorder_point_list = [], [], []

    for _, row in base.iterrows():
        lo, hi = category_stock_range(row["Category"])
        on_hand = int(np.random.randint(lo, hi + 1))
        reorder_point = max(int(lo + (hi - lo) * 0.25), 2)
        on_order = int(max(0, np.random.normal(loc=reorder_point * 0.3, scale=reorder_point * 0.2)))
        if np.random.rand() < 0.15:
            on_order += int(np.random.randint(1, max(2, reorder_point // 3)))

        on_hand_list.append(on_hand)
        on_order_list.append(on_order)
        reorder_point_list.append(reorder_point)

    df_stock = base.copy()
    df_stock["OnHandUnits"] = on_hand_list
    df_stock["OnOrderUnits"] = on_order_list
    df_stock["ReorderPoint"] = reorder_point_list
    df_stock["AsOfDate"] = as_of
    df_stock["ReorderFlag"] = df_stock["OnHandUnits"] < df_stock["ReorderPoint"]

    # Save next to the sales CSV
    STOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_stock.to_csv(STOCK_PATH, index=False)

    print(f"✅ Generated {STOCK_PATH} with {len(df_stock):,} rows "
          f"(unique StoreID+Product pairs, AsOf {as_of}).")

if __name__ == "__main__":
    main()
