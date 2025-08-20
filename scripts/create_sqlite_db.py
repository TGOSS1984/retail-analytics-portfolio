from pathlib import Path
import pandas as pd
import sqlite3

csv_path = Path("data/dummy_retail_sales.csv")
db_path  = Path("data/retail.db")

if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found at {csv_path.resolve()}")

# Load CSV and normalise date to ISO string (SQLite stores as TEXT)
df = pd.read_csv(csv_path, parse_dates=["Date"])
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

# Write to SQLite
db_path.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(db_path) as conn:
    df.to_sql("retail_sales", conn, if_exists="replace", index=False)
    cur = conn.cursor()
    # helpful indexes for faster queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_year ON retail_sales (Year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_category ON retail_sales (Category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_region ON retail_sales (Region)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_invoice ON retail_sales (InvoiceID)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_promo ON retail_sales (IsPromo)")
    conn.commit()

print(f"✅ Created SQLite DB at {db_path} with table 'retail_sales'")
