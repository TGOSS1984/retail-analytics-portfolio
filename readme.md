# Retail Analytics Portfolio – Sales & Budget (Dummy Data)

**What’s inside**
- Synthetic retail dataset (2024–2025) with invoices, products, category, gender, brand, units, sales, margin, and budget.
- SQL queries for ATV, items/transaction, YoY, and budget variance.
- A Jupyter notebook for quick exploration.
- Power BI dashboard showing core merchandising KPIs.

**Business questions**
- How are sales tracking vs budget and last year?
- Which categories/brands drive sales and margin?
- What are ATV and items per transaction by month?

**Tech**
- Python (pandas/numpy), SQL, Power BI
- Data: `data/dummy_retail_sales.csv`

**How to reproduce**
```bash
python -m venv .venv
# activate venv (see instructions above)
pip install pandas numpy faker jupyter matplotlib
python scripts/generate_dummy_sales.py
