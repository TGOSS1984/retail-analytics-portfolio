-- Average Transaction Value (ATV) and Items per Transaction
WITH invoice_totals AS (
  SELECT InvoiceID,
         SUM(Sales) AS sales_total,
         SUM(UnitsSold) AS units_total
  FROM retail_sales
  GROUP BY InvoiceID
)
SELECT
  AVG(sales_total) AS avg_transaction_value,
  AVG(units_total) AS items_per_transaction
FROM invoice_totals;
