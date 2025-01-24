import pandas as pd
import sqlite3

excel_path = "reference/test_table.xlsx"
orders_df = pd.read_excel(excel_path, sheet_name="Orders")
tickers_df = pd.read_excel(excel_path, sheet_name="Tickers")

# Create an in-memory SQLite database
conn = sqlite3.connect(":memory:")

# Load DataFrames into SQLite
orders_df.to_sql("Orders", conn, index=False, if_exists="replace")
tickers_df.to_sql("Tickers", conn, index=False, if_exists="replace")

# Define the SQL query
query = """
SELECT t.ticker_name, SUM(o.shares) AS total_shares
FROM Orders o
JOIN Tickers t ON o.ticker_id = t.ticker_id
WHERE t.ticker IN ('ACME', 'DELTA')
  AND o.date BETWEEN '2023-12-01' AND '2023-12-31'
GROUP BY t.ticker_name;
"""

# Execute the query
result = pd.read_sql_query(query, conn)

# Display the result
print(result)
