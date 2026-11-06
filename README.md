# Online Retail: ETL and RFM Features

A Python pipeline that cleans retail transactions and produces a customer-level RFM table for segmentation analysis.

**Stack:** Python · pandas · Excel/CSV

## Pipeline

```text
Online Retail.xlsx → transaction cleaning → TotalPrice → customer-level RFM → CSV
```

Cleaning removes rows without a customer ID, cancelled invoices, and non-positive quantities or unit prices.

| Feature | Definition |
| :--- | :--- |
| Recency | Days since the customer's latest purchase, relative to one day after the dataset's latest transaction |
| Frequency | Number of distinct invoices |
| Monetary | Sum of `Quantity × UnitPrice` across retained transactions |

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python src/etl_rfm.py
```

The input is `Online Retail.xlsx`. Outputs are:

- `data/online_retail_clean.csv`: cleaned transaction rows.
- `data/rfm.csv`: one row per customer.

See [src/etl_rfm.py](src/etl_rfm.py) for preprocessing and aggregation rules. The current public branch covers ETL and feature construction; clustering models are outside this implementation.
