# Customer Segmentation: ETL and RFM

## Setup

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the ETL pipeline

The script reads `Online Retail.xlsx`, removes invalid transactions, creates
`TotalPrice`, and aggregates customer-level RFM features.

```bash
python src/etl_rfm.py
```

Generated files:

- `data/online_retail_clean.csv`: valid transaction-level data.
- `data/rfm.csv`: one row per customer with `Recency`, `Frequency`, and `Monetary`.

## Run GMM customer segmentation

The GMM pipeline runs independently of K-Means. It caps the top 1% of each RFM
feature, applies `log1p` and standardization, then compares 2-10 components
and four covariance structures. The default covariance regularization is 0.1
because the nearly discrete `Frequency` values made the earlier 1e-6 run
overconfident. Read [the GMM guide](docs/GMM_GUIDE.md) before interpreting
the clusters or comparing them with the modeling notebook.

```bash
python src/gmm_modeling.py
python src/gmm_robustness.py
python src/verify_gmm.py
```

The reviewed run writes to `results/gmm_regularized/`. The superseded
`results/gmm/` experiment is kept locally and is not part of the GMM submission.
Generated files include:

- `tables/*.csv`: assignments, segment profiles, strategies, model selection, and robustness checks.
- `figures/*.png`: four diagnostic charts.
- `model/gmm_model.joblib`: fitted model plus preprocessing metadata.
- `run_summary.json`: input fingerprint, configuration, and final metrics.

See [results/README.md](results/README.md) for the run folders and output layout.

Run `python src/gmm_modeling.py --help` to customize the input, output,
component range, winsorization quantile, covariance regularization, or seed.
The GMM guide explains the selection rule, limitations, and how to interpret
the outputs. Bibica business data is analyzed separately and is not an input
to this repository's GMM run.

The [20 October GMM review](docs/GMM_REVIEW_2026-10-20.md) also records a
separate run on the 4,320-customer feature table from the teammate's open PR.
Those results are kept in `results/gmm_group_features/` and are not silently
mixed with the default 4,338-customer run.

For a direct comparison with the teammate's K-Means on the same 4,320 rows and
the same uncapped `log1p` + `StandardScaler` matrix, read the
[K-Means/GMM comparison](docs/KMEANS_GMM_COMPARISON_2026-10-20.md). The
reproducible comparison code is `src/compare_public_models.py`.
