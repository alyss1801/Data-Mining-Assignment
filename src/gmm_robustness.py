"""Reproducible sensitivity and initialization checks for the public RFM GMM."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from gmm_modeling import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    evaluate_candidates,
    load_rfm,
    preprocess_rfm,
    select_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--winsor-quantile", type=float, default=0.99)
    args = parser.parse_args()
    if not 0.90 <= args.winsor_quantile <= 1.0:
        raise ValueError("--winsor-quantile must be between 0.90 and 1.00.")

    rfm = load_rfm(args.input.resolve())
    x, _, _, _ = preprocess_rfm(rfm, args.winsor_quantile)
    rows = []
    for reg_covar in (0.03, 0.05, 0.10, 0.20):
        scores = evaluate_candidates(x, 2, 10, 42, reg_covar)
        selected = select_candidate(scores)
        rows.append({
            "reg_covar": reg_covar,
            "n_components": int(selected["n_components"]),
            "covariance_type": str(selected["covariance_type"]),
            "bic": float(selected["bic"]),
            "smallest_segment": int(selected["min_cluster_count"]),
            "eligible_candidates": int(scores["stable_candidate"].sum()),
        })

    sensitivity = pd.DataFrame(rows)
    chosen = sensitivity.loc[np.isclose(sensitivity["reg_covar"], 0.10)].iloc[0]
    labels_by_seed = {}
    bics = {}
    for seed in range(5):
        model = GaussianMixture(
            n_components=int(chosen["n_components"]),
            covariance_type=str(chosen["covariance_type"]),
            reg_covar=0.10,
            n_init=10,
            max_iter=500,
            random_state=seed,
        ).fit(x)
        labels_by_seed[seed] = model.predict(x)
        bics[seed] = float(model.bic(x))
    stability = pd.DataFrame([
        {"seed_a": a, "seed_b": b,
         "adjusted_rand_index": adjusted_rand_score(labels_by_seed[a], labels_by_seed[b]),
         "bic_a": bics[a], "bic_b": bics[b]}
        for a, b in combinations(labels_by_seed, 2)
    ])

    tables_dir = args.output_dir.resolve() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(tables_dir / "regularization_sensitivity.csv", index=False)
    stability.to_csv(tables_dir / "seed_stability.csv", index=False)
    print(sensitivity.to_string(index=False))
    print(f"Pairwise ARI: min={stability['adjusted_rand_index'].min():.4f}, "
          f"median={stability['adjusted_rand_index'].median():.4f}")


if __name__ == "__main__":
    main()
