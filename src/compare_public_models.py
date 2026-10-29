"""Compare K-Means and GMM on the exact same public Online Retail RFM matrix.

The input should be the group's customer_features.csv. No Bibica data is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFM = ["Recency", "Frequency", "Monetary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "customer_features.csv"
    )
    parser.add_argument(
        "--gmm-dir", type=Path, default=PROJECT_ROOT / "results" / "gmm_shared_matrix"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "public_model_comparison",
    )
    parser.add_argument(
        "--notebook-labels", type=Path,
        help="Optional outputs/customer_segments.csv from the teammate's PR.",
    )
    return parser.parse_args()


def evaluate(name: str, x: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str]:
    counts = np.unique(labels, return_counts=True)[1]
    return {
        "Model": name,
        "Components": len(counts),
        "Silhouette": float(silhouette_score(x, labels)),
        "DaviesBouldin": float(davies_bouldin_score(x, labels)),
        "CalinskiHarabasz": float(calinski_harabasz_score(x, labels)),
        "SmallestSegment": int(counts.min()),
        "LargestSegment": int(counts.max()),
        "SmallestShare": float(counts.min() / len(labels)),
        "LargestShare": float(counts.max() / len(labels)),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    gmm_dir = args.gmm_dir.resolve()
    output_dir = args.output_dir.resolve()

    features = pd.read_csv(input_path)
    if features["CustomerID"].isna().any() or not features["CustomerID"].is_unique:
        raise ValueError("CustomerID must be complete and unique.")
    if features[RFM].isna().any().any() or not np.isfinite(features[RFM].to_numpy()).all():
        raise ValueError("RFM features must be finite and complete.")
    if (features[RFM] <= 0).any().any():
        raise ValueError("RFM features must be positive.")

    gmm_summary = json.loads((gmm_dir / "run_summary.json").read_text(encoding="utf-8"))
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if input_sha != gmm_summary["input_sha256"]:
        raise ValueError("GMM result was fitted on a different input file.")
    if gmm_summary["preprocessing"]["winsor_quantile"] != 1.0:
        raise ValueError("GMM must be fitted without capping for the notebook's shared matrix.")

    x = StandardScaler().fit_transform(np.log1p(features[RFM]))
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42).fit(x)
    gmm4 = GaussianMixture(
        n_components=4, covariance_type="full", reg_covar=0.1,
        n_init=30, max_iter=1_000, random_state=42,
    ).fit(x)
    if not gmm4.converged_:
        raise RuntimeError("Four-component GMM did not converge.")

    saved = pd.read_csv(gmm_dir / "tables" / "customer_segments.csv")
    matched = features[["CustomerID"]].merge(
        saved[["CustomerID", "SegmentID"]], on="CustomerID", validate="one_to_one"
    )
    if len(matched) != len(features):
        raise ValueError("GMM assignments do not cover every customer.")

    labels = {
        "K-Means k=4": kmeans.labels_,
        "GMM k=4": gmm4.predict(x),
        f"GMM BIC k={gmm_summary['selected_model']['n_components']}":
            matched["SegmentID"].to_numpy(),
    }
    comparison = pd.DataFrame([evaluate(name, x, values) for name, values in labels.items()])
    agreement = pd.DataFrame(
        [
            {"ModelA": a, "ModelB": b, "AdjustedRandIndex": float(adjusted_rand_score(la, lb))}
            for i, (a, la) in enumerate(labels.items())
            for b, lb in list(labels.items())[i + 1:]
        ]
    )

    notebook_check = None
    if args.notebook_labels:
        notebook = pd.read_csv(args.notebook_labels.resolve())
        aligned = features[["CustomerID"]].merge(
            notebook[["CustomerID", "KMeans_Cluster", "GMM_Cluster"]],
            on="CustomerID", validate="one_to_one",
        )
        if len(aligned) != len(features):
            raise ValueError("Notebook output does not cover every customer.")
        notebook_check = {
            "kmeans_ari": float(adjusted_rand_score(labels["K-Means k=4"], aligned["KMeans_Cluster"])),
            "notebook_gmm_vs_refit_gmm4_ari": float(adjusted_rand_score(labels["GMM k=4"], aligned["GMM_Cluster"])),
        }

    summary = {
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "customer_count": len(features),
        "shared_preprocessing": "log1p then StandardScaler; no winsorization",
        "sklearn_version": sklearn_version,
        "kmeans": {"k": 4, "n_init": 20, "seed": 42, "inertia": float(kmeans.inertia_)},
        "gmm_fixed_k4": {
            "covariance_type": "full", "reg_covar": 0.1, "n_init": 30,
            "seed": 42, "bic": float(gmm4.bic(x)),
            "median_max_posterior": float(np.median(gmm4.predict_proba(x).max(axis=1))),
        },
        "gmm_bic_selected": {
            "k": int(gmm_summary["selected_model"]["n_components"]),
            "bic": float(gmm_summary["selected_model"]["bic"]),
        },
        "notebook_reproduction": notebook_check,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    agreement.to_csv(tables_dir / "model_agreement.csv", index=False)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(agreement.to_string(index=False))
    if notebook_check:
        print("Notebook reproduction:", notebook_check)


if __name__ == "__main__":
    main()
