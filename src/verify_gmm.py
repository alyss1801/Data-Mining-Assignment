"""Verify saved GMM outputs against a trusted local model bundle and input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gmm_modeling import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, FEATURES, load_rfm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest(), (
        "Input file differs from the training run"
    )
    rfm = load_rfm(input_path)
    customers = pd.read_csv(output_dir / "tables" / "customer_segments.csv")
    profiles = pd.read_csv(output_dir / "tables" / "segment_profiles.csv")
    strategies = pd.read_csv(output_dir / "tables" / "segment_strategies.csv")
    assert len(rfm) == len(customers) == summary["customer_count"]
    assert customers["CustomerID"].is_unique
    assert np.array_equal(rfm["CustomerID"].to_numpy(), customers["CustomerID"].to_numpy())

    # Load only a trusted bundle created by this local pipeline.
    bundle = joblib.load(output_dir / "model" / "gmm_model.joblib")
    assert bundle["features"] == FEATURES
    capped = rfm[FEATURES].copy()
    for feature in FEATURES:
        capped[feature] = capped[feature].clip(upper=bundle["upper_caps"][feature])
    x = bundle["scaler"].transform(np.log1p(capped))
    model = bundle["model"]
    raw_labels = model.predict(x)
    probabilities = model.predict_proba(x)
    mapped = np.array([bundle["raw_cluster_to_segment_id"][int(label)] for label in raw_labels])
    assert np.array_equal(mapped, customers["SegmentID"].to_numpy())
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(customers["AssignmentConfidence"], probabilities.max(axis=1))
    assert np.allclose(
        customers["ProbabilityMargin"],
        np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2],
    )
    for raw_id, segment_id in bundle["raw_cluster_to_segment_id"].items():
        assert np.allclose(customers[f"Probability_S{segment_id}"], probabilities[:, raw_id])
    assert np.isclose(model.bic(x), summary["selected_model"]["bic"])
    assert np.isclose(model.aic(x), summary["selected_model"]["aic"])
    assert profiles["CustomerCount"].sum() == len(rfm)
    assert np.isclose(profiles["CustomerShare"].sum(), 1.0)
    assert np.isclose(profiles["MonetaryShare"].sum(), 1.0)
    assert set(profiles["SegmentID"]) == set(strategies["SegmentID"])
    for name in (
        "model_selection.png", "pca_segments.png", "segment_profile_heatmap.png",
        "assignment_confidence.png",
    ):
        assert (output_dir / "figures" / name).stat().st_size > 0
    print(f"PASS: {len(rfm):,} customers, {len(profiles)} segments; "
          "saved assignments, probabilities, metrics and profiles match the bundle.")


if __name__ == "__main__":
    main()
