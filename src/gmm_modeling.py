

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "rfm.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "gmm_regularized"
FEATURES = ["Recency", "Frequency", "Monetary"]
COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
DEFAULT_REG_COVAR = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment RFM customers using Gaussian Mixture Models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument(
        "--winsor-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used to cap RFM features before log transform.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reg-covar",
        type=float,
        default=DEFAULT_REG_COVAR,
        help="Nonnegative covariance regularization; default 0.1 limits near-zero Frequency variance.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_components < 2:
        raise ValueError("--min-components must be at least 2.")
    if args.max_components < args.min_components:
        raise ValueError("--max-components must be >= --min-components.")
    if not 0.90 <= args.winsor_quantile <= 1.0:
        raise ValueError("--winsor-quantile must be between 0.90 and 1.00.")
    if not np.isfinite(args.reg_covar) or args.reg_covar < 0:
        raise ValueError("--reg-covar must be finite and nonnegative.")


def load_rfm(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"RFM dataset not found: {path}")

    rfm = pd.read_csv(path)
    required = {"CustomerID", *FEATURES}
    missing = required.difference(rfm.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rfm = rfm[["CustomerID", *FEATURES]].copy()
    for column in FEATURES:
        rfm[column] = pd.to_numeric(rfm[column], errors="coerce")

    if rfm[["CustomerID", *FEATURES]].isna().any().any():
        null_counts = rfm[["CustomerID", *FEATURES]].isna().sum()
        raise ValueError(f"Null/non-numeric values found:\n{null_counts}")
    if rfm["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique in the RFM dataset.")
    if not np.isfinite(rfm[FEATURES].to_numpy()).all():
        raise ValueError("RFM features contain infinite values.")
    if (rfm[FEATURES] <= 0).any().any():
        raise ValueError("Recency, Frequency, and Monetary must all be positive.")

    return rfm.sort_values("CustomerID").reset_index(drop=True)


def preprocess_rfm(
    rfm: pd.DataFrame, winsor_quantile: float
) -> tuple[np.ndarray, StandardScaler, dict[str, float], pd.DataFrame]:
    """Cap extreme values, apply log1p, and standardize the three features."""
    caps = {
        feature: float(rfm[feature].quantile(winsor_quantile))
        for feature in FEATURES
    }
    capped = rfm[FEATURES].copy()
    for feature in FEATURES:
        capped[feature] = capped[feature].clip(upper=caps[feature])

    logged = np.log1p(capped)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(logged)
    return scaled, scaler, caps, logged


def evaluate_candidates(
    x_scaled: np.ndarray,
    min_components: int,
    max_components: int,
    seed: int,
    reg_covar: float,
) -> pd.DataFrame:
    """Evaluate GMM candidates across component counts and covariance types."""
    rows: list[dict[str, Any]] = []
    n_customers = len(x_scaled)
    min_stable_count = max(20, int(np.ceil(n_customers * 0.01)))

    for covariance_type in COVARIANCE_TYPES:
        for n_components in range(min_components, max_components + 1):
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                n_init=10,
                max_iter=500,
                reg_covar=reg_covar,
                random_state=seed,
            )
            labels = model.fit_predict(x_scaled)
            counts = np.bincount(labels, minlength=n_components)
            unique_labels = np.unique(labels)

            if len(unique_labels) > 1:
                silhouette = silhouette_score(
                    x_scaled,
                    labels,
                    sample_size=min(2_000, n_customers),
                    random_state=seed,
                )
                calinski = calinski_harabasz_score(x_scaled, labels)
                davies = davies_bouldin_score(x_scaled, labels)
            else:
                silhouette = np.nan
                calinski = np.nan
                davies = np.nan

            rows.append(
                {
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "reg_covar": reg_covar,
                    "converged": bool(model.converged_),
                    "n_iter": int(model.n_iter_),
                    "bic": float(model.bic(x_scaled)),
                    "aic": float(model.aic(x_scaled)),
                    "silhouette": float(silhouette),
                    "calinski_harabasz": float(calinski),
                    "davies_bouldin": float(davies),
                    "min_cluster_count": int(counts.min()),
                    "max_cluster_count": int(counts.max()),
                    "min_cluster_share": float(counts.min() / n_customers),
                    "stable_candidate": bool(
                        model.converged_ and counts.min() >= min_stable_count
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["bic", "n_components", "covariance_type"]
    ).reset_index(drop=True)


def select_candidate(scores: pd.DataFrame) -> pd.Series:
    """Select the lowest-BIC converged candidate meeting the size rule."""
    stable = scores[scores["stable_candidate"]]
    if not stable.empty:
        return stable.loc[stable["bic"].idxmin()]
    raise RuntimeError(
        "No GMM candidate met convergence and minimum-segment-size rules; "
        "review the component range or regularization."
    )


def fit_final_model(
    x_scaled: np.ndarray, selected: pd.Series, seed: int, reg_covar: float
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    model = GaussianMixture(
        n_components=int(selected["n_components"]),
        covariance_type=str(selected["covariance_type"]),
        n_init=30,
        max_iter=1_000,
        reg_covar=reg_covar,
        random_state=seed,
    )
    labels = model.fit_predict(x_scaled)
    probabilities = model.predict_proba(x_scaled)
    if not model.converged_:
        raise RuntimeError("The selected final GMM did not converge.")
    return model, labels, probabilities


def ordered_cluster_mapping(rfm: pd.DataFrame, labels: np.ndarray) -> dict[int, int]:
    """Create deterministic segment IDs ordered from strongest to weakest RFM."""
    profile = rfm.assign(_raw_cluster=labels).groupby("_raw_cluster")[FEATURES].median()
    logged = np.log1p(profile)
    standard = (logged - logged.mean()) / logged.std(ddof=0).replace(0, 1)
    value_score = -standard["Recency"] + standard["Frequency"] + standard["Monetary"]
    ordered_raw_ids = value_score.sort_values(ascending=False).index.tolist()
    return {int(raw_id): rank + 1 for rank, raw_id in enumerate(ordered_raw_ids)}


def profile_level(value: float, low_cut: float, high_cut: float, labels: tuple[str, str, str]) -> str:
    if value <= low_cut:
        return labels[0]
    if value >= high_cut:
        return labels[2]
    return labels[1]


def build_segment_names(
    rfm: pd.DataFrame, segment_ids: np.ndarray
) -> tuple[dict[int, str], dict[int, str]]:
    medians = rfm.assign(SegmentID=segment_ids).groupby("SegmentID")[FEATURES].median()
    cuts = {
        feature: (
            float(rfm[feature].quantile(1 / 3)),
            float(rfm[feature].quantile(2 / 3)),
        )
        for feature in FEATURES
    }
    top_frequency = float(rfm["Frequency"].quantile(0.90))
    top_monetary = float(rfm["Monetary"].quantile(0.90))

    names: dict[int, str] = {}
    strategies: dict[int, str] = {}
    for segment_id, row in medians.iterrows():
        recency = profile_level(
            row["Recency"], *cuts["Recency"], ("Recent", "Cooling", "Dormant")
        )
        frequency = profile_level(
            row["Frequency"], *cuts["Frequency"], ("Rare", "Occasional", "Frequent")
        )
        monetary = profile_level(
            row["Monetary"], *cuts["Monetary"], ("Low value", "Mid value", "High value")
        )
        top_tier = row["Frequency"] >= top_frequency and row["Monetary"] >= top_monetary
        if top_tier:
            frequency = "Very frequent"
            monetary = "Very high value"
        names[int(segment_id)] = (
            f"S{int(segment_id)} - {recency} / {frequency} / {monetary}"
        )

        if top_tier and recency == "Recent":
            strategy = "VIP retention, early access, loyalty rewards, and referral programs."
        elif recency == "Recent" and frequency == "Frequent" and monetary == "High value":
            strategy = "Recognize repeat purchases; test loyalty benefits and relevant cross-sell."
        elif recency == "Dormant" and monetary == "High value":
            strategy = "High-priority win-back campaign with personalized offers and outreach."
        elif recency == "Dormant" and frequency == "Rare":
            strategy = "Low-cost reactivation; suppress after repeated non-response."
        elif recency == "Recent" and frequency == "Rare":
            strategy = "Onboarding and second-purchase incentive to build a repeat habit."
        elif frequency == "Frequent" and monetary != "High value":
            strategy = "Cross-sell, bundles, and threshold offers to increase basket value."
        elif monetary == "High value":
            strategy = "Premium recommendations and proactive retention monitoring."
        elif recency == "Cooling":
            strategy = "Timed reminders, replenishment prompts, and personalized recommendations."
        else:
            strategy = "Nurture with relevant content and controlled promotional testing."
        strategies[int(segment_id)] = strategy

    return names, strategies


def build_outputs(
    rfm: pd.DataFrame,
    raw_labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, int], dict[int, str]]:
    mapping = ordered_cluster_mapping(rfm, raw_labels)
    segment_ids = np.array([mapping[int(label)] for label in raw_labels])
    names, strategies = build_segment_names(rfm, segment_ids)

    customers = rfm.copy()
    customers["SegmentID"] = segment_ids
    customers["Segment"] = customers["SegmentID"].map(names)
    customers["AssignmentConfidence"] = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    customers["ProbabilityMargin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    for raw_id, segment_id in sorted(mapping.items(), key=lambda item: item[1]):
        customers[f"Probability_S{segment_id}"] = probabilities[:, raw_id]

    profiles = (
        customers.groupby(["SegmentID", "Segment"], as_index=False)
        .agg(
            CustomerCount=("CustomerID", "size"),
            RecencyMean=("Recency", "mean"),
            RecencyMedian=("Recency", "median"),
            FrequencyMean=("Frequency", "mean"),
            FrequencyMedian=("Frequency", "median"),
            MonetaryMean=("Monetary", "mean"),
            MonetaryMedian=("Monetary", "median"),
            TotalMonetary=("Monetary", "sum"),
            MeanConfidence=("AssignmentConfidence", "mean"),
            MedianConfidence=("AssignmentConfidence", "median"),
        )
        .sort_values("SegmentID")
    )
    profiles.insert(
        3,
        "CustomerShare",
        profiles["CustomerCount"] / len(customers),
    )
    profiles["MonetaryShare"] = profiles["TotalMonetary"] / profiles["TotalMonetary"].sum()

    strategy_table = profiles[["SegmentID", "Segment", "CustomerCount", "CustomerShare"]].copy()
    strategy_table["RecommendedStrategy"] = strategy_table["SegmentID"].map(strategies)
    return customers, profiles, strategy_table, mapping, names


def plot_model_selection(scores: pd.DataFrame, selected: pd.Series, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for covariance_type in COVARIANCE_TYPES:
        subset = scores[scores["covariance_type"] == covariance_type].sort_values("n_components")
        axes[0].plot(subset["n_components"], subset["bic"], marker="o", label=covariance_type)
        axes[1].plot(subset["n_components"], subset["aic"], marker="o", label=covariance_type)

    axes[0].scatter(
        selected["n_components"], selected["bic"], color="black", marker="*",
        s=180, label="Selected by BIC"
    )
    axes[1].scatter(
        selected["n_components"], selected["aic"], color="black", marker="*",
        s=180, label="Selected by BIC"
    )
    for axis, metric in zip(axes, ("BIC", "AIC")):
        axis.set_title(f"GMM {metric} (lower is better)")
        axis.set_xlabel("Number of components")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.legend(title="Covariance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca_segments(
    x_scaled: np.ndarray, customers: pd.DataFrame, path: Path, seed: int
) -> dict[str, float]:
    pca = PCA(n_components=2, random_state=seed)
    points = pca.fit_transform(x_scaled)
    plot_data = pd.DataFrame(
        {"PC1": points[:, 0], "PC2": points[:, 1], "Segment": customers["Segment"]}
    )
    fig, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.scatterplot(
        data=plot_data,
        x="PC1",
        y="PC2",
        hue="Segment",
        s=26,
        alpha=0.68,
        linewidth=0,
        palette="tab10",
        ax=axis,
    )
    axis.set_title("GMM customer segments projected with PCA")
    axis.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
    }


def plot_profile_heatmap(profiles: pd.DataFrame, path: Path) -> None:
    medians = profiles.set_index("Segment")[["RecencyMedian", "FrequencyMedian", "MonetaryMedian"]]
    logged = np.log1p(medians)
    standardized = (logged - logged.mean()) / logged.std(ddof=0).replace(0, 1)
    standardized["RecencyMedian"] *= -1
    standardized.columns = ["Recentness", "Frequency", "Monetary"]

    fig_height = max(5, 0.7 * len(standardized) + 2)
    fig, axis = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    sns.heatmap(
        standardized,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Relative segment profile (higher/greener is stronger)")
    axis.set_xlabel("")
    axis.set_ylabel("")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confidence(customers: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        data=customers,
        x="AssignmentConfidence",
        hue="Segment",
        bins=25,
        element="step",
        stat="density",
        common_norm=False,
        ax=axis,
    )
    axis.set_title("Largest GMM posterior probability by segment")
    axis.set_xlim(0, 1)
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    model_dir = output_dir / "model"
    for directory in (tables_dir, figures_dir, model_dir):
        directory.mkdir(exist_ok=True)

    input_path = args.input.resolve()
    rfm = load_rfm(input_path)
    if args.max_components > len(rfm):
        raise ValueError("--max-components cannot exceed the number of customers.")
    x_scaled, scaler, caps, _ = preprocess_rfm(rfm, args.winsor_quantile)
    scores = evaluate_candidates(
        x_scaled,
        args.min_components,
        args.max_components,
        args.seed,
        args.reg_covar,
    )
    selected = select_candidate(scores)
    model, raw_labels, probabilities = fit_final_model(
        x_scaled, selected, args.seed, args.reg_covar
    )
    final_counts = np.bincount(raw_labels, minlength=model.n_components)
    min_stable_count = max(20, int(np.ceil(len(rfm) * 0.01)))
    if final_counts.min() < min_stable_count:
        raise RuntimeError(
            "The final refit produced a segment below the minimum size; "
            "review the candidate grid or regularization."
        )
    final_silhouette = silhouette_score(
        x_scaled, raw_labels, sample_size=min(2_000, len(rfm)), random_state=args.seed
    )
    final_bic = float(model.bic(x_scaled))
    final_aic = float(model.aic(x_scaled))
    customers, profiles, strategies, mapping, names = build_outputs(
        rfm, raw_labels, probabilities
    )

    scores.to_csv(tables_dir / "model_selection.csv", index=False)
    customers.to_csv(tables_dir / "customer_segments.csv", index=False)
    profiles.to_csv(tables_dir / "segment_profiles.csv", index=False)
    strategies.to_csv(tables_dir / "segment_strategies.csv", index=False)

    plot_model_selection(scores, selected, figures_dir / "model_selection.png")
    pca_summary = plot_pca_segments(
        x_scaled, customers, figures_dir / "pca_segments.png", args.seed
    )
    plot_profile_heatmap(profiles, figures_dir / "segment_profile_heatmap.png")
    plot_confidence(customers, figures_dir / "assignment_confidence.png")

    selected_summary = {
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "customer_count": len(rfm),
        "features": FEATURES,
        "software_versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "preprocessing": {
            "winsor_quantile": args.winsor_quantile,
            "upper_caps": caps,
            "transformation": "log1p",
            "scaler": "StandardScaler",
        },
        "reg_covar": args.reg_covar,
        "selection_rule": (
            "Minimum BIC among converged candidates whose smallest cluster has "
            "at least max(20 customers, 1% of the dataset), with fixed reg_covar."
        ),
        "selected_model": {
            "n_components": int(selected["n_components"]),
            "covariance_type": str(selected["covariance_type"]),
            "selection_bic": float(selected["bic"]),
            "bic": final_bic,
            "aic": final_aic,
            "silhouette": float(final_silhouette),
            "calinski_harabasz": float(calinski_harabasz_score(x_scaled, raw_labels)),
            "davies_bouldin": float(davies_bouldin_score(x_scaled, raw_labels)),
            "smallest_segment": int(final_counts.min()),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
        },
        "posterior_diagnostics": {
            "max_probability_median": float(np.median(probabilities.max(axis=1))),
            "share_max_probability_over_0_99": float(
                np.mean(probabilities.max(axis=1) > 0.99)
            ),
        },
        "pca": pca_summary,
        "segment_names": {str(key): value for key, value in names.items()},
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(selected_summary, file, indent=2, ensure_ascii=False)

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "winsor_quantile": args.winsor_quantile,
        "reg_covar": args.reg_covar,
        "upper_caps": caps,
        "raw_cluster_to_segment_id": mapping,
        "segment_names": names,
        "selected_model": selected_summary["selected_model"],
    }
    joblib.dump(bundle, model_dir / "gmm_model.joblib")

    print(f"Customers: {len(rfm):,}")
    print(
        "Selected GMM: "
        f"components={int(selected['n_components'])}, "
        f"covariance={selected['covariance_type']}"
    )
    print(f"BIC: {final_bic:.2f}")
    print(f"Silhouette: {final_silhouette:.4f}")
    print(f"Outputs: {output_dir}")
    print("\nSegment profiles:")
    print(
        profiles[
            [
                "Segment",
                "CustomerCount",
                "CustomerShare",
                "RecencyMedian",
                "FrequencyMedian",
                "MonetaryMedian",
                "MeanConfidence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
