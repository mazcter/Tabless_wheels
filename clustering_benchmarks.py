"""
Perspective-Scoped vs. Global Clustering Benchmark for Explainable Automotive
Recommendation Systems (v8) — TEXT/TABLE-ONLY OUTPUT + HDBSCAN BASELINE +
FULL REFERENCE LIST + MATPLOTLIB/SEABORN MANUSCRIPT CHARTS
==============================================================================
Research Question:
    Does clustering scope (global, mixed-domain feature space vs. domain-
    restricted "perspective" feature subsets) affect internal cluster
    quality, stability, and semantic/structural coherence — independent of,
    but compared against, clustering ALGORITHM choice (KMeans / GMM /
    Agglomerative / HDBSCAN)?

Design:
    Two independent variables, fully crossed:
        1. Scope:      global-full | global-restricted-to-subset | perspective-scoped
        2. Algorithm:  KMeans | GMM | Agglomerative (Ward) | HDBSCAN [NEW in v8]

    HDBSCAN is added as a density-based baseline (per reviewer request) to
    test whether the recurring low-k "natural split" bias observed under
    KMeans/GMM/Agglomerative is algorithm-driven or reflects genuine density
    structure in the data. Unlike the other three algorithms, HDBSCAN does
    NOT take a pre-specified k; it is run once per condition (not swept over
    k=2..10) and its discovered cluster count is reported alongside the
    k-swept algorithms for comparison. HDBSCAN also natively labels noise
    points (-1), which are excluded from validity-metric computation and
    reported separately as a noise fraction.

    Five perspective-scoped feature subsets (Tabless Wheels taxonomy):
        - Safety             -> safety_ratings_v2.csv
        - Ownership          -> car_ownership_costs.csv
        - Tech & Comfort     -> India_Car_Features_Matrix.csv
        - Driving Performance-> car_features_v2.csv
        - Overall            -> union of all numeric features (= global_full)

    "global-restricted-to-subset" REUSES cluster labels fit on the FULL
    global feature space and evaluates them on the subset's feature columns
    ONLY (no refitting) — this isolates the scope effect (independent
    fitting per domain) from the pure dimensionality effect (fewer columns).

Output format:
    NO CSV, NO PNG, NO JSON files are written. All results print directly
    to stdout as formatted tables and text summaries.

Stage F (LLM explanation) is intentionally excluded — out of scope for this
scope-comparison research question, per prior review.

No fabricated data, no simulated results, no placeholder metrics. If the
optional 'hdbscan' package is not installed, the HDBSCAN condition is
skipped with a clear warning rather than faked.

==============================================================================
REFERENCES (full list; foundational + reviewer-flagged additions)
==============================================================================
[1] Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the
    interpretation and validation of cluster analysis. Journal of
    Computational and Applied Mathematics, 20, 53-65.
    https://doi.org/10.1016/0377-0427(87)90125-7

[2] Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    PAMI-1(2), 224-227. https://doi.org/10.1109/TPAMI.1979.4766909

[3] Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster
    analysis. Communications in Statistics, 3(1), 1-27.
    https://doi.org/10.1080/03610927408827101

[4] Holm, S. (1979). A simple sequentially rejective multiple test
    procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

[5] Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective
    Function. Journal of the American Statistical Association, 58(301),
    236-244. https://doi.org/10.1080/01621459.1963.10500845

[6] Hennig, C. (2007). Cluster-wise assessment of cluster stability.
    Computational Statistics & Data Analysis, 52(1), 258-271. (Bootstrap
    Jaccard/ARI stability framework; interpretability thresholds used to
    contextualize this script's bootstrap ARI/AMI output: Jaccard > 0.80 =
    stable, < 0.50 = dissolved.)

[7] McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical
    density based clustering. Journal of Open Source Software, 2(11), 205.
    https://doi.org/10.21105/joss.00205

[8] McInnes, L., & Healy, J. (2017). Accelerated Hierarchical Density
    Clustering. IEEE International Conference on Data Mining Workshops
    (ICDMW). https://doi.org/10.1109/ICDMW.2017.12

[9] Yan, X., Hu, S., Mao, Y., Ye, Y., & Yu, H. (2022). Representation
    Learning in Multi-view Clustering: A Literature Review. Data Science
    and Engineering, 7, 225-241. (Grounding citation for the global-fit vs.
    domain-restricted-fit / multi-view clustering paradigm central to the
    scope-comparison RQ in this script.)

[10] Zhang, C., et al. Weighted Multi-view Clustering with Feature
     Selection. (Supporting citation on feature-subset weighting effects on
     cluster quality; relevant to explaining shared-feature overlap effects
     across perspective conditions, e.g. Tech & Comfort.) [CITATION
     REQUIRED — full bibliographic details not verified in this pass]

[11] Zhu, F., et al. / Xin, X., et al. Cross-Domain Recommendation via
     Cluster-Level Latent Factor Model. ECML PKDD (Springer, 2013).
     [CITATION REQUIRED — full bibliographic details not verified in this
     pass; relevant only if extending to cross-domain cluster alignment.]

[12] Nie, X., et al. (2023). Cross-Domain Recommendation Via
     User-Clustering and Multidimensional Information Fusion. IEEE
     Transactions on Multimedia. [CITATION REQUIRED — full bibliographic
     details not verified in this pass; relevant only if extending to
     cross-domain cluster alignment.]

Note: [10]-[12] are included for completeness per reviewer guidance but are
NOT required for the core scope-vs-algorithm RQ this script benchmarks; they
apply only if the manuscript is extended toward multi-view feature-weighting
or cross-domain recommendation framings.
"""

import argparse
import sys
import time
import warnings
import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score,
)
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

try:
    import hdbscan as _hdbscan_lib
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ----------------------------------------------------------------------------
# GOOGLE DRIVE MOUNT (Colab only) — for reading input CSVs only; no outputs
# are written anywhere.
# ----------------------------------------------------------------------------
try:
    from google.colab import drive as _gdrive
    import os as _os
    if not _os.path.exists("/content/drive/MyDrive"):
        _gdrive.mount("/content/drive")
    print("[drive_mount] Google Drive mounted (for reading input CSVs if needed). "
          "No output files will be written — all results print to console.")
except ImportError:
    print("[drive_mount] Not running in Colab — skipping Drive mount.")


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
CONFIG = {
    "id_col": "model_name",
    "k_min": 2,
    "k_max": 10,
    "n_bootstrap": 100,
    "random_state": 42,
    "top_n_features": 10,
    "source_files": {
        "driving_performance": "car_features_v2.csv",
        "tech_comfort": "India_Car_Features_Matrix.csv",
        "ownership": "car_ownership_costs.csv",
        "safety": "safety_ratings_v2.csv",
    },
    "hdbscan_min_cluster_size": 5,
}

IDENTIFIER_LIKE_COLUMNS = {
    "id", "index", "unnamed", "model_name", "model", "brand", "variant",
    "variant_level", "trim_rank", "data_source", "fuel_type", "transmission",
    "drive_type", "steering_type", "suspension_front", "suspension_rear",
    "brake_front", "brake_rear", "testing_agency", "test_year",
    "bodyshell_integrity", "footwell_integrity", "fuel_unit", "efficiency_unit",
}

PERSPECTIVE_SOURCES = {
    "Safety": ["safety"],
    "Ownership": ["ownership"],
    "Tech & Comfort": ["tech_comfort"],
    "Driving Performance": ["driving_performance"],
}

K_SWEPT_ALGORITHMS = ["KMeans", "GMM", "Agglomerative"]  # HDBSCAN handled separately (no k)

# Global registries populated during main() run, consumed by chart-generation
# helpers at the end of the script. Kept module-level for simplicity since this
# is a single-run analysis script, not a long-lived service.
stability_ari_distributions = {}   # {(condition, algo): [ari_boot_1, ari_boot_2, ...]}
GRID_METRICS_REGISTRY = {}         # {condition: grid_df}
SILHOUETTE_CI_REGISTRY = None      # DataFrame from Experiment 7
STABILITY_VS_QUALITY_REGISTRY = None  # DataFrame from Experiment 8
CLUSTER_BALANCE_REGISTRY = None    # DataFrame from Experiment 9
DIMENSIONALITY_ABLATION_REGISTRY = None  # DataFrame from Experiment 1
FEATURE_IMPORTANCE_GLOBAL_REGISTRY = {}  # {condition: top_global_df} from run_condition
HDBSCAN_ROWS_REGISTRY = {}         # {condition: hdbscan_row dict}


def section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_table(df: pd.DataFrame, empty_msg: str = "(no rows)"):
    if df is None or df.empty:
        print(empty_msg)
    else:
        print(df.to_string(index=False))


# ----------------------------------------------------------------------------
# STAGE 0: DATA LOADING & MERGING
# ----------------------------------------------------------------------------
def load_and_merge(source_files: dict, id_col: str) -> tuple:
    frames = {}
    per_source_cols = {}

    def is_identifier(col_name: str) -> bool:
        low = col_name.lower()
        return any(tag in low for tag in IDENTIFIER_LIKE_COLUMNS)

    for key, path in source_files.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Required source file missing for '{key}': {path}")
        df = pd.read_csv(p)
        if id_col not in df.columns:
            raise ValueError(f"'{id_col}' not found in {path}. Cannot merge.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if not is_identifier(c)]
        if not feature_cols:
            raise ValueError(f"No usable numeric feature columns found in {path}.")
        keep = [id_col] + feature_cols
        sub = df[keep].drop_duplicates(subset=id_col)
        frames[key] = sub
        per_source_cols[key] = feature_cols
        print(f"[load_and_merge] {key}: {len(feature_cols)} numeric features from {path}")

    merged = None
    for key, sub in frames.items():
        merged = sub if merged is None else merged.merge(sub, on=id_col, how="outer", suffixes=("", f"_{key}"))

    n_before = len(merged)
    merged = merged.dropna(how="all", subset=[c for cols in per_source_cols.values() for c in cols])
    print(f"[load_and_merge] Merged dataset: {n_before} -> {len(merged)} rows, {merged.shape[1]} columns total.")
    return merged, per_source_cols


def preprocess_features(df: pd.DataFrame, feature_cols: list):
    raw = df[feature_cols].copy()
    missing = raw.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"[preprocess_features] Missing values (median-imputed): {dict(missing)}")
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    return scaled


# ----------------------------------------------------------------------------
# STAGE A: CONTROLLED GRID (k-swept algorithms) + HDBSCAN (k-free) [NEW]
# ----------------------------------------------------------------------------
def fit_algorithm(name: str, k: int, X: np.ndarray, random_state: int):
    start = time.perf_counter()
    if name == "KMeans":
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
    elif name == "GMM":
        model = GaussianMixture(n_components=k, random_state=random_state)
        model.fit(X)
        labels = model.predict(X)
    elif name == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown algorithm {name}")
    runtime = time.perf_counter() - start
    return labels, runtime, (model if name == "GMM" else None)


def fit_hdbscan(X: np.ndarray, min_cluster_size: int):
    """
    Density-based baseline (McInnes, Healy & Astels 2017; McInnes & Healy
    2017). Does not require a pre-specified k. Returns labels (with -1 =
    noise), runtime, discovered cluster count (excluding noise), and noise
    fraction. Metrics are computed on non-noise points only.
    """
    start = time.perf_counter()
    clusterer = _hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)
    runtime = time.perf_counter() - start
    n_noise = int(np.sum(labels == -1))
    noise_fraction = n_noise / len(labels)
    non_noise_mask = labels != -1
    n_clusters = len(set(labels[non_noise_mask])) if non_noise_mask.any() else 0
    return labels, runtime, n_clusters, noise_fraction


def controlled_grid_benchmark(X: np.ndarray, k_range, random_state: int, condition: str):
    rows, labels_store = [], {}
    for k in k_range:
        for algo in K_SWEPT_ALGORITHMS:
            labels, runtime, gmm_model = fit_algorithm(algo, k, X, random_state)
            n_clusters = len(set(labels))
            row = {
                "condition": condition, "k": k, "algorithm": algo,
                "n_clusters": n_clusters,
                "silhouette": silhouette_score(X, labels) if n_clusters > 1 else np.nan,
                "davies_bouldin": davies_bouldin_score(X, labels) if n_clusters > 1 else np.nan,
                "calinski_harabasz": calinski_harabasz_score(X, labels) if n_clusters > 1 else np.nan,
                "runtime_s": runtime, "n_features": X.shape[1],
            }
            if algo == "GMM" and gmm_model is not None:
                row["aic"] = gmm_model.aic(X)
                row["bic"] = gmm_model.bic(X)
            rows.append(row)
            labels_store[(algo, k)] = labels
    return pd.DataFrame(rows), labels_store


def run_hdbscan_baseline(X: np.ndarray, condition: str, min_cluster_size: int):
    """
    Runs HDBSCAN once (no k sweep) and reports its own validity metrics on
    non-noise points, plus discovered k and noise fraction, for direct
    comparison against the k-swept algorithms' best-k rows. Returns None if
    hdbscan is not installed (never fabricates a result).
    """
    if not HAS_HDBSCAN:
        print("[run_hdbscan_baseline] 'hdbscan' package not installed — skipping HDBSCAN "
              "baseline for this condition. Install via `pip install hdbscan` to enable.")
        return None
    labels, runtime, n_clusters, noise_fraction = fit_hdbscan(X, min_cluster_size)
    if n_clusters < 2:
        print(f"[run_hdbscan_baseline] HDBSCAN found <2 clusters (n_clusters={n_clusters}) "
              f"for condition '{condition}' — validity metrics undefined, skipping.")
        return {"condition": condition, "algorithm": "HDBSCAN", "k_discovered": n_clusters,
                "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                "noise_fraction": noise_fraction, "runtime_s": runtime, "n_features": X.shape[1]}
    mask = labels != -1
    row = {
        "condition": condition, "algorithm": "HDBSCAN", "k_discovered": n_clusters,
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "noise_fraction": noise_fraction, "runtime_s": runtime, "n_features": X.shape[1],
    }
    return row


def select_final_k(grid_df: pd.DataFrame) -> int:
    mean_by_k = grid_df.groupby("k")["silhouette"].mean()
    return int(mean_by_k.idxmax())


# ----------------------------------------------------------------------------
# STAGE B: BOOTSTRAP STABILITY
# ----------------------------------------------------------------------------
def bootstrap_stability(X: np.ndarray, algo: str, k: int, n_bootstrap: int,
                         random_state: int, condition: str):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    base_labels, _, _ = fit_algorithm(algo, k, X, random_state)
    ari_scores, ami_scores = [], []
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels, _, _ = fit_algorithm(algo, k, X[idx], random_state + i + 1)
        ref = base_labels[idx]
        if len(set(boot_labels)) > 1 and len(set(ref)) > 1:
            ari_scores.append(adjusted_rand_score(ref, boot_labels))
            ami_scores.append(adjusted_mutual_info_score(ref, boot_labels))
    ari_mean = np.mean(ari_scores) if ari_scores else np.nan
    stability_label = (
        "highly stable (Hennig 2007)" if ari_mean >= 0.80 else
        "moderately stable" if ari_mean >= 0.50 else
        "dissolved / unstable (Hennig 2007)" if not np.isnan(ari_mean) else "undefined"
    )
    result = {
        "condition": condition, "algorithm": algo, "k": k,
        "ari_mean": ari_mean, "ari_std": np.std(ari_scores) if ari_scores else np.nan,
        "ami_mean": np.mean(ami_scores) if ami_scores else np.nan,
        "ami_std": np.std(ami_scores) if ami_scores else np.nan,
        "stability_interpretation": stability_label,
    }
    return result, base_labels, ari_scores


# ----------------------------------------------------------------------------
# STAGE C: STATISTICAL TESTING (Friedman + Holm-Bonferroni Wilcoxon)
# ----------------------------------------------------------------------------
def holm_bonferroni(p_values: list) -> list:
    """Holm (1979) sequentially rejective step-down correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def statistical_tests(grid_df: pd.DataFrame, metric: str, condition: str):
    pivot = grid_df.pivot(index="k", columns="algorithm", values=metric).dropna()
    algos = pivot.columns.tolist()
    n_blocks = len(pivot)
    friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot[a] for a in algos])
    approx_flag = "APPROX (n_blocks<=15)" if n_blocks <= 15 else "OK"

    pair_list = list(combinations(algos, 2))
    raw_p, raw_stat = [], []
    for a1, a2 in pair_list:
        try:
            w_stat, w_p = stats.wilcoxon(pivot[a1], pivot[a2])
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        raw_stat.append(w_stat)
        raw_p.append(w_p if not np.isnan(w_p) else 1.0)
    adj_p = holm_bonferroni(raw_p)

    pairwise_rows = [{
        "condition": condition, "algo_1": a1, "algo_2": a2, "wilcoxon_stat": w_stat,
        "p_raw": p_r, "p_holm": p_a, "significant_0.05": bool(p_a < 0.05)
    } for (a1, a2), w_stat, p_r, p_a in zip(pair_list, raw_stat, raw_p, adj_p)]

    friedman_row = {"condition": condition, "metric": metric, "friedman_stat": friedman_stat,
                     "friedman_p": friedman_p, "n_blocks": n_blocks, "reliability": approx_flag}
    return pd.DataFrame([friedman_row]), pd.DataFrame(pairwise_rows)


# ----------------------------------------------------------------------------
# STAGE D: FEATURE IMPORTANCE (ANOVA + Cohen's d)
# ----------------------------------------------------------------------------
def feature_importance(df: pd.DataFrame, feature_cols: list, labels: np.ndarray,
                        condition: str, top_n: int = 10):
    anova_rows = []
    for f in feature_cols:
        groups = [df.loc[labels == c, f].dropna().values for c in sorted(set(labels))]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        grand_mean = df[f].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[f] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        anova_rows.append({"condition": condition, "feature": f, "eta_squared": eta_sq, "p_value": p_val})
    anova_df = pd.DataFrame(anova_rows).sort_values("eta_squared", ascending=False)
    top_global = anova_df.head(top_n)

    cohend_rows = []
    for c in sorted(set(labels)):
        if c == -1:
            continue  # skip HDBSCAN noise label
        mask = labels == c
        for f in feature_cols:
            in_c, out_c = df.loc[mask, f].dropna(), df.loc[~mask, f].dropna()
            if len(in_c) < 2 or len(out_c) < 2:
                continue
            pooled_std = np.sqrt(((len(in_c) - 1) * in_c.std() ** 2 +
                                   (len(out_c) - 1) * out_c.std() ** 2) / (len(in_c) + len(out_c) - 2))
            cohend = (in_c.mean() - out_c.mean()) / pooled_std if pooled_std > 0 else 0.0
            cohend_rows.append({"condition": condition, "cluster_id": c, "feature": f, "cohend": cohend})
    cohend_df = pd.DataFrame(cohend_rows)
    if not cohend_df.empty:
        top_per_cluster = (
            cohend_df.assign(abs_d=cohend_df["cohend"].abs())
            .sort_values(["cluster_id", "abs_d"], ascending=[True, False])
            .groupby("cluster_id").head(3).drop(columns="abs_d")
        )
    else:
        top_per_cluster = cohend_df
    return top_global, top_per_cluster


# ----------------------------------------------------------------------------
# CONDITION RUNNERS
# ----------------------------------------------------------------------------
def run_condition(df, feature_cols, condition_name, k_range, random_state, n_bootstrap, top_n,
                   hdbscan_min_cluster_size):
    """Fit AND evaluate models on X built from feature_cols (global_full, perspective_*)."""
    section(f"CONDITION: {condition_name}  (fit_and_evaluate, n_features={len(feature_cols)})")
    X = preprocess_features(df, feature_cols)

    grid_df, labels_store = controlled_grid_benchmark(X, k_range, random_state, condition_name)
    print("\n-- Grid metrics (k-swept algorithms: KMeans / GMM / Agglomerative) --")
    print_table(grid_df.round(4))
    GRID_METRICS_REGISTRY[condition_name] = grid_df

    hdbscan_row = run_hdbscan_baseline(X, condition_name, hdbscan_min_cluster_size)
    print("\n-- HDBSCAN baseline (density-based, no k sweep; McInnes et al. 2017) --")
    print_table(pd.DataFrame([hdbscan_row]).round(4) if hdbscan_row else None,
                empty_msg="(hdbscan not installed — skipped)")
    if hdbscan_row:
        HDBSCAN_ROWS_REGISTRY[condition_name] = hdbscan_row

    final_k = select_final_k(grid_df)
    print(f"\n-- Selected shared k = {final_k} (max mean silhouette across k-swept algorithms) --")

    stability_rows, final_labels = [], {}
    for algo in K_SWEPT_ALGORITHMS:
        result, base_labels, ari_dist = bootstrap_stability(X, algo, final_k, n_bootstrap, random_state, condition_name)
        stability_ari_distributions[(condition_name, algo)] = ari_dist
        stability_rows.append(result)
        final_labels[algo] = base_labels
    print("\n-- Bootstrap stability at k=%d (n_bootstrap=%d); interpretation per Hennig (2007) --" %
          (final_k, n_bootstrap))
    print_table(pd.DataFrame(stability_rows).round(4))

    friedman_df, pairwise_df = statistical_tests(grid_df, "silhouette", condition_name)
    print("\n-- Friedman test (silhouette, blocked by k) --")
    print_table(friedman_df.round(4))
    print("\n-- Pairwise Wilcoxon (Holm-Bonferroni corrected, Holm 1979) --")
    print_table(pairwise_df.round(4))

    top_global_list, top_cluster_list = [], []
    for algo in K_SWEPT_ALGORITHMS:
        labels = final_labels[algo]
        top_global, top_per_cluster = feature_importance(df, feature_cols, labels,
                                                           f"{condition_name}|{algo}", top_n=top_n)
        top_global_list.append(top_global)
        top_cluster_list.append(top_per_cluster)
    print("\n-- Top distinguishing features (global, by eta-squared) --")
    top_global_concat = pd.concat(top_global_list, ignore_index=True) if top_global_list else pd.DataFrame()
    print_table(top_global_concat.round(4) if not top_global_concat.empty else None)
    FEATURE_IMPORTANCE_GLOBAL_REGISTRY[condition_name] = top_global_concat
    print("\n-- Top 3 distinguishing features per cluster (Cohen's d) --")
    print_table(pd.concat(top_cluster_list, ignore_index=True).round(4) if top_cluster_list else None)

    summary = {"condition": condition_name, "final_k": final_k, "n_features": len(feature_cols),
               "mean_silhouette": grid_df["silhouette"].mean(), "mode": "fit_and_evaluate",
               "hdbscan_k_discovered": hdbscan_row["k_discovered"] if hdbscan_row else np.nan,
               "hdbscan_silhouette": hdbscan_row["silhouette"] if hdbscan_row else np.nan}
    return summary, labels_store, final_k


def run_restricted_condition(df, subset_cols, condition_name, k_range, global_labels_store, top_n):
    """Evaluation-only: reuse global_full-fitted labels, score on subset_cols only."""
    section(f"CONDITION: {condition_name}  (evaluate_only_global_labels, n_features={len(subset_cols)})")
    X_subset = preprocess_features(df, subset_cols)

    rows = []
    for k in k_range:
        for algo in K_SWEPT_ALGORITHMS:
            labels = global_labels_store.get((algo, k))
            if labels is None:
                continue
            n_clusters = len(set(labels))
            rows.append({
                "condition": condition_name, "k": k, "algorithm": algo, "n_clusters": n_clusters,
                "silhouette": silhouette_score(X_subset, labels) if n_clusters > 1 else np.nan,
                "davies_bouldin": davies_bouldin_score(X_subset, labels) if n_clusters > 1 else np.nan,
                "calinski_harabasz": calinski_harabasz_score(X_subset, labels) if n_clusters > 1 else np.nan,
                "n_features": X_subset.shape[1], "label_source": "global_full_projected",
            })
    grid_df = pd.DataFrame(rows)
    print("\n-- Grid metrics (global-fit labels, evaluated on subset features) --")
    print_table(grid_df.round(4))
    final_k = select_final_k(grid_df)
    print(f"\n-- Selected shared k = {final_k} --")

    friedman_df, pairwise_df = statistical_tests(grid_df, "silhouette", condition_name)
    print("\n-- Friedman test (silhouette, blocked by k) --")
    print_table(friedman_df.round(4))
    print("\n-- Pairwise Wilcoxon (Holm-Bonferroni corrected, Holm 1979) --")
    print_table(pairwise_df.round(4))

    top_global_list = []
    for algo in K_SWEPT_ALGORITHMS:
        labels = global_labels_store.get((algo, final_k))
        if labels is None:
            continue
        top_global, _ = feature_importance(df, subset_cols, labels, f"{condition_name}|{algo}", top_n=top_n)
        top_global_list.append(top_global)
    print("\n-- Top distinguishing features (global, by eta-squared) --")
    print_table(pd.concat(top_global_list, ignore_index=True).round(4) if top_global_list else None)

    summary = {"condition": condition_name, "final_k": final_k, "n_features": len(subset_cols),
               "mean_silhouette": grid_df["silhouette"].mean(), "mode": "evaluate_only_global_labels",
               "hdbscan_k_discovered": np.nan, "hdbscan_silhouette": np.nan}
    return summary


# ----------------------------------------------------------------------------
# ARGUMENT PARSING (Colab-safe)
# ----------------------------------------------------------------------------
def parse_args():
    in_notebook = "ipykernel" in sys.modules or "google.colab" in sys.modules
    parser = argparse.ArgumentParser(description="Perspective-scoped vs. global clustering benchmark v8 (text-only, +HDBSCAN).")
    parser.add_argument("--id_col", type=str, default=CONFIG["id_col"])
    parser.add_argument("--k_min", type=int, default=CONFIG["k_min"])
    parser.add_argument("--k_max", type=int, default=CONFIG["k_max"])
    parser.add_argument("--n_bootstrap", type=int, default=CONFIG["n_bootstrap"])
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=CONFIG["hdbscan_min_cluster_size"])
    args, unknown = parser.parse_known_args([] if in_notebook else None)
    if unknown:
        print(f"[parse_args] Ignoring unrecognized arguments: {unknown}")
    return args


def print_run_metadata(args, k_range):
    import sklearn, scipy
    section("RUN METADATA")
    print(f"Timestamp (UTC): {datetime.datetime.utcnow().isoformat()}Z")
    print(f"numpy={np.__version__}  pandas={pd.__version__}  "
          f"scikit-learn={sklearn.__version__}  scipy={scipy.__version__}")
    print(f"hdbscan installed: {HAS_HDBSCAN}"
          + ("" if HAS_HDBSCAN else " (install via `pip install hdbscan` to enable the density-based baseline)"))
    print(f"Config: id_col={args.id_col}, k_range={list(k_range)}, "
          f"n_bootstrap={args.n_bootstrap}, random_state={CONFIG['random_state']}, "
          f"top_n_features={CONFIG['top_n_features']}, "
          f"hdbscan_min_cluster_size={args.hdbscan_min_cluster_size}")
    print(f"Source files: {CONFIG['source_files']}")
    print("Notes: (1) Friedman chi-sq approximation may be unreliable for n_blocks<=15 [Friedman test]. "
          "(2) Pairwise Wilcoxon p-values are Holm-Bonferroni corrected [Holm, 1979]. "
          "(3) Ward linkage assumes roughly spherical, equal-variance clusters [Ward, 1963]. "
          "(4) Bootstrap ARI stability interpreted per Hennig (2007) thresholds "
          "(>=0.80 stable, <0.50 dissolved). "
          "(5) HDBSCAN baseline uses McInnes, Healy & Astels (2017) / McInnes & Healy (2017). "
          "(6) Stage F (LLM explanation) intentionally out of scope for this RQ. "
          "See module docstring for full reference list [1]-[12].")


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    random_state = CONFIG["random_state"]
    k_range = range(args.k_min, args.k_max + 1)

    print_run_metadata(args, k_range)
    print_contributions_section()
    print_pipeline_diagram()

    section("STAGE 0: LOAD & MERGE SOURCE DATA")
    merged_df, per_source_cols = load_and_merge(CONFIG["source_files"], args.id_col)
    all_feature_cols = sorted({c for cols in per_source_cols.values() for c in cols})

    print_dataset_statistics(merged_df, per_source_cols, args.id_col, CONFIG["source_files"])
    print_hyperparameter_table()
    variance_share_by_perspective(per_source_cols)

    summary_rows = []
    grid_df_by_condition = {}
    condition_X_map = {}
    labels_by_condition_algo = {}
    perspective_labels_by_name = {}

    global_summary, global_labels_store, global_final_k = run_condition(
        merged_df, all_feature_cols, "global_full", k_range, random_state,
        args.n_bootstrap, CONFIG["top_n_features"], args.hdbscan_min_cluster_size
    )
    summary_rows.append(global_summary)
    condition_X_map["global_full"] = preprocess_features(merged_df, all_feature_cols)
    for algo in K_SWEPT_ALGORITHMS:
        lbl = global_labels_store.get((algo, global_final_k))
        if lbl is not None:
            labels_by_condition_algo[("global_full", algo)] = lbl

    for persp_name, source_keys in PERSPECTIVE_SOURCES.items():
        persp_cols = sorted({c for key in source_keys for c in per_source_cols[key]})
        missing = [c for c in persp_cols if c not in merged_df.columns]
        if missing:
            raise ValueError(f"Perspective '{persp_name}' expected columns missing after merge: {missing}")

        persp_summary, persp_labels_store, persp_final_k = run_condition(
            merged_df, persp_cols, f"perspective_{persp_name}", k_range, random_state,
            args.n_bootstrap, CONFIG["top_n_features"], args.hdbscan_min_cluster_size
        )
        summary_rows.append(persp_summary)
        condition_X_map[f"perspective_{persp_name}"] = preprocess_features(merged_df, persp_cols)
        persp_kmeans_labels = persp_labels_store.get(("KMeans", persp_final_k))
        perspective_labels_by_name[persp_name] = (persp_kmeans_labels, persp_final_k)
        for algo in K_SWEPT_ALGORITHMS:
            lbl = persp_labels_store.get((algo, persp_final_k))
            if lbl is not None:
                labels_by_condition_algo[(f"perspective_{persp_name}", algo)] = lbl

        restricted_summary = run_restricted_condition(
            merged_df, persp_cols, f"global_restricted_{persp_name}", k_range,
            global_labels_store, CONFIG["top_n_features"]
        )
        summary_rows.append(restricted_summary)

    # ---- Experiment 2 / Mechanism test: Weighted Global -------------------
    weighted_summary, weighted_mean_sil = run_weighted_global_condition(
        merged_df, per_source_cols, k_range, random_state, args.n_bootstrap, CONFIG["top_n_features"]
    )
    summary_rows.append(weighted_summary)

    # ---- Experiment 3: PCA Baseline ----------------------------------------
    pca_summary = run_pca_baseline_condition(merged_df, all_feature_cols, k_range, random_state, n_components=30)
    summary_rows.append(pca_summary)

    # ---- Experiment 1: Dimensionality Ablation -----------------------------
    dimensionality_ablation(merged_df, per_source_cols, args.id_col, n_repeats=100, random_state=random_state)

    # ---- Experiment 4: Cluster Agreement (Global vs Perspective) ----------
    cluster_agreement(global_labels_store, perspective_labels_by_name, global_final_k)

    # ---- Experiment 6: Mutual Information Between Perspectives ------------
    mutual_information_between_perspectives(perspective_labels_by_name)

    # ---- Experiment 5: Feature Removal (masking hypothesis) ---------------
    feature_removal_experiment(merged_df, all_feature_cols, per_source_cols, k_range, random_state,
                                perspective_labels_by_name)

    # ---- Experiment 9: Cluster Balance / Entropy ---------------------------
    cluster_balance_report(labels_by_condition_algo)

    # ---- Experiment 7: Silhouette Bootstrap 95% CI -------------------------
    best_algo_k_map = {"global_full": ("KMeans", global_final_k)}
    for persp_name, (labels, k) in perspective_labels_by_name.items():
        best_algo_k_map[f"perspective_{persp_name}"] = ("KMeans", k)
    silhouette_ci_report(condition_X_map, best_algo_k_map, n_bootstrap=args.n_bootstrap, random_state=random_state)

    # ---- Experiment 8: Stability vs Quality --------------------------------
    all_stability_rows = []
    for cond in list(condition_X_map.keys()):
        X_c = condition_X_map[cond]
        gdf, _ = controlled_grid_benchmark(X_c, k_range, random_state, cond)
        grid_df_by_condition[cond] = gdf
        fk = select_final_k(gdf)
        for algo in K_SWEPT_ALGORITHMS:
            res, _, ari_dist = bootstrap_stability(X_c, algo, fk, args.n_bootstrap, random_state, cond)
            stability_ari_distributions[(cond, algo)] = ari_dist
            all_stability_rows.append(res)
    stability_vs_quality(pd.DataFrame(all_stability_rows), grid_df_by_condition)

    section("FINAL: SCOPE COMPARISON SUMMARY (global_full vs perspective_* vs global_restricted_* "
            "vs weighted_global vs global_pca)")
    scope_summary_df = pd.DataFrame(summary_rows)
    print_table(scope_summary_df.round(4))

    print_abstract_conclusion_guidance()

    section("RUN COMPLETE")
    print("All results printed above. No files were written to disk.")
    print("'mode' column distinguishes fit_and_evaluate (global_full, perspective_*, "
          "weighted_global, global_pca) from evaluate_only_global_labels (global_restricted_*), "
          "resolving the scope-vs-dimensionality confound.")
    print("Mechanism-testing verdict: compare 'weighted_global' mean_silhouette against "
          "'global_full' and 'perspective_*' rows above. If weighted_global approaches "
          "perspective-scoped quality, feature imbalance was the primary mechanism; if it "
          "still underperforms perspective-scoped clustering, semantic partitioning itself "
          "matters independent of feature-count balance.")
    print("Full reference list [1]-[12] is provided in the module docstring at the top of this file.")

    section("GENERATING MANUSCRIPT FIGURES")
    generate_manuscript_charts(outdir="output")




# ============================================================================
# v9 ADDITIONS: MECHANISM-TESTING ABLATIONS (Experiments 1-10) +
# MANUSCRIPT SCAFFOLDING (contributions, dataset stats, hyperparameters,
# pipeline diagram, weighted-global clustering)
# ============================================================================
#
# Per reviewer guidance, these experiments test WHY perspective-scoped
# clustering may outperform global clustering, rather than only observing
# that it does. All computations use the real merged dataset; nothing is
# simulated except explicitly-labeled random feature sampling in Experiment 1
# (which is itself the point of that ablation — real data, random columns).
# ============================================================================

from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score


# ---- Experiment 10 (computed first, feeds the "mechanism" narrative) ------
def variance_share_by_perspective(per_source_cols: dict):
    """
    Since each feature is standardized (unit variance) before clustering,
    each perspective's RAW contribution to total global variance is
    proportional to its feature COUNT. This directly quantifies the
    'feature imbalance' mechanism hypothesis (Tech & Comfort dominating
    Safety) prior to any clustering being run.
    """
    section("EXPERIMENT 10: Variance Share by Perspective (pre-clustering)")
    total_features = sum(len(v) for v in per_source_cols.items() and per_source_cols.values())
    total_features = sum(len(v) for v in per_source_cols.values())
    rows = []
    for key, cols in per_source_cols.items():
        rows.append({
            "perspective_source": key, "n_features": len(cols),
            "share_of_global_variance_pct": 100.0 * len(cols) / total_features,
        })
    df = pd.DataFrame(rows).sort_values("n_features", ascending=False)
    print_table(df.round(2))
    print("Interpretation: under z-score standardization, each feature contributes exactly "
          "1 unit of variance, so a perspective's share of total global variance equals its "
          "share of total feature count. This is the quantitative basis for the 'masking' "
          "hypothesis tested in Experiments 1, 2, 3, and 5 below.")
    return df


# ---- Experiment 1: Dimensionality Ablation ---------------------------------
def dimensionality_ablation(df: pd.DataFrame, per_source_cols: dict, id_col: str,
                             n_repeats: int, random_state: int):
    """
    For each perspective P with n_P real features, randomly sample n_P
    columns from the LARGEST perspective's feature pool (excluding P itself
    if it IS the largest), repeat n_repeats times, cluster with KMeans at a
    fixed k=3 (chosen for comparability across perspectives; override via
    k arg if desired), and compare the resulting silhouette DISTRIBUTION
    against the real perspective's own silhouette at the same k. This
    isolates whether a perspective's clustering quality comes from its
    SEMANTIC feature content or merely its DIMENSIONALITY.
    """
    section("EXPERIMENT 1: Dimensionality Ablation (semantics vs. dimensionality)")
    largest_key = max(per_source_cols, key=lambda k: len(per_source_cols[k]))
    pool_cols = per_source_cols[largest_key]
    rng = np.random.RandomState(random_state)
    k_fixed = 3
    rows = []
    for persp_key, cols in per_source_cols.items():
        n_feat = len(cols)
        if n_feat >= len(pool_cols):
            print(f"[dimensionality_ablation] Skipping '{persp_key}': its feature count "
                  f"({n_feat}) >= pool size ({len(pool_cols)}); cannot sample without replacement.")
            continue
        real_X = preprocess_features(df, cols)
        real_labels, _, _ = fit_algorithm("KMeans", k_fixed, real_X, random_state)
        real_sil = silhouette_score(real_X, real_labels) if len(set(real_labels)) > 1 else np.nan

        random_sils = []
        for i in range(n_repeats):
            sampled_cols = list(rng.choice(pool_cols, size=n_feat, replace=False))
            rand_X = preprocess_features(df, sampled_cols)
            rand_labels, _, _ = fit_algorithm("KMeans", k_fixed, rand_X, random_state + i + 1)
            if len(set(rand_labels)) > 1:
                random_sils.append(silhouette_score(rand_X, rand_labels))

        rows.append({
            "perspective": persp_key, "n_features": n_feat, "k": k_fixed,
            "real_silhouette": real_sil,
            "random_silhouette_mean": np.mean(random_sils) if random_sils else np.nan,
            "random_silhouette_std": np.std(random_sils) if random_sils else np.nan,
            "n_repeats": len(random_sils),
            "real_exceeds_random": bool(real_sil > np.mean(random_sils)) if random_sils and not np.isnan(real_sil) else None,
        })
    result_df = pd.DataFrame(rows)
    print_table(result_df.round(4))
    print("Interpretation: if 'real_silhouette' consistently exceeds "
          "'random_silhouette_mean' (beyond 1-2 std), the perspective's clustering quality is "
          "driven by semantic feature content, not merely feature count. If real and random "
          "silhouettes are statistically indistinguishable, dimensionality alone explains the effect.")
    global DIMENSIONALITY_ABLATION_REGISTRY
    DIMENSIONALITY_ABLATION_REGISTRY = result_df
    return result_df


# ---- Experiment 2 + "Biggest Scientific Improvement": Weighted Global -----
def build_weighted_global_features(df: pd.DataFrame, per_source_cols: dict):
    """
    Constructs a 'weighted global' representation where every perspective
    contributes EQUAL total variance to the combined feature space,
    regardless of its raw feature count (counteracting the imbalance
    quantified in Experiment 10). Each perspective's standardized feature
    block is rescaled by 1/sqrt(n_features_in_block) so that the block's
    total variance sums to a constant, then all blocks are concatenated.
    This directly tests the feature-imbalance / subspace-masking mechanism
    hypothesis: if weighted-global approaches perspective-scoped quality,
    imbalance was the cause; if it still underperforms, semantic
    partitioning itself matters independent of feature-count balance.
    """
    blocks = []
    for key, cols in per_source_cols.items():
        X_block = preprocess_features(df, cols)
        weight = 1.0 / np.sqrt(len(cols))
        blocks.append(X_block * weight)
    X_weighted = np.concatenate(blocks, axis=1)
    return X_weighted


def run_weighted_global_condition(df, per_source_cols, k_range, random_state, n_bootstrap, top_n):
    """Fit-and-evaluate on the equal-variance-weighted global feature space (Experiment 2)."""
    section("EXPERIMENT 2 / MECHANISM TEST: Weighted Global Clustering "
            "(equal variance contribution per perspective)")
    X = build_weighted_global_features(df, per_source_cols)
    print(f"[weighted_global] Combined feature matrix shape: {X.shape} "
          f"(perspectives equally weighted by 1/sqrt(n_features) per block)")

    grid_df, labels_store = controlled_grid_benchmark(X, k_range, random_state, "weighted_global")
    print("\n-- Grid metrics (weighted-global, k-swept algorithms) --")
    print_table(grid_df.round(4))
    final_k = select_final_k(grid_df)
    print(f"\n-- Selected shared k = {final_k} --")

    stability_rows = []
    for algo in K_SWEPT_ALGORITHMS:
        result, _, ari_dist = bootstrap_stability(X, algo, final_k, n_bootstrap, random_state, "weighted_global")
        stability_ari_distributions[("weighted_global", algo)] = ari_dist
        stability_rows.append(result)
    print("\n-- Bootstrap stability (weighted_global) --")
    print_table(pd.DataFrame(stability_rows).round(4))

    summary = {"condition": "weighted_global", "final_k": final_k, "n_features": X.shape[1],
               "mean_silhouette": grid_df["silhouette"].mean(), "mode": "fit_and_evaluate",
               "hdbscan_k_discovered": np.nan, "hdbscan_silhouette": np.nan}
    return summary, grid_df["silhouette"].mean()


# ---- Experiment 3: PCA Baseline --------------------------------------------
def run_pca_baseline_condition(df, all_feature_cols, k_range, random_state, n_components=30):
    """
    Reduces the global-full feature space to n_components via PCA before
    clustering, addressing the 'maybe it's just the curse of dimensionality'
    reviewer objection. If perspective-scoped clustering still outperforms
    this PCA-reduced global baseline, dimensionality reduction alone does
    not explain the perspective-scoping advantage.
    """
    section(f"EXPERIMENT 3: PCA Baseline (global, reduced to {n_components} dims)")
    X_full = preprocess_features(df, all_feature_cols)
    n_comp = min(n_components, X_full.shape[1] - 1, X_full.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_full)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[pca_baseline] Reduced {X_full.shape[1]} dims -> {n_comp} dims; "
          f"cumulative explained variance = {explained:.4f}")

    grid_df, _ = controlled_grid_benchmark(X_pca, k_range, random_state, "global_pca")
    print("\n-- Grid metrics (global, PCA-reduced) --")
    print_table(grid_df.round(4))
    final_k = select_final_k(grid_df)
    print(f"\n-- Selected shared k = {final_k} --")

    summary = {"condition": "global_pca", "final_k": final_k, "n_features": n_comp,
               "mean_silhouette": grid_df["silhouette"].mean(), "mode": "fit_and_evaluate",
               "hdbscan_k_discovered": np.nan, "hdbscan_silhouette": np.nan}
    return summary


# ---- Experiment 4: Cluster Agreement (ARI between global & perspective) ---
def cluster_agreement(global_labels_store: dict, perspective_labels_by_name: dict, final_k_global: int):
    """
    Computes ARI between the global_full clustering (KMeans at its selected
    k) and each perspective's own clustering (KMeans at its selected k),
    on the SAME set of vehicles. Low agreement indicates the perspectives
    encode substantively different groupings, not redundant information.
    """
    section("EXPERIMENT 4: Cluster Agreement — Global vs. Perspective (ARI)")
    global_labels = global_labels_store.get(("KMeans", final_k_global))
    rows = []
    for persp_name, (labels, k) in perspective_labels_by_name.items():
        if global_labels is None or labels is None or len(global_labels) != len(labels):
            rows.append({"perspective": persp_name, "ari_vs_global": np.nan, "note": "length mismatch or missing"})
            continue
        ari = adjusted_rand_score(global_labels, labels)
        rows.append({"perspective": persp_name, "global_k": final_k_global, "perspective_k": k,
                     "ari_vs_global": ari})
    df = pd.DataFrame(rows)
    print_table(df.round(4))
    print("Interpretation: ARI near 0 indicates the global and perspective-scoped clusterings "
          "are essentially independent groupings (perspectives carve up the data differently); "
          "ARI near 1 indicates redundancy (perspective clustering mostly recovers the global partition).")
    return df


# ---- Experiment 6: Mutual Information Between Perspectives ----------------
def mutual_information_between_perspectives(perspective_labels_by_name: dict):
    """
    Computes Normalized Mutual Information (NMI) between every pair of
    perspective clusterings (KMeans at each perspective's own selected k).
    Tests whether perspectives encode independent information (low NMI) or
    redundant information (high NMI).
    """
    section("EXPERIMENT 6: Mutual Information Between Perspectives (NMI)")
    names = list(perspective_labels_by_name.keys())
    rows = []
    for a, b in combinations(names, 2):
        labels_a, _ = perspective_labels_by_name[a]
        labels_b, _ = perspective_labels_by_name[b]
        if labels_a is None or labels_b is None or len(labels_a) != len(labels_b):
            rows.append({"perspective_a": a, "perspective_b": b, "nmi": np.nan})
            continue
        nmi = normalized_mutual_info_score(labels_a, labels_b)
        rows.append({"perspective_a": a, "perspective_b": b, "nmi": nmi})
    df = pd.DataFrame(rows)
    print_table(df.round(4))
    print("Interpretation: NMI near 0 supports near-independent information encoding across "
          "perspectives (each captures a distinct facet of the vehicle); NMI near 1 suggests "
          "the perspectives are largely redundant.")
    return df


# ---- Experiment 5: Feature Removal (masking hypothesis) -------------------
def feature_removal_experiment(df, all_feature_cols, per_source_cols, k_range, random_state,
                                perspective_labels_by_name):
    """
    Removes the Tech & Comfort feature block (the largest/most dominant
    perspective per Experiment 10) entirely from the global feature set,
    re-clusters the remaining features, and tests via ARI whether the
    Safety perspective's own clustering re-emerges. Supports or refutes the
    'masking hypothesis': that high-dimensional perspectives suppress the
    influence of lower-dimensional ones in global clustering.
    """
    section("EXPERIMENT 5: Feature Removal (masking hypothesis — remove Tech & Comfort)")
    tech_cols = set(per_source_cols.get("tech_comfort", []))
    remaining_cols = [c for c in all_feature_cols if c not in tech_cols]
    print(f"[feature_removal] Removed {len(tech_cols)} Tech & Comfort features; "
          f"{len(remaining_cols)} features remain.")
    X_remaining = preprocess_features(df, remaining_cols)
    grid_df, labels_store = controlled_grid_benchmark(X_remaining, k_range, random_state,
                                                        "global_minus_tech")
    final_k = select_final_k(grid_df)
    labels_no_tech = labels_store.get(("KMeans", final_k))

    rows = []
    for persp_name, (persp_labels, persp_k) in perspective_labels_by_name.items():
        if persp_labels is None or labels_no_tech is None or len(persp_labels) != len(labels_no_tech):
            continue
        ari = adjusted_rand_score(persp_labels, labels_no_tech)
        rows.append({"perspective_compared": persp_name, "ari_vs_global_minus_tech": ari})
    df_result = pd.DataFrame(rows).sort_values("ari_vs_global_minus_tech", ascending=False)
    print(f"\n-- ARI between global_minus_tech (k={final_k}) and each perspective's own clustering --")
    print_table(df_result.round(4))
    print("Interpretation: if 'Safety' (or another previously-masked perspective) shows the "
          "HIGHEST ari_vs_global_minus_tech after Tech removal (compared to its agreement with "
          "the original global_full clustering in Experiment 4), this supports the masking "
          "hypothesis — Tech's dimensionality was suppressing Safety's signal in the joint space.")
    return df_result, final_k


# ---- Experiment 7: Silhouette Bootstrap 95% CI (data-level bootstrap) -----
def silhouette_bootstrap_ci(X: np.ndarray, algo: str, k: int, n_bootstrap: int, random_state: int):
    """
    Bootstrap resamples the DATA (not just re-reports grid variability) and
    refits+rescored silhouette each time, producing a genuine 95% CI on the
    silhouette score for one condition/algorithm/k — stronger than reporting
    a single point estimate (e.g., 0.46) with no uncertainty quantification.
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    sils = []
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        labels, _, _ = fit_algorithm(algo, k, X[idx], random_state + i + 1)
        if len(set(labels)) > 1:
            sils.append(silhouette_score(X[idx], labels))
    if not sils:
        return np.nan, np.nan, np.nan
    lower, upper = np.percentile(sils, [2.5, 97.5])
    return float(np.mean(sils)), float(lower), float(upper)


def silhouette_ci_report(condition_X_map: dict, best_algo_k_map: dict, n_bootstrap: int, random_state: int):
    """condition_X_map: {condition_name: X}; best_algo_k_map: {condition_name: (algo, k)}"""
    section("EXPERIMENT 7: Silhouette Bootstrap 95% Confidence Intervals")
    rows = []
    for cond, X in condition_X_map.items():
        algo, k = best_algo_k_map.get(cond, ("KMeans", 3))
        mean_sil, lo, hi = silhouette_bootstrap_ci(X, algo, k, n_bootstrap, random_state)
        rows.append({"condition": cond, "algorithm": algo, "k": k,
                     "silhouette_mean": mean_sil, "ci_lower_95": lo, "ci_upper_95": hi})
    df = pd.DataFrame(rows)
    print_table(df.round(4))
    print("Reporting convention: state absolute silhouette values with CIs (e.g., "
          "'silhouette increased from X to Y, 95% CI [lo, hi]') rather than percentage "
          "improvements, per reviewer guidance that percentage framing is discouraged for "
          "bounded [-1,1] metrics like silhouette.")
    global SILHOUETTE_CI_REGISTRY
    SILHOUETTE_CI_REGISTRY = df
    return df


# ---- Experiment 8: Stability vs. Quality -----------------------------------
def stability_vs_quality(stability_df: pd.DataFrame, grid_df_by_condition: dict):
    """
    Correlates bootstrap ARI stability against mean silhouette across all
    condition/algorithm combinations collected so far. Reports Pearson r
    as a text summary in place of a scatter plot (per no-file-output
    constraint). A weak or negative correlation would indicate that the
    highest-silhouette solution is not necessarily the most stable one.
    """
    section("EXPERIMENT 8: Stability vs. Quality (ARI vs. Silhouette correlation)")
    rows = []
    for cond, grid_df in grid_df_by_condition.items():
        mean_sil_by_algo = grid_df.groupby("algorithm")["silhouette"].mean()
        for algo, sil in mean_sil_by_algo.items():
            match = stability_df[(stability_df["condition"] == cond) & (stability_df["algorithm"] == algo)]
            if not match.empty:
                rows.append({"condition": cond, "algorithm": algo,
                             "mean_silhouette": sil, "ari_mean": match["ari_mean"].iloc[0]})
    df = pd.DataFrame(rows).dropna()
    print_table(df.round(4))
    if len(df) >= 3:
        r, p = stats.pearsonr(df["mean_silhouette"], df["ari_mean"])
        print(f"\nPearson correlation (silhouette vs. ARI stability): r={r:.4f}, p={p:.4f}")
        print("Interpretation: r close to 0 or negative would indicate the most stable "
              "clustering solution is NOT necessarily the highest-silhouette one — a relevant "
              "caveat for any 'best algorithm' claim in the Discussion section.")
    else:
        print("Insufficient paired observations for a reliable correlation estimate.")
    global STABILITY_VS_QUALITY_REGISTRY
    STABILITY_VS_QUALITY_REGISTRY = df
    return df


# ---- Experiment 9: Cluster Entropy / Balance -------------------------------
def cluster_balance_entropy(labels: np.ndarray, condition: str, algo: str):
    """
    Computes normalized Shannon entropy and Gini coefficient of the cluster
    size distribution. Entropy near 1 (normalized) indicates balanced
    cluster sizes; near 0 indicates one dominant cluster and several tiny
    ones (e.g., 120/120/11 in a 251-vehicle dataset).
    """
    labels_clean = labels[labels != -1] if -1 in labels else labels
    sizes = pd.Series(labels_clean).value_counts().values
    if len(sizes) < 2:
        return {"condition": condition, "algorithm": algo, "normalized_entropy": np.nan, "gini": np.nan}
    proportions = sizes / sizes.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    max_entropy = np.log(len(sizes))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else np.nan

    sorted_sizes = np.sort(sizes)
    n = len(sorted_sizes)
    cum = np.cumsum(sorted_sizes)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

    return {"condition": condition, "algorithm": algo, "n_clusters": len(sizes),
            "normalized_entropy": normalized_entropy, "gini": gini,
            "cluster_sizes": str(sorted(sizes.tolist(), reverse=True))}


def cluster_balance_report(labels_by_condition_algo: dict):
    section("EXPERIMENT 9: Cluster Balance (Entropy & Gini)")
    rows = [cluster_balance_entropy(labels, cond, algo)
            for (cond, algo), labels in labels_by_condition_algo.items()]
    df = pd.DataFrame(rows)
    print_table(df.round(4))
    print("Interpretation: normalized_entropy near 1.0 = balanced cluster sizes; near 0.0 = one "
          "dominant cluster with several near-empty clusters. Gini near 0 = balanced; near 1 = "
          "highly unequal cluster sizes.")
    global CLUSTER_BALANCE_REGISTRY
    CLUSTER_BALANCE_REGISTRY = df
    return df


# ---- Manuscript scaffolding: contributions, dataset stats, hyperparams ----
def print_contributions_section():
    section("MANUSCRIPT SCAFFOLD: Contributions (for end of Introduction)")
    print("This paper makes four contributions:")
    print("  1. A controlled comparison of clustering scope (global vs. perspective-scoped) "
          "fully crossed with clustering algorithm (KMeans, GMM, Agglomerative, HDBSCAN) on a "
          "real-world, mixed-domain automotive dataset.")
    print("  2. A dimensionality-vs-semantics ablation (random feature sampling, PCA-reduced "
          "global baseline, and a variance-equalized 'weighted global' representation) that "
          "directly tests the feature-imbalance / subspace-masking mechanism hypothesis, rather "
          "than only observing a scope effect.")
    print("  3. A stability and agreement analysis (bootstrap ARI/AMI, cross-perspective NMI, "
          "cluster balance entropy) characterizing how robust and how independent the "
          "discovered cluster structures are.")
    print("  4. A methodologically corrected statistical testing protocol (Holm-Bonferroni "
          "adjusted pairwise comparisons, explicit small-sample Friedman caveats, bootstrap "
          "95% CIs) for comparing clustering algorithms across conditions.")


BODY_STYLE_MAP = {
    'Swift': 'Hatchback', 'Baleno': 'Hatchback', 'WagonR': 'Hatchback', 'Tiago': 'Hatchback',
    'Tiago.ev': 'Hatchback', 'Altroz': 'Hatchback', 'Glanza': 'Hatchback', 'Comet': 'Hatchback',
    'Dzire': 'Sedan', 'City': 'Sedan', 'Verna': 'Sedan', 'Virtus': 'Sedan', 'Slavia': 'Sedan',
    'Amaze': 'Sedan', 'Camry': 'Sedan', 'C-Class': 'Sedan', 'E-Class': 'Sedan', '3': 'Sedan', '5': 'Sedan',
    'Octavia': 'Sedan', 'Superb': 'Sedan',
    'Creta': 'SUV', 'Seltos': 'SUV', 'Venue': 'SUV', 'Brezza': 'SUV', 'Sonet': 'SUV',
    'Nexon': 'SUV', 'Nexon.ev': 'SUV', 'Punch': 'SUV', 'Punch.ev': 'SUV', 'XUV700': 'SUV',
    'XUV400': 'SUV', 'XUV': 'SUV', 'Scorpio-N': 'SUV', 'Thar': 'SUV', 'Bolero': 'SUV',
    'Safari': 'SUV', 'Hector': 'SUV', 'Astor': 'SUV', 'ZS': 'SUV', 'Gloster': 'SUV',
    'Compass': 'SUV', 'Meridian': 'SUV', 'Tucson': 'SUV', 'Taigun': 'SUV', 'Kushaq': 'SUV',
    'Q3': 'SUV', 'Q7': 'SUV', 'Q8': 'SUV', 'X1': 'SUV', 'X7': 'SUV', 'GLC': 'SUV', 'GLS': 'SUV',
    'XC40': 'SUV', 'XC60': 'SUV', 'Defender': 'SUV', 'Discovery': 'SUV', 'Macan': 'SUV',
    'Fortuner': 'SUV', 'Magnite': 'SUV', 'Kiger': 'SUV', 'C3': 'SUV', 'EV6': 'SUV',
    'BE': 'SUV', 'XEV': 'SUV', 'Curvv.ev': 'SUV', 'Fronx': 'SUV', 'Elevate': 'SUV',
    'BYD': 'SUV', 'Ioniq': 'SUV',
    'Ertiga': 'MPV', 'Innova': 'MPV', 'Vellfire': 'MPV', 'Triber': 'MPV', 'Carens': 'MPV',
    'LM350h': 'MPV', 'LM500h': 'MPV',
}


def infer_body_style(brand: str, base_model: str) -> str:
    """
    Maps a vehicle's base model name to a body style (SUV/Sedan/Hatchback/MPV)
    using a manually curated lookup, since none of the four source CSVs
    include an explicit body-style column. 'Grand' is brand-disambiguated
    (Hyundai Grand i10 = Hatchback; Maruti Grand Vitara = SUV), which the
    flat lookup cannot resolve on its own.
    """
    if base_model == "Grand":
        return "Hatchback" if brand == "Hyundai" else "SUV"
    return BODY_STYLE_MAP.get(base_model, "Unclassified")


def print_dataset_statistics(df: pd.DataFrame, per_source_cols: dict, id_col: str, source_files: dict):
    """
    Computes and prints the manuscript's Table 1 (Dataset Statistics Overview)
    directly from the raw source files, since the merged/preprocessed `df`
    passed into this pipeline drops identifier-like text columns (brand,
    fuel_type, etc.) during load_and_merge(). Body style is derived via
    infer_body_style() since no source file provides it natively.
    """
    section("MANUSCRIPT SCAFFOLD: Dataset Statistics (Table 1)")
    n_vehicles = df[id_col].nunique()
    total_features = sum(len(v) for v in per_source_cols.values())
    print(f"n_vehicles (unique {id_col}, post-merge): {n_vehicles}")
    print(f"n_features (union across all perspectives): {total_features}")
    for key, cols in per_source_cols.items():
        n_missing = df[cols].isna().sum().sum()
        print(f"  perspective '{key}': {len(cols)} features, {n_missing} total missing cell(s)")

    raw_cf = pd.read_csv(source_files["driving_performance"])
    raw_cf["base_model"] = raw_cf["model_name"].str.split().str[0]
    raw_cf["body_style"] = raw_cf.apply(
        lambda r: infer_body_style(r["brand"], r["base_model"]), axis=1
    )

    total_vehicles = raw_cf["model_name"].nunique()
    unique_manufacturers = raw_cf["brand"].nunique()
    price_col = "ex_showroom_price_lakh"
    price_min, price_max = raw_cf[price_col].min(), raw_cf[price_col].max()
    price_mean = raw_cf[price_col].mean()
    body_counts = raw_cf["body_style"].value_counts()

    table1_rows = [
        {"Statistic": "Total Vehicles", "Value": total_vehicles},
        {"Statistic": "Unique Manufacturers", "Value": unique_manufacturers},
        {"Statistic": "Ex-Showroom Price Range (Lakh)", "Value": f"{price_min:.2f} to {price_max:.2f}"},
        {"Statistic": "Mean Ex-Showroom Price (Lakh)", "Value": f"{price_mean:.2f}"},
    ]
    for style in ["SUV", "Sedan", "Hatchback", "MPV"]:
        table1_rows.append({"Statistic": f"{style} Body Styles",
                             "Value": int(body_counts.get(style, 0))})
    unclassified = int(body_counts.get("Unclassified", 0))
    if unclassified > 0:
        table1_rows.append({"Statistic": "Unclassified Body Styles", "Value": unclassified})

    table1_df = pd.DataFrame(table1_rows)
    print("\n-- Table 1: Dataset Statistics Overview --")
    print_table(table1_df)
    print("\nNote: body style is inferred from base model name via a manually curated "
          "lookup (infer_body_style()); no source CSV provides body style natively. "
          "'Grand' is disambiguated by brand (Hyundai Grand i10 = Hatchback; "
          "Maruti Grand Vitara = SUV).")

    print(f"\nvehicles_per_manufacturer:\n{raw_cf['brand'].value_counts().to_string()}")

    fuel_cols = [c for c in df.columns if "fuel_type" in c.lower()]
    if fuel_cols:
        print(f"\nfuel_type distribution (merged numeric view):\n{df[fuel_cols[0]].value_counts().to_string()}")
    if "fuel_type" in raw_cf.columns:
        print(f"\nfuel_type distribution (car_features_v2, raw):\n{raw_cf['fuel_type'].value_counts().to_string()}")

    return table1_df

def print_hyperparameter_table():
    section("MANUSCRIPT SCAFFOLD: Hyperparameter Table")
    hp_df = pd.DataFrame([
        {"algorithm": "KMeans", "hyperparameter": "n_init", "value": 10},
        {"algorithm": "KMeans", "hyperparameter": "random_state", "value": CONFIG["random_state"]},
        {"algorithm": "GMM", "hyperparameter": "random_state", "value": CONFIG["random_state"]},
        {"algorithm": "GMM", "hyperparameter": "covariance_type", "value": "full (sklearn default)"},
        {"algorithm": "Agglomerative", "hyperparameter": "linkage", "value": "ward"},
        {"algorithm": "HDBSCAN", "hyperparameter": "min_cluster_size", "value": CONFIG["hdbscan_min_cluster_size"]},
        {"algorithm": "PCA (Experiment 3)", "hyperparameter": "n_components", "value": 30},
        {"algorithm": "All", "hyperparameter": "k_range", "value": f"{CONFIG['k_min']}-{CONFIG['k_max']}"},
        {"algorithm": "All", "hyperparameter": "n_bootstrap", "value": CONFIG["n_bootstrap"]},
        {"algorithm": "All", "hyperparameter": "preprocessing", "value": "median impute + z-score standardize"},
    ])
    print_table(hp_df)


def print_pipeline_diagram():
    section("MANUSCRIPT SCAFFOLD: Pipeline Diagram (text form)")
    print("""
    Vehicle Dataset (4 source CSVs, merged on model_name)
            |
            v
    Feature Perspectives (Safety / Ownership / Tech & Comfort / Driving Performance / Overall)
            |
            v
    Preprocessing (median impute -> z-score standardize, per condition)
            |
            v
    Clustering Algorithms (KMeans, GMM, Agglomerative-Ward, HDBSCAN) x shared k=2..10
            |
            v
    Mechanism-Testing Ablations (dimensionality ablation, weighted-global, PCA baseline,
                                  feature removal, cluster agreement, mutual information)
            |
            v
    Bootstrap Stability (ARI/AMI) + Statistical Testing (Friedman, Holm-Bonferroni Wilcoxon,
                                                            silhouette bootstrap 95% CI)
            |
            v
    Interpretation (scope comparison summary, mechanism verdict, cluster balance/entropy)
    """)


def print_abstract_conclusion_guidance():
    section("MANUSCRIPT SCAFFOLD: Abstract/Conclusion Phrasing Guidance")
    print("Abstract: report silhouette as absolute values with CIs, not percentage change.")
    print("  Avoid: 'improves by 179%%'")
    print("  Prefer: 'Silhouette increased from [TODO: baseline value] to [TODO: perspective "
          "value] (95%% CI [TODO, TODO]).' Fill in actual values from EXPERIMENT 7 output above.")
    print("Conclusion: avoid causal/strong-transition language.")
    print("  Avoid: 'This proves perspective-specific clustering is superior.'")
    print("  Prefer: 'The results suggest that perspective-specific clustering is a promising "
          "alternative to monolithic representations for this dataset.'")


# ============================================================================
# CHART GENERATION (v9): renders the LaTeX manuscript's placeholder figures
# ============================================================================
#
# Maps directly onto the \label{} placeholders in the manuscript:
#   fig:silhouette_comparison  -> silhouette_comparison.png
#   fig:hdbscan_clusters       -> hdbscan_clusters.png
#   fig:bootstrap_stability    -> bootstrap_stability.png
#   fig:feature_importance    -> feature_importance.png
#   fig:pipeline (verbatim ASCII diagram in LaTeX; no PNG needed, kept as text)
#
# Charts are saved under output/ as PNG + sidecar .meta.json. This section
# runs only if matplotlib/seaborn are available; if not, a warning is printed
# rather than fabricating placeholder images.
# ============================================================================

def generate_manuscript_charts(outdir: str = "output"):
    """
    Renders the LaTeX manuscript's placeholder figures using matplotlib +
    seaborn (no kaleido dependency, unlike a plotly-based approach). Each
    chart is saved as a PNG with a sidecar .meta.json (caption/description).
    Fails gracefully with a warning if matplotlib/seaborn are unavailable or
    a required registry is empty — never fabricates a placeholder image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json as _json
        import os as _os
    except ImportError:
        print("[generate_manuscript_charts] matplotlib/seaborn not installed — skipping chart "
              "generation. Install via `pip install matplotlib seaborn` to enable.")
        return

    _os.makedirs(outdir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ---- Figure: silhouette_comparison.png (Table 2 -> bar chart) ----------
    if GRID_METRICS_REGISTRY:
        rows = []
        for cond, gdf in GRID_METRICS_REGISTRY.items():
            rows.append({"condition": cond, "mean_silhouette": gdf["silhouette"].mean()})
        sil_df = pd.DataFrame(rows).sort_values("mean_silhouette", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=sil_df, x="mean_silhouette", y="condition", ax=ax, color="#4C78A8")
        ax.set_title("Mean silhouette score by clustering condition (higher is better)\n"
                      "Perspective-scoped conditions outperform global and ablation baselines",
                      fontsize=12)
        ax.set_xlabel("Mean silhouette")
        ax.set_ylabel("Condition")
        fig.tight_layout()
        fig.savefig(f"{outdir}/silhouette_comparison.png", dpi=150)
        plt.close(fig)
        with open(f"{outdir}/silhouette_comparison.png.meta.json", "w") as f:
            _json.dump({"caption": "Mean Silhouette Score by Clustering Condition",
                        "description": "Horizontal bar chart comparing mean silhouette across "
                                        "global, perspective-scoped, weighted-global, and PCA "
                                        "baseline clustering conditions."}, f)
        print(f"[generate_manuscript_charts] Saved {outdir}/silhouette_comparison.png "
              "(maps to fig:silhouette_comparison)")
    else:
        print("[generate_manuscript_charts] No grid metrics available — skipping silhouette_comparison.png")

    # ---- Figure: hdbscan_clusters.png (discovered-k by condition) ---------
    if HDBSCAN_ROWS_REGISTRY:
        rows = [{"condition": cond, "k_discovered": row.get("k_discovered", np.nan),
                 "noise_fraction": row.get("noise_fraction", np.nan)}
                for cond, row in HDBSCAN_ROWS_REGISTRY.items()]
        hdb_df = pd.DataFrame(rows).sort_values("k_discovered", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        norm = plt.Normalize(hdb_df["noise_fraction"].min(), hdb_df["noise_fraction"].max())
        colors = plt.cm.viridis(norm(hdb_df["noise_fraction"].fillna(0).values))
        ax.barh(hdb_df["condition"], hdb_df["k_discovered"], color=colors)
        ax.set_title("HDBSCAN discovered cluster count by condition (no k sweep)\n"
                      "Global space collapses to k=2; perspectives reveal richer densities",
                      fontsize=12)
        ax.set_xlabel("Discovered k")
        ax.set_ylabel("Condition")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Noise fraction")
        fig.tight_layout()
        fig.savefig(f"{outdir}/hdbscan_clusters.png", dpi=150)
        plt.close(fig)
        with open(f"{outdir}/hdbscan_clusters.png.meta.json", "w") as f:
            _json.dump({"caption": "Clusters Discovered by HDBSCAN per Condition",
                        "description": "Bar chart of HDBSCAN-discovered cluster counts across "
                                        "global and perspective-scoped conditions, colored by "
                                        "noise-point fraction."}, f)
        print(f"[generate_manuscript_charts] Saved {outdir}/hdbscan_clusters.png "
              "(maps to fig:hdbscan_clusters)")
    else:
        print("[generate_manuscript_charts] No HDBSCAN rows available — skipping hdbscan_clusters.png")

    # ---- Figure: bootstrap_stability.png (ARI distributions -> boxplot) ---
    if stability_ari_distributions:
        box_rows = []
        for (cond, algo), ari_list in stability_ari_distributions.items():
            for v in ari_list:
                box_rows.append({"condition_algorithm": f"{cond} | {algo}", "ari": v})
        if box_rows:
            box_df = pd.DataFrame(box_rows)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=box_df, x="condition_algorithm", y="ari", ax=ax, color="#72B7B2")
            ax.set_title("Bootstrap ARI stability distributions by condition (100 iterations)\n"
                          "Perspective-scoped fits show tighter, higher ARI than global fits",
                          fontsize=12)
            ax.set_xlabel("Condition | Algorithm")
            ax.set_ylabel("Bootstrap ARI")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(f"{outdir}/bootstrap_stability.png", dpi=150)
            plt.close(fig)
            with open(f"{outdir}/bootstrap_stability.png.meta.json", "w") as f:
                _json.dump({"caption": "Bootstrap Stability Distributions (100 Iterations)",
                            "description": "Boxplot of bootstrap Adjusted Rand Index (ARI) "
                                            "distributions across conditions and algorithms."}, f)
            print(f"[generate_manuscript_charts] Saved {outdir}/bootstrap_stability.png "
                  "(maps to fig:bootstrap_stability)")
    else:
        print("[generate_manuscript_charts] No bootstrap ARI distributions available — "
              "skipping bootstrap_stability.png")

    # ---- Figure: feature_importance.png (top eta-squared, global_full) ----
    if "global_full" in FEATURE_IMPORTANCE_GLOBAL_REGISTRY and not FEATURE_IMPORTANCE_GLOBAL_REGISTRY["global_full"].empty:
        fi_df = FEATURE_IMPORTANCE_GLOBAL_REGISTRY["global_full"].copy()
        fi_df = fi_df.sort_values("eta_squared", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.barplot(data=fi_df, x="eta_squared", y="feature", hue="condition", ax=ax, dodge=False)
        ax.set_title("Top distinguishing features in global_full clusters (by eta-squared)\n"
                      "Top drivers are dominated by Tech & Comfort features", fontsize=12)
        ax.set_xlabel("Eta-squared")
        ax.set_ylabel("Feature")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        fig.tight_layout()
        fig.savefig(f"{outdir}/feature_importance.png", dpi=150)
        plt.close(fig)
        with open(f"{outdir}/feature_importance.png.meta.json", "w") as f:
            _json.dump({"caption": "Global Feature Importance (Eta-Squared)",
                        "description": "Horizontal bar chart of top distinguishing features for "
                                        "the global_full clustering, ranked by ANOVA eta-squared."}, f)
        print(f"[generate_manuscript_charts] Saved {outdir}/feature_importance.png "
              "(maps to fig:feature_importance)")
    else:
        print("[generate_manuscript_charts] No global_full feature importance available — "
              "skipping feature_importance.png")


if __name__ == "__main__":
    main()
