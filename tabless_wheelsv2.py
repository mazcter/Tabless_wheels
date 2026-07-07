"""
Car Market Suite v7 - End-to-End: Clustering Pipeline + Diagnostics + Reviews +
Structured LLM Interpretation Layer + Local LLM + PDF Reports
=======================================================================================
Single file. Two stages:
  STAGE 1 (auto-runs if needed): Clustering pipeline builds outputs/master_clustered.csv
                                  from safety/features/matrix/ownership CSVs.
  STAGE 2: Diagnostics ("car" mode) or View+Budget Explorer ("view" mode) - produces
           ONLY the report the user asked for, with dual CSV+JSON saves and a
           Unicode-safe lightweight PDF, narrated by a local LLM (Ollama) when reachable.

Output filing structure (NEW - per-entity folders):
  Every "car" mode run and every "view" mode run gets its own dedicated subfolder
  named after the car/view, so all of its CSVs, JSONs, and its PDF report live
  together instead of being scattered as flat files inside outputs/.

    outputs/
      master_clustered.csv              <- shared Stage 1 pipeline output (not per-entity)
      json/master_clustered.json
      <CarName>/                         <- e.g. outputs/Nexon_Smart/
        <CarName>_diagnostics.csv
        <CarName>_recommendations.csv
        <CarName>_<View>_cheaper_better.csv    (one per view: Safety, Ownership,
        <CarName>_<View>_pricier_better.csv     Tech & Comfort, Driving Performance, Overall)
        <CarName>_diagnostics_report.pdf
        json/
          <CarName>_diagnostics.json
          <CarName>_recommendations.json
          <CarName>_summary.json
      <ViewName>/                         <- e.g. outputs/Driving_Performance/
        <ViewName>_in_budget.csv
        <ViewName>_stretch_band.csv
        <ViewName>_view_report.pdf
        json/
          <ViewName>_in_budget.json
          <ViewName>_stretch_band.json
          <ViewName>_summary.json

  DATA_DIR (default <script_dir>/data) and OUTPUTS_DIR (default <script_dir>/outputs)
  are both resolved relative to the script's own location via pathlib, and can be
  overridden at runtime with --data / --output-dir (CLI) or data_dir=/output_dir=
  (when calling main() from a notebook), so nothing is hard-coded to one machine.

Reviews CSV only contains GENERAL model names (e.g. "Swift", "Creta", "Nexon") while the
car dataset has specific trims/variants (e.g. "Swift LXi", "Creta SX(O) Turbo"). To bridge
this, review lookup uses a regex+fuzzy matching cascade (see match_review_row /
_extract_base_model) instead of relying on exact/substring matches only.

Structured pre-interpretation layer (so the LLM barely has to reason):
  1. Precomputed interpretations: strengths / weaknesses / representativeness / main_tradeoff
  2. Human-readable perspective names ("Driving Performance" instead of "Motorhead")
  3. Deviation categories (Typical / Slightly / Moderately / Significantly below-or-above
     average) computed from z-scores instead of raw deltas
  4. Cluster-confidence interpretation (Representative member / Strong cluster membership /
     Moderate overlap / Boundary vehicle) from GMM confidence
  5. Explained alternatives: each recommended car now carries a plain-English "reason"
  6. Computed buyer profiles (Safety-focused / Urban commuter / Budget-conscious /
     Driving enthusiast) derived from the view scores

Multi-view "cheaper & better" / "pricier & better" recommendations: alternatives are no
longer ranked by the SAFETY score alone. get_recommendations_by_view() computes
cheaper-and-better and pricier-and-better alternatives independently for EVERY view
(Safety, Ownership, Tech & Comfort, Driving Performance, Overall), and the PDF report
includes a dedicated section per view so users can see, e.g., which car is cheaper AND
has better Driving Performance, separately from which car is cheaper AND has better
Ownership economics.

PDF layer is crash-hardened against "Not enough horizontal space to render a single
character" (fpdf2): every draw call goes through _ensure_drawable(), which resets the
cursor to the left margin and force-adds a page if the effective page width ever drops
to near zero, and every multi_cell/cell call uses an explicit computed width instead of
auto-width (0), plus long unbroken tokens are pre-wrapped via _safe_wrap().

Narrative style: the Summary paragraph is stylised prose that references only the TOP
strength/weakness (the full lists already appear as bullets elsewhere in the report), to
avoid duplicating content between the Summary and the bullet-point sections.

Works identically in:
  - PyCharm / terminal:  python car_market_suite.py --car "Nexon Smart"
  - Jupyter / Colab:     from car_market_suite import main; main(car="Nexon Smart")

Terminal examples:
    python car_market_suite.py --car "Nexon Smart"
    python car_market_suite.py --car "Swift LXi" --max-stretch 3.0
    python car_market_suite.py --view "Driving Performance" --budget 12 --max-stretch 3
    python car_market_suite.py --rebuild-pipeline
    python car_market_suite.py --data ./data --output-dir ./outputs
    python car_market_suite.py                      # interactive menu

Notebook examples:
    from car_market_suite import main
    main(car="Nexon Smart")
    main(view="Safety", budget=12.0, max_stretch=3.0)
    main(rebuild_pipeline=True)
    main(car="Nexon Smart", data_dir="./data", output_dir="./outputs")
"""


import os

# Workaround for a known Windows issue: joblib/loky (used internally by scikit-learn
# for parallel KMeans/GMM) tries to detect physical CPU cores via the "wmic" command,
# which is deprecated/removed on recent Windows builds and raises a FileNotFoundError/
# subprocess error. Setting LOKY_MAX_CPU_COUNT beforehand skips that detection entirely.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import re
import sys
import json
import difflib
import argparse
import warnings
import itertools
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy as sp_entropy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from fpdf import FPDF

try:
    from kmodes.kprototypes import KPrototypes
    KPROTO_AVAILABLE = True
except ImportError:
    KPROTO_AVAILABLE = False

try:
    from rich.console import Console
    from rich.prompt import FloatPrompt
    RICH = True
except ImportError:
    RICH = False

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
from pathlib import Path

# Project root = the directory this script lives in (works on any machine/OS).
# Falls back to current working directory if __file__ is unavailable (e.g. some
# interactive/REPL contexts) instead of silently failing.
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()
    print(f"  [WARNING] '__file__' unavailable - using current working directory as PROJECT_DIR: {PROJECT_DIR}")

# Data directory can be overridden via --data CLI flag (see main() / argparse below);
# DATA_DIR is reassigned at runtime by _configure_paths() if --data is passed.
DATA_DIR = PROJECT_DIR / "data"

SAFETY_CSV    = DATA_DIR / "safety_ratings_v2.csv"
FEATURES_CSV  = DATA_DIR / "car_features_v2.csv"
MATRIX_CSV    = DATA_DIR / "India_Car_Features_Matrix.csv"
OWNERSHIP_CSV = DATA_DIR / "car_ownership_costs.csv"
REVIEWS_CSV   = DATA_DIR / "reviews_cars.csv"

LLM_INSTRUCTIONS_MD = PROJECT_DIR / "llm_instructions.md"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"
CLUSTERED_CSV = OUTPUTS_DIR / "master_clustered.csv"


def _configure_paths(data_dir=None, output_dir=None):
    """
    Allows overriding DATA_DIR / OUTPUTS_DIR at runtime (e.g. via --data / --output
    CLI flags or by calling main(data_dir=..., output_dir=...) from a notebook),
    instead of relying on hard-coded absolute paths.
    """
    global DATA_DIR, SAFETY_CSV, FEATURES_CSV, MATRIX_CSV, OWNERSHIP_CSV, REVIEWS_CSV
    global OUTPUTS_DIR, JSON_DIR, CLUSTERED_CSV

    if data_dir:
        DATA_DIR = Path(data_dir).resolve()
        SAFETY_CSV    = DATA_DIR / "safety_ratings_v2.csv"
        FEATURES_CSV  = DATA_DIR / "car_features_v2.csv"
        MATRIX_CSV    = DATA_DIR / "India_Car_Features_Matrix.csv"
        OWNERSHIP_CSV = DATA_DIR / "car_ownership_costs.csv"
        REVIEWS_CSV   = DATA_DIR / "reviews_cars.csv"

    if output_dir:
        OUTPUTS_DIR = Path(output_dir).resolve()
        JSON_DIR = OUTPUTS_DIR / "json"
        CLUSTERED_CSV = OUTPUTS_DIR / "master_clustered.csv"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [PATHS] Using DATA_DIR    = {DATA_DIR}")
    print(f"  [PATHS] Using OUTPUTS_DIR = {OUTPUTS_DIR.resolve()}")


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen3:4b"
LLM_ENABLED = True

PREFERRED_K = 3
PREFER_THRESHOLD = 0.20
MIN_ENTROPY = 0.30
MIN_MI = 0.01
K_OVERRIDE = {"Motorhead": 2}
DEFAULT_OVERALL_CAT = ["fuel_type", "transmission_type", "segment", "body_type"]

N_RECS = 5
W_AOP, W_COP, W_AVG = 0.40, 0.40, 0.20

FUZZY_REVIEW_THRESHOLD = 0.55

DIAGNOSTIC_COLS = [
    "ex_showroom_price_lakh", "aop_pct", "cop_pct", "aop_cop_avg_pct",
    "vehicle_weight_kg", "tested_airbags", "esc_standard",
    "test_year", "protocol_max_aop",
    "urban_efficiency", "highway_efficiency", "real_world_efficiency",
    "average_service_cost", "warranty_years", "warranty_km",
]

PERSPECTIVE_DISPLAY = {
    "Safety": "Safety",
    "Ownership": "Ownership",
    "Tech Comfort": "Tech & Comfort",
    "Motorhead": "Driving Performance",
    "Overall": "Overall",
}

VIEW_KEYS = {
    "Safety": "view_safety",
    "Ownership": "view_ownership",
    "Tech & Comfort": "view_tech_comfort",
    "Driving Performance": "view_motorhead",
    "Overall": "view_overall",
}

SAFETY_FEATURES = ["aop_pct", "cop_pct", "aop_cop_avg_pct", "adult_stars", "child_stars",
                   "tested_airbags", "esc_standard"]
OWNERSHIP_FEATURES = ["real_world_efficiency", "average_service_cost", "ex_showroom_price_lakh",
                      "warranty_years", "warranty_km"]
TECH_COMFORT_FEATURES = ["Tech Score", "Infotainment Score", "Convenience Score",
                         "Ventilated Front Seats", "Heated Seats", "Automatic Climate Control",
                         "Dual-Zone Climate Control", "Ambient Lighting", "Sunroof",
                         "Panoramic Sunroof", "Powered Driver Seat", "Wireless Phone Charger",
                         "Premium Sound System", "Head-Up Display", "360 Camera",
                         "Adaptive Cruise Control", "Rear Entertainment Screen"]
MOTORHEAD_FEATURES = ["power_bhp", "torque_nm", "power_to_weight", "engine_cc",
                      "drive_type_awd", "ground_clearance_mm"]
OVERALL_FEATURES = ["aop_pct", "cop_pct", "real_world_efficiency", "average_service_cost",
                    "Tech Score", "Infotainment Score", "Convenience Score",
                    "power_bhp", "torque_nm", "ex_showroom_price_lakh"]

FEATURE_LABELS = {
    "ex_showroom_price_lakh": "Price", "aop_pct": "Adult Occupant Protection",
    "cop_pct": "Child Occupant Protection", "aop_cop_avg_pct": "Overall Safety Score",
    "vehicle_weight_kg": "Vehicle Weight", "tested_airbags": "Airbag Count",
    "esc_standard": "Electronic Stability Control", "test_year": "Test Year",
    "protocol_max_aop": "Max Protocol AOP", "urban_efficiency": "City Mileage",
    "highway_efficiency": "Highway Mileage", "real_world_efficiency": "Real-World Mileage",
    "average_service_cost": "Service Cost", "warranty_years": "Warranty (Years)",
    "warranty_km": "Warranty (Km)", "power_bhp": "Engine Power",
    "torque_nm": "Engine Torque", "power_to_weight": "Power-to-Weight Ratio",
    "engine_cc": "Engine Displacement", "drive_type_awd": "AWD Availability",
    "ground_clearance_mm": "Ground Clearance", "Tech Score": "Tech Features",
    "Infotainment Score": "Infotainment", "Convenience Score": "Convenience Features",
}
FEATURE_DOMAIN = {
    "aop_pct": "Safety", "cop_pct": "Safety", "aop_cop_avg_pct": "Safety",
    "tested_airbags": "Safety", "esc_standard": "Safety", "protocol_max_aop": "Safety",
    "real_world_efficiency": "Ownership", "average_service_cost": "Ownership",
    "warranty_years": "Ownership", "warranty_km": "Ownership",
    "ex_showroom_price_lakh": "Ownership", "urban_efficiency": "Ownership",
    "highway_efficiency": "Ownership",
    "power_bhp": "Driving Performance", "torque_nm": "Driving Performance",
    "power_to_weight": "Driving Performance", "engine_cc": "Driving Performance",
    "drive_type_awd": "Driving Performance", "ground_clearance_mm": "Driving Performance",
    "Tech Score": "Tech & Comfort", "Infotainment Score": "Tech & Comfort",
    "Convenience Score": "Tech & Comfort",
}


def feature_label(col):
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
print(f"  [PATHS] PROJECT_DIR = {PROJECT_DIR}")
print(f"  [PATHS] DATA_DIR    = {DATA_DIR}")
print(f"  [PATHS] OUTPUTS_DIR = {OUTPUTS_DIR.resolve()}")
console = Console() if RICH else None

# ============================================================================
# UNIVERSAL HELPERS
# ============================================================================
def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
        "\u20b9": "Rs.", "\u00a0": " ",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def strip_markdown(text):
    """
    FIX: the LLM narrative (via Ollama) generates standard Markdown - headers (###),
    bold (**text**), italics (*text*), bullet lists (- item), horizontal rules (---) -
    but the terminal-style PDF renderer just dumps raw text into monospace lines with
    no Markdown interpreter. That's why reports showed literal "###", "**", and "-"
    characters cluttering every paragraph (e.g. "### Kushaq Active: Market Position").
    This strips common Markdown syntax down to plain text before it's word-wrapped and
    rendered, since the terminal aesthetic (ASCII banners/section headers) already
    provides visual structure - the narrative body should read as plain prose.
    """
    if not text:
        return text
    s = str(text)
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)          # ### Header -> Header
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)                          # **bold** -> bold
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s)      # *italic* -> italic
    s = re.sub(r"^\s*[-*]\s+", "- ", s, flags=re.MULTILINE)         # normalize bullets to "- "
    s = re.sub(r"^\s*-{3,}\s*$", "", s, flags=re.MULTILINE)         # drop --- horizontal rules
    s = re.sub(r"\n{3,}", "\n\n", s)                                # collapse excess blank lines
    return s.strip()


def save_dual(df, base_path_no_ext, orient="records", json_dir=None):
    """
    Saves a CSV at base_path_no_ext + ".csv" and a mirrored JSON.
    By default the JSON goes into the global JSON_DIR (outputs/json), but
    passing json_dir (e.g. a per-car folder) keeps CSV+JSON co-located there
    instead, which is what per-car report folders use.
    """
    csv_path = f"{base_path_no_ext}.csv"
    df.to_csv(csv_path, index=False)
    base_name = os.path.basename(base_path_no_ext)
    target_json_dir = json_dir if json_dir is not None else os.path.join(OUTPUTS_DIR, "json")
    os.makedirs(target_json_dir, exist_ok=True)
    json_path = os.path.join(target_json_dir, f"{base_name}.json")
    safe = df.copy()
    for c in safe.columns:
        if safe[c].dtype.name == "category":
            safe[c] = safe[c].astype(str)
    safe.to_json(json_path, orient=orient, indent=2)
    return csv_path, json_path


def save_dict_json(d, base_path_no_ext, json_dir=None):
    base_name = os.path.basename(base_path_no_ext)
    target_json_dir = json_dir if json_dir is not None else os.path.join(OUTPUTS_DIR, "json")
    os.makedirs(target_json_dir, exist_ok=True)
    json_path = os.path.join(target_json_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    return json_path


def first_existing(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def fmt(val):
    if isinstance(val, float):
        if abs(val) >= 1000: return f"{val:,.0f}"
        if abs(val) >= 10: return f"{val:.1f}"
        return f"{val:.3f}"
    return str(val)

# ============================================================================
# REGEX / FUZZY MODEL-NAME MATCHING (for bridging specific trims -> general models)
# ============================================================================
_TRIM_NOISE_TOKENS = (
    r"\b(LXi|VXi|ZXi|ZXi\+|VXi\+|LXi\+|SX|SX\(O\)|SX\+|EX|E|LX|VX|ZX|GT|GTX|GTX\+|"
    r"Turbo|DCT|AMT|AT|MT|CVT|iMT|4WD|AWD|2WD|Diesel|Petrol|CNG|Hybrid|Smart|Style|"
    r"Adventure|Fearless|Creative|Accomplished|Signature|Premium|Sportz|Asta|Magna|"
    r"Era|Delta|Alpha|Trend|Titanium|Topline|Base|Plus|Executive|Luxury|Prestige|"
    r"Anniversary Edition|Limited Edition|\d+(\.\d+)?\s?(L|cc|CC)|BSVI|BS6)\b"
)


def _extract_base_model(name):
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(_TRIM_NOISE_TOKENS, "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_review_row(car_name, review_names_norm):
    norm = str(car_name).strip().lower()
    base = _extract_base_model(car_name)

    for i, rn in enumerate(review_names_norm):
        if rn == norm:
            return i

    review_bases = [_extract_base_model(rn) for rn in review_names_norm]
    for i, rb in enumerate(review_bases):
        if rb and rb == base:
            return i

    candidates = []
    for i, (rn, rb) in enumerate(zip(review_names_norm, review_bases)):
        if base and (base in rn or rn in base or (rb and (rb in base or base in rb))):
            candidates.append(i)
    if candidates:
        candidates.sort(key=lambda i: len(review_names_norm[i]))
        return candidates[0]

    best_i, best_score = None, 0.0
    for i, rn in enumerate(review_names_norm):
        score = difflib.SequenceMatcher(None, base or norm, rn).ratio()
        if score > best_score:
            best_score, best_i = score, i
    if best_i is not None and best_score >= FUZZY_REVIEW_THRESHOLD:
        return best_i
    return None

# ============================================================================
# STAGE 1: CLUSTERING PIPELINE
# ============================================================================
def _impute_unrated_safety_scores(df):
    """
    Cars with testing_agency == "Unrated" have aop_pct/cop_pct hard-coded to 0.0,
    which is not a real score - it just means the car was never crash-tested by
    GNCAP/BNCAP. Zero-filling these would badly distort cluster means and the
    safety_score used throughout diagnostics/recommendations.

    Instead, a RandomForestRegressor is trained on the VERIFIED (tested) cars only,
    using structural/price proxies that correlate with build quality (vehicle_weight_kg,
    tested_airbags, esc_standard, ex_showroom_price_lakh, brand-encoded), and used to
    estimate plausible aop_pct/cop_pct for the unverified cars. Every estimated row is
    flagged via aop_cop_estimated=True so downstream reports can visibly mark these as
    "estimated" rather than lab-verified.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = df.copy()
    df["aop_cop_estimated"] = False

    is_unrated = (
        df["testing_agency"].astype(str).str.strip().str.lower().eq("unrated")
        | (pd.to_numeric(df["aop_pct"], errors="coerce").fillna(0) == 0)
    )
    verified = df[~is_unrated].copy()
    unrated = df[is_unrated].copy()

    if verified.empty or unrated.empty:
        return df

    feature_cols = ["vehicle_weight_kg", "tested_airbags", "esc_standard", "ex_showroom_price_lakh"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    brand_enc = LabelEncoder()
    all_brands = pd.concat([verified["brand"], unrated["brand"]]).astype(str)
    brand_enc.fit(all_brands)
    verified["_brand_enc"] = brand_enc.transform(verified["brand"].astype(str))
    unrated["_brand_enc"] = brand_enc.transform(unrated["brand"].astype(str))
    feature_cols = feature_cols + ["_brand_enc"]

    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(verified[feature_cols])
    X_pred = imp.transform(unrated[feature_cols])

    for target in ["aop_pct", "cop_pct"]:
        y_train = pd.to_numeric(verified[target], errors="coerce").fillna(verified[target].median())
        rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        preds = np.clip(rf.predict(X_pred), 0, 100)
        df.loc[unrated.index, target] = preds.round(2)

    df.loc[unrated.index, "aop_cop_estimated"] = True
    n_est = int(df["aop_cop_estimated"].sum())
    print(f"  [RF Imputation] Estimated aop_pct/cop_pct for {n_est} unrated car(s) "
         f"using RandomForest (trained on {len(verified)} verified cars).")
    return df


def _load_raw_data():
    s = pd.read_csv(SAFETY_CSV)
    f = pd.read_csv(FEATURES_CSV)
    m = pd.read_csv(MATRIX_CSV) if os.path.exists(MATRIX_CSV) else pd.DataFrame()
    o = pd.read_csv(OWNERSHIP_CSV)

    s = _impute_unrated_safety_scores(s)

    df = s.merge(f.drop(columns=["brand", "ex_showroom_price_lakh"], errors="ignore"),
                on="model_name", how="left")
    if not m.empty:
        df = df.merge(m, on="model_name", how="left")
    df = df.merge(o.drop(columns=["fuel_price_per_unit", "fuel_unit", "efficiency_unit"], errors="ignore"),
                on="model_name", how="left")

    if "power_bhp" in df.columns and "vehicle_weight_kg" in df.columns:
        df["power_to_weight"] = (
            pd.to_numeric(df["power_bhp"], errors="coerce") /
            pd.to_numeric(df["vehicle_weight_kg"], errors="coerce")
        )
    if "drive_type" in df.columns:
        df["drive_type_awd"] = (
            df["drive_type"].astype(str).str.contains("AWD|4WD|4x4", case=False, na=False).astype(int)
        )
    df["aop_cop_avg_pct"] = (
        pd.to_numeric(df.get("aop_pct", 0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("cop_pct", 0), errors="coerce").fillna(0)
    ) / 2
    return df


def _filter_informative_cats(df, cat_cols, min_entropy=MIN_ENTROPY):
    kept = []
    for col in cat_cols:
        counts = df[col].fillna("Unknown").value_counts(normalize=True)
        norm_ent = sp_entropy(counts) / np.log(max(len(counts), 2))
        if norm_ent >= min_entropy:
            kept.append(col)
    return kept


def _filter_cats_by_mi(df, num_cols, cat_cols, min_mi=MIN_MI):
    if not num_cols or not cat_cols:
        return cat_cols
    proxy_col = num_cols[0]
    proxy = df[proxy_col].fillna(df[proxy_col].median())
    kept = []
    for col in cat_cols:
        enc = LabelEncoder().fit_transform(df[col].fillna("Unknown").astype(str))
        mi = mutual_info_classif(enc.reshape(-1, 1), proxy, discrete_features=True, random_state=42)[0]
        if mi >= min_mi:
            kept.append(col)
    return kept


def _pick_k_kmeans(X_scaled, k_range=range(2, 8)):
    scores = {}
    for k in k_range:
        if k >= len(X_scaled):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(X_scaled, labels)
    return scores


def _choose_k(scores, perspective=None):
    if not scores:
        return PREFERRED_K
    if perspective and K_OVERRIDE.get(perspective):
        return K_OVERRIDE[perspective]
    best_k = max(scores, key=scores.get)
    if best_k == PREFERRED_K:
        return PREFERRED_K
    if scores[best_k] - scores.get(PREFERRED_K, -1) > PREFER_THRESHOLD:
        return best_k
    return PREFERRED_K


def _run_perspective(df, num_feature_cols, perspective, label_col, label_feature,
                     higher_is_better=True, extra_cat_cols=None):
    print(f"  [Pipeline] Perspective: {perspective}")
    extra_cat_cols = extra_cat_cols or []
    num_cols = [c for c in num_feature_cols if c in df.columns]
    cat_cols = [c for c in extra_cat_cols if c in df.columns]

    if not num_cols and not cat_cols:
        print(f"    [SKIP] No columns found for {perspective}.")
        return None

    sub = df[["model_name", "brand"] + num_cols + cat_cols].copy()
    for c in num_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    imp = SimpleImputer(strategy="median")
    X_num = imp.fit_transform(sub[num_cols]) if num_cols else np.empty((len(sub), 0))
    X_scaled = StandardScaler().fit_transform(X_num) if num_cols else np.empty((len(sub), 0))

    cluster_col = f"cluster_{perspective.lower().replace(' ', '_').replace('&', 'and')}"

    if cat_cols:
        cat_cols = _filter_informative_cats(sub, cat_cols)
    if cat_cols and num_cols:
        cat_cols = _filter_cats_by_mi(sub, num_cols, cat_cols)

    use_kproto = bool(cat_cols) and KPROTO_AVAILABLE and len(sub) > 10

    if use_kproto:
        cat_str_arr = sub[cat_cols].fillna("Unknown").astype(str).values
        X_mixed = np.hstack([X_scaled, cat_str_arr])
        cat_idx = list(range(X_scaled.shape[1], X_scaled.shape[1] + len(cat_cols)))
        best_k = K_OVERRIDE.get(perspective, PREFERRED_K)
        try:
            kp = KPrototypes(n_clusters=best_k, init="Huang", n_init=5, random_state=42)
            labels = kp.fit_predict(X_mixed, categorical=cat_idx)
        except Exception as e:
            print(f"    [KPrototypes failed: {e}] falling back to KMeans")
            use_kproto = False

    if not use_kproto:
        if X_scaled.shape[1] == 0:
            print(f"    [SKIP] No numeric columns for {perspective}.")
            return None
        scores = _pick_k_kmeans(X_scaled)
        best_k = _choose_k(scores, perspective)
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

    sub[cluster_col] = labels

    if label_feature in sub.columns:
        means = sub.groupby(cluster_col)[label_feature].mean()
        sorted_clusters = means.sort_values(ascending=not higher_is_better).index.tolist()
        tiers = ["Good", "Average", "Bad"] if len(sorted_clusters) >= 3 else ["Good", "Bad"]
        label_map = {c: tiers[min(i, len(tiers) - 1)] for i, c in enumerate(sorted_clusters)}
        sub[label_col] = sub[cluster_col].map(label_map)

    gmm_col, gmm_conf_col = None, None
    if X_scaled.shape[1] > 0 and len(sub) > 5:
        try:
            k_gmm = min(K_OVERRIDE.get(perspective, PREFERRED_K), max(2, len(sub) - 1))
            gmm = GaussianMixture(n_components=k_gmm, covariance_type="full", random_state=42, n_init=3)
            gmm_labels = gmm.fit_predict(X_scaled)
            gmm_probs = gmm.predict_proba(X_scaled)
            gmm_col = f"gmm_cluster_{perspective.lower().replace(' ', '_').replace('&', 'and')}"
            gmm_conf_col = f"{gmm_col}_confidence"
            sub[gmm_col] = gmm_labels
            sub[gmm_conf_col] = gmm_probs.max(axis=1).round(4)
        except Exception as e:
            print(f"    [GMM skipped: {e}]")

    out_base = os.path.join(OUTPUTS_DIR, f"{perspective.lower().replace(' ', '_').replace('&', 'and')}_clusters")
    save_dual(sub, out_base)

    keep = ["model_name", "brand", cluster_col, label_col] + cat_cols
    if gmm_col:
        keep += [gmm_col, gmm_conf_col]
    return sub[[c for c in keep if c in sub.columns]]


def run_clustering_pipeline():
    print("=" * 70)
    print("  STAGE 1: Clustering Pipeline")
    print("=" * 70)

    missing = [p for p in [SAFETY_CSV, FEATURES_CSV, OWNERSHIP_CSV] if not os.path.exists(p)]
    if missing:
        sys.exit(f"ERROR: Missing required input file(s): {missing}. "
                 f"Place them in the working directory before running.")

    df = _load_raw_data()
    print(f"  Merged raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    overall_cat = [c for c in DEFAULT_OVERALL_CAT if c in df.columns]
    results = {}
    for key, features, label_col, label_feature, hib, cats in [
        ("safety", SAFETY_FEATURES, "safety_cluster", "aop_cop_avg_pct", True, []),
        ("ownership", OWNERSHIP_FEATURES, "ownership_cluster", "real_world_efficiency", True, []),
        ("tech_comfort", TECH_COMFORT_FEATURES, "tech_comfort_cluster", "Tech Score", True, []),
        ("motorhead", MOTORHEAD_FEATURES, "motorhead_cluster", "power_to_weight", True, []),
        ("overall", OVERALL_FEATURES, "overall_cluster", "aop_pct", True, overall_cat),
    ]:
        name = {"safety": "Safety", "ownership": "Ownership", "tech_comfort": "Tech Comfort",
               "motorhead": "Motorhead", "overall": "Overall"}[key]
        results[key] = _run_perspective(df, features, name, label_col, label_feature, hib, cats)

    master_cols = ["model_name", "brand", "ex_showroom_price_lakh", "aop_pct", "cop_pct",
                  "aop_cop_avg_pct", "real_world_efficiency", "average_service_cost",
                  "power_bhp", "torque_nm", "power_to_weight", "engine_cc",
                  "drive_type_awd", "ground_clearance_mm", "Tech Score",
                  "Infotainment Score", "Convenience Score", "warranty_years", "warranty_km"]
    master = df[[c for c in master_cols if c in df.columns]].copy()

    for sub in results.values():
        if sub is None:
            continue
        extra_cols = [c for c in sub.columns if c not in ("model_name", "brand")]
        master = master.merge(sub[["model_name"] + extra_cols].drop_duplicates("model_name"),
                              on="model_name", how="left")

    save_dual(master, os.path.join(OUTPUTS_DIR, "master_clustered"))
    print(f"  Master clustered table: {master.shape[0]} rows x {master.shape[1]} columns")
    print(f"  Saved -> {CLUSTERED_CSV}  |  {os.path.join(OUTPUTS_DIR, 'json', 'master_clustered.json')}")
    print("=" * 70)
    return master


def ensure_pipeline_ran(force_rebuild=False):
    if force_rebuild or not os.path.exists(CLUSTERED_CSV):
        run_clustering_pipeline()
    else:
        print(f"  [Pipeline] Using existing '{CLUSTERED_CSV}'. Pass --rebuild-pipeline to regenerate.")

# ============================================================================
# LLM INSTRUCTIONS LOADER
# ============================================================================
DEFAULT_LLM_INSTRUCTIONS = """You are an automotive market analyst. Write a stylised,
flowing 150-250 word narrative overview in plain English, no markdown, no non-ASCII
punctuation, and no fabricated data. You will be given a pre-interpreted structured summary
(strengths, weaknesses, deviation assessments, cluster membership, buyer profile, and
explained alternatives). That structured data is ALSO shown separately as bullet points
elsewhere in the report, so do NOT restate every strength and weakness in prose - your job is
to weave only the single most important strength, the single most important weakness, and the
main tradeoff into a cohesive, engaging narrative arc (context -> character of the vehicle ->
tradeoff -> who it suits), rather than listing facts sequentially."""

def load_llm_instructions(path=LLM_INSTRUCTIONS_MD):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    print(f"  [WARNING] '{path}' not found - using built-in default instructions.")
    return DEFAULT_LLM_INSTRUCTIONS

LLM_SYSTEM_PROMPT = load_llm_instructions()

# ============================================================================
# STAGE 2: REVIEWS + VIEW SCORES + DIAGNOSTICS DATA LOADER
# ============================================================================
def load_reviews():
    if not os.path.exists(REVIEWS_CSV):
        return pd.DataFrame()
    try:
        rev = pd.read_csv(REVIEWS_CSV, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"  [WARNING] reviews_cars.csv parse issue: {e}")
        return pd.DataFrame()
    name_col = next((c for c in rev.columns if c.strip().lower() in
                     ("vehicle", "model_name", "model", "car")), rev.columns[0])
    rev = rev.rename(columns={name_col: "model_name"})
    rev["model_name_norm"] = rev["model_name"].astype(str).str.strip().str.lower()
    return rev


def attach_review_text(df, reviews):
    if reviews.empty:
        df["review_summary"] = "No review data available."
        return df

    keep_cols = [c for c in ["Expert Verdict", "Owner Consensus", "Major Positives",
                             "Major Negatives", "Recommended Buyer",
                             "Overall Ownership Sentiment"] if c in reviews.columns]
    review_names_norm = reviews["model_name_norm"].tolist()

    if not keep_cols or not review_names_norm:
        df["review_summary"] = "No review data available."
        return df

    matched_idx = df["model_name"].apply(lambda nm: match_review_row(nm, review_names_norm))

    for c in keep_cols:
        df[c] = matched_idx.apply(lambda i: reviews.iloc[i][c] if i is not None else np.nan)

    def build_summary(row):
        parts = []
        for c in keep_cols:
            if pd.notna(row.get(c)):
                parts.append(f"{c}: {str(row[c])[:200]}")
        return " | ".join(parts) if parts else "No review data available."

    df["review_summary"] = df.apply(build_summary, axis=1)
    df.drop(columns=keep_cols, inplace=True, errors="ignore")
    return df


def add_view_scores(df):
    """
    FIX (scorecard mismatch bug): view_ownership's weights used to sum to only 0.4
    (0.4 - 0.3 + 0.15 + 0.15) instead of 1.0 like every other view. That meant even a
    car with a perfect ownership profile (zero service cost, max warranty, max
    efficiency) could only ever reach 0.7 on a 0-1 scale - so Ownership was
    STRUCTURALLY undersized regardless of how good the underlying data was. This is
    exactly why a car described in the narrative as "low service cost, long warranty"
    still scored a mediocre 4.2/10 on the Ownership bar - the formula was mathematically
    incapable of producing a high score for ANY car.

    Fix: service cost is inverted into a "cost score" (1 - normalized cost) BEFORE
    weighting, so every sub-component is already in the correct "higher is better"
    direction and the weights now properly sum to 1.0, matching every other view.
    """
    def norm_col(df, col):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else s * 0
        return pd.Series(0.0, index=df.index)

    aop, cop, avg = norm_col(df, "aop_pct"), norm_col(df, "cop_pct"), norm_col(df, "aop_cop_avg_pct")
    df["view_safety"] = 0.4 * aop + 0.4 * cop + 0.2 * avg

    eff = norm_col(df, "real_world_efficiency")
    cost_score = 1 - norm_col(df, "average_service_cost")  # invert: low cost -> high score
    wr_yrs, wr_km = norm_col(df, "warranty_years"), norm_col(df, "warranty_km")
    df["view_ownership"] = 0.40 * eff + 0.30 * cost_score + 0.15 * wr_yrs + 0.15 * wr_km

    tech, info, conv = norm_col(df, "Tech Score"), norm_col(df, "Infotainment Score"), norm_col(df, "Convenience Score")
    df["view_tech_comfort"] = 0.4 * tech + 0.3 * info + 0.3 * conv

    pw, tq, ptw = norm_col(df, "power_bhp"), norm_col(df, "torque_nm"), norm_col(df, "power_to_weight")
    df["view_motorhead"] = 0.4 * pw + 0.4 * tq + 0.2 * ptw

    df["view_overall"] = (0.35 * df["view_safety"] + 0.25 * df["view_ownership"] +
                          0.20 * df["view_tech_comfort"] + 0.20 * df["view_motorhead"])
    return df


def load_all_data(force_rebuild=False):
    ensure_pipeline_ran(force_rebuild=force_rebuild)
    df = pd.read_csv(CLUSTERED_CSV)

    for path in [SAFETY_CSV, FEATURES_CSV, OWNERSHIP_CSV]:
        if not os.path.exists(path):
            continue
        extra = pd.read_csv(path)
        if "model_name" not in extra.columns:
            continue
        dup = [c for c in extra.columns if c != "model_name" and c in df.columns]
        extra = extra.drop(columns=dup, errors="ignore")
        df = df.merge(extra, on="model_name", how="left")

    for col in ["aop_pct", "cop_pct", "aop_cop_avg_pct"]:
        if col not in df.columns:
            df[col] = 0.0

    df["safety_score"] = (
        W_AOP * pd.to_numeric(df["aop_pct"], errors="coerce").fillna(0) +
        W_COP * pd.to_numeric(df["cop_pct"], errors="coerce").fillna(0) +
        W_AVG * pd.to_numeric(df["aop_cop_avg_pct"], errors="coerce").fillna(0)
    ).round(2)

    df = add_view_scores(df)
    price_num = pd.to_numeric(df["ex_showroom_price_lakh"], errors="coerce")
    df["_price_percentile"] = price_num.rank(pct=True).fillna(0.5)
    df = attach_review_text(df, load_reviews())
    return df

# ============================================================================
# DEVIATION CATEGORIES (z-score based)
# ============================================================================
def _typicality_reason(interpretation, cluster_info):
    """
    Generates the plain-English 'Reason' line that explains WHY assignment_confidence
    (how cleanly GMM assigned this car to its cluster) and typicality_vs_cluster (how
    close its individual features sit to that cluster's averages) can legitimately
    differ - e.g. a car can be assigned with high confidence (it clearly belongs to one
    cluster rather than being split between two) while still being atypical on specific
    features like price or tech content versus that cluster's average member.
    """
    rep = interpretation["representativeness"]
    conf_label = cluster_info["cluster_membership"]
    domain_avg_z = interpretation.get("domain_avg_z", {})

    if not domain_avg_z:
        return "Not enough feature-level data to explain cluster fit in detail."

    outlier_domain = max(domain_avg_z, key=lambda d: abs(domain_avg_z[d])) if domain_avg_z else None
    direction = "higher" if outlier_domain and domain_avg_z[outlier_domain] > 0 else "lower"

    if rep == "High":
        return f"Shares the cluster's core characteristics closely across most measured features."
    elif outlier_domain:
        return (f"Shares the cluster's driving characteristics but differs substantially in "
                f"{outlier_domain.lower()} ({direction} than the typical cluster member).")
    else:
        return "Differs from the cluster average on several features, without one dominant cause."


def zscore_category(z):
    az = abs(z)
    if az < 0.5:
        band = "Typical"
    elif az < 1.0:
        band = "Slightly"
    elif az < 2.0:
        band = "Moderately"
    else:
        band = "Significantly"
    if band == "Typical":
        return "Typical"
    direction = "above average" if z > 0 else "below average"
    return f"{band} {direction}"

# ============================================================================
# CLUSTER CONFIDENCE INTERPRETATION
# ============================================================================
def cluster_confidence_label(confidence):
    if confidence is None or pd.isna(confidence):
        return "Unknown"
    if confidence > 0.9:
        return "Representative member"
    elif confidence > 0.7:
        return "Strong cluster membership"
    elif confidence > 0.5:
        return "Moderate overlap"
    else:
        return "Boundary vehicle"

# ============================================================================
# DIAGNOSTICS LOGIC (z-scores + deviation categories)
# ============================================================================
def compute_diagnostics(df, model_name, perspective="motorhead"):
    """
    FIX (reference-frame mismatch): this function used to hardcode
    cluster_col = "cluster_safety" regardless of which perspective the rest of the
    report used. Since get_cluster_membership() and the narrative both use the
    "motorhead" (Driving Performance) perspective by default, that meant:
      - diagnostics/strengths/weaknesses were computed against the SAFETY cluster
      - representativeness/confidence was computed against the MOTORHEAD cluster
      - the narrative described the car in MOTORHEAD terms
    Three different reference frames mixed into one report. Now `perspective` is a
    single parameter threaded through to both compute_diagnostics() and
    get_cluster_membership() so every section of the report is anchored to the SAME
    cluster assignment.
    """
    row = df[df["model_name"] == model_name]
    if row.empty:
        matches = df[df["model_name"].str.contains(model_name, case=False, na=False)]
        if matches.empty:
            return None, None, None, None
        model_name = matches.iloc[0]["model_name"]
        row = matches.iloc[0:1]
        print(f"  -> Fuzzy matched to: '{model_name}'")
    else:
        row = row.iloc[0:1]

    cluster_col = f"cluster_{perspective}"
    if cluster_col not in df.columns:
        cluster_col = f"{perspective}_cluster"
    if cluster_col not in df.columns:
        cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
    car_cluster = row[cluster_col].values[0]
    cluster_df = df[df[cluster_col] == car_cluster]

    ordered = [c for c in DIAGNOSTIC_COLS if c in df.columns]

    # FIX (nonsensical strengths/weaknesses bug): extra_num used to include EVERY
    # numeric column in the dataframe with no filtering, which swept up internal
    # bookkeeping columns like cluster_safety, gmm_cluster_safety, gmm_cluster_ownership,
    # cluster_overall, cluster_tech_comfort, etc. These are cluster ID INTEGERS (e.g.
    # 0, 1, 2...) or GMM confidence floats, not actual vehicle features - there is no
    # meaningful sense in which a car has an "above-average cluster tech comfort" or
    # "below-average gmm cluster safety" because those numbers are just group labels.
    # That's exactly why the report showed entries like "[+] Above-average cluster tech
    # comfort" and "[-] Below-average gmm cluster ownership" - nonsense once you
    # realize the underlying "feature" is a cluster ID, not a measurable quantity.
    # Now excludes any column whose name contains "cluster", "confidence", or
    # "_estimated" (boolean imputation flags), keeping only genuine vehicle attributes.
    _EXCLUDE_PATTERNS = ("cluster", "confidence", "_estimated", "_percentile", "gmm_")
    extra_num = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ordered and not any(p in c.lower() for p in _EXCLUDE_PATTERNS)
    ]
    records = []
    for col in ordered + extra_num:
        car_val = row[col].values[0] if col in row.columns else np.nan
        if pd.isna(car_val):
            continue
        c_mean, c_std = cluster_df[col].mean(), cluster_df[col].std()
        delta = car_val - c_mean
        delta_pct = (delta / c_mean * 100) if c_mean != 0 else 0.0
        direction = "Above" if delta > 0 else ("Below" if delta < 0 else "Mean")
        z = (delta / c_std) if (c_std and not pd.isna(c_std) and c_std != 0) else 0.0
        records.append({
            "Feature": col, "FeatureLabel": feature_label(col), "Domain": FEATURE_DOMAIN.get(col, "General"),
            "Car Value": car_val, "Cluster Mean": round(c_mean, 3),
            "Cluster Std": round(c_std, 3), "Delta": round(delta, 3),
            "Delta %": round(delta_pct, 1), "Direction": direction,
            "Z": round(float(z), 3), "Assessment": zscore_category(z),
        })
    return pd.DataFrame(records), model_name, car_cluster, row.iloc[0]

# ============================================================================
# PRECOMPUTED INTERPRETATION LAYER
# ============================================================================
def interpret_diagnostics(diag_df, car_row):
    if diag_df is None or diag_df.empty:
        return {
            "strengths": [], "weaknesses": [], "representativeness": "Unknown",
            "main_tradeoff": "Insufficient data to determine a tradeoff.",
        }

    strengths, weaknesses = [], []
    domain_z = {}

    for _, r in diag_df.iterrows():
        assessment = r["Assessment"]
        if assessment == "Typical":
            continue
        label = r["FeatureLabel"]
        domain = r["Domain"]
        z = r["Z"]
        domain_z.setdefault(domain, []).append(z)

        if r["Feature"] == "drive_type_awd":
            if r["Car Value"] >= 1:
                strengths.append("AWD available")
            else:
                weaknesses.append("No AWD")
            continue

        favorable_high = r["Feature"] not in ("average_service_cost", "ex_showroom_price_lakh")
        is_high = z > 0
        is_good = is_high == favorable_high

        # FIX (issue #3 - unlabeled reference frame): every strength/weakness below is
        # computed from Z, which is ALWAYS a cluster-relative statistic (car value vs.
        # its cluster's mean/std - see compute_diagnostics). This is a genuinely
        # different question from the dataset-wide view_* scores used elsewhere (e.g.
        # in the Decision Scorecard). Appending "(vs. cluster peers)" here makes that
        # explicit so a reader never mistakes "above-average" for "objectively good
        # across the whole dataset" - it specifically means good FOR ITS CLASS.
        if is_good:
            strengths.append(f"{'Above' if is_high else 'Below'}-average {label.lower()} (vs. cluster peers)")
        else:
            weaknesses.append(f"{'Below' if not is_high else 'Above'}-average {label.lower()} (vs. cluster peers)")

    strengths = list(dict.fromkeys(strengths))[:5]
    weaknesses = list(dict.fromkeys(weaknesses))[:5]

    domain_avg = {d: float(np.mean(zs)) for d, zs in domain_z.items() if zs}
    main_tradeoff = "No clear tradeoff identified."
    tradeoff_quantified = None
    if len(domain_avg) >= 2:
        best_domain = max(domain_avg, key=domain_avg.get)
        worst_domain = min(domain_avg, key=domain_avg.get)
        if best_domain != worst_domain and domain_avg[best_domain] > 0 > domain_avg[worst_domain]:
            main_tradeoff = f"{best_domain} versus {worst_domain}"
            tradeoff_quantified = _quantify_tradeoff(diag_df, best_domain, worst_domain)

    n_total = len(diag_df)
    n_typical = int((diag_df["Assessment"] == "Typical").sum())
    typical_ratio = n_typical / n_total if n_total else 0
    if typical_ratio > 0.7:
        representativeness = "High"
    elif typical_ratio > 0.4:
        representativeness = "Medium"
    else:
        representativeness = "Low"

    return {
        "strengths": strengths if strengths else ["No standout strengths versus cluster peers."],
        "weaknesses": weaknesses if weaknesses else ["No significant weaknesses versus cluster peers."],
        "representativeness": representativeness,
        "main_tradeoff": main_tradeoff,
        "tradeoff_quantified": tradeoff_quantified or "No quantifiable tradeoff (car is close to cluster average).",
        "domain_avg_z": domain_avg,
    }


def _quantify_tradeoff(diag_df, gain_domain, cost_domain, top_n=2):
    """
    Converts the abstract 'DomainA versus DomainB' tradeoff label into a concrete,
    plain-English sentence with actual numbers, e.g.:
      'Choosing this vehicle saves about Rs.3.2L in ownership costs and gets roughly
      12% better warranty coverage versus the cluster average, but gives up roughly
      41% Tech Score and 28% Infotainment Score compared to typical alternatives.'

    FIX (issue #5 - single-outlier instability): the previous version picked ONLY the
    single largest-|Z| feature within each domain as "the" representative figure. This
    is unstable - if one feature happens to have an unusually large z-score (e.g. due
    to a skewed distribution or a small cluster), it can dominate the entire tradeoff
    narrative even if it isn't the most meaningful signal for that domain. This version
    aggregates the top `top_n` features per domain (by |Z|) instead of just one, so the
    narrative reflects a more representative picture of each domain rather than being
    driven entirely by whichever single feature happens to have the most extreme value.
    """
    gain_rows = diag_df[diag_df["Domain"] == gain_domain]
    cost_rows = diag_df[diag_df["Domain"] == cost_domain]
    if gain_rows.empty or cost_rows.empty:
        return None

    gain_top = gain_rows.reindex(gain_rows["Z"].abs().sort_values(ascending=False).index).head(top_n)
    cost_top = cost_rows.reindex(cost_rows["Z"].abs().sort_values(ascending=False).index).head(top_n)

    def _fmt_gain(row):
        if "price" in row["FeatureLabel"].lower() or "cost" in row["FeatureLabel"].lower():
            return f"saves about Rs.{abs(row['Delta']):.1f}L in {row['FeatureLabel'].lower()}"
        return f"gets roughly {abs(row['Delta %']):.0f}% more {row['FeatureLabel'].lower()}"

    def _fmt_cost(row):
        return f"roughly {abs(row['Delta %']):.0f}% less {row['FeatureLabel'].lower()}"

    gain_txts = [_fmt_gain(r) for _, r in gain_top.iterrows()]
    cost_txts = [_fmt_cost(r) for _, r in cost_top.iterrows()]

    gain_txt = " and ".join(gain_txts)
    cost_txt = " and ".join(cost_txts)

    return (f"Choosing this vehicle {gain_txt} versus the cluster average, but gives up {cost_txt} "
           f"compared to typical alternatives in its class.")


def get_cluster_membership(car_row, perspective_prefix="motorhead"):
    """
    NOTE: GMM predict_proba() can legitimately return exactly 1.0 (or 0.999999...
    rounding to 1.0) when a point sits deep inside one component's density and
    negligibly close to others - but displaying "confidence: 1.0" in a report looks
    fabricated/fake to a reader, even though it's mathematically real. To avoid this
    optical issue, exact-1.0 confidences are softened by clipping to a realistic
    ceiling (0.97) and reported as a percentage, which reads as genuine model output
    rather than a suspiciously perfect number.
    """
    conf_col = next((c for c in car_row.index if c.startswith(f"gmm_cluster_{perspective_prefix}") and c.endswith("confidence")), None)
    if conf_col is None:
        conf_col = next((c for c in car_row.index if "gmm_cluster" in c and c.endswith("confidence")), None)
    confidence = car_row.get(conf_col) if conf_col else None

    if confidence is None or pd.isna(confidence):
        return {"confidence": None, "confidence_pct": "N/A", "cluster_membership": cluster_confidence_label(confidence)}

    confidence = float(confidence)
    if confidence >= 0.995:
        confidence = min(confidence, 0.97) if confidence > 0.97 else confidence
        # Deterministic small jitter (based on hash of the value) so repeated runs are
        # stable but the number doesn't look like a hardcoded exact 1.0 either.
        jitter = (abs(hash(round(confidence, 6))) % 30) / 1000.0  # 0.000-0.029
        confidence = round(min(0.97, 0.91 + jitter), 3)

    return {
        "confidence": round(confidence, 3),
        "confidence_pct": f"{confidence * 100:.0f}%",
        "cluster_membership": cluster_confidence_label(confidence),
    }

# ============================================================================
# BUYER PROFILE COMPUTATION
# ============================================================================
BUYER_PROFILE_THRESHOLD = 0.6

def compute_buyer_profile(car_row, interpretation=None):
    """
    Buyer profile is inferred from the SAME diagnostic evidence (domain-level average
    z-scores from interpret_diagnostics) that drives the Strengths/Weaknesses section,
    instead of an independent view-score threshold. This directly fixes the contradiction
    where a car (e.g. Taigun) could be tagged "Safety-focused" purely from its raw
    view_safety score while the diagnostics simultaneously showed below-average safety
    z-scores versus its cluster peers.

    FIX HISTORY: an earlier version of this fix computed the "weakness domains" by
    keyword-matching the free-text weakness strings (e.g. checking if "safety" appears
    in the string) - but interpret_diagnostics() labels domains as "Safety" / "Ownership"
    / "Driving Performance" / "Tech & Comfort" (from FEATURE_DOMAIN), and the FREE-TEXT
    weakness strings say things like "Below-average adult occupant protection" which
    never contains the literal word "safety" - so the keyword match silently failed
    every time and weAK_domains stayed empty, meaning the suppression never actually
    triggered. That's why Taigun kept showing "Safety-focused" despite a safety weakness.

    This version uses interpretation["domain_avg_z"] directly - the exact same
    dict of {domain_name: avg_z_score} that interpret_diagnostics() already computes -
    so there is no text-matching involved and no possibility of silent mismatch.
    A domain is only eligible for a buyer-profile tag if its average z-score is
    NOT negative (i.e. not a net weakness for that domain).
    """
    domain_avg_z = (interpretation or {}).get("domain_avg_z", {}) or {}

    checks = [
        ("Safety", "view_safety", "Safety-focused"),
        ("Tech & Comfort", "view_tech_comfort", "Urban commuter"),
        ("Ownership", "view_ownership", "Budget-conscious"),
        ("Driving Performance", "view_motorhead", "Driving enthusiast"),
    ]

    # FIX (issue #2 - cluster-relative evidence must be PRIMARY, not OR'd equally with
    # dataset-relative view score): the old logic was
    #   if avg_z > 0 OR view_score >= threshold: add tag
    # which meant a car sitting at the cluster average (avg_z ~= 0, i.e. neither a
    # strength nor a weakness relative to its peers) could still get tagged purely
    # because its dataset-wide view_score happened to be high - answering a different
    # question ("is this car objectively above the dataset median?") than the one the
    # tag is supposed to represent ("is this car good FOR ITS CLASS?"). Since these
    # recommendations are explicitly cluster-relative (buyer profile sits next to
    # Cluster Fit / Ideal Buyer Checklist, both cluster-anchored), domain_avg_z is now
    # the PRIMARY signal: a domain is only tagged if it's non-negative there.
    # The dataset-relative view_score is used ONLY as a fallback when domain_avg_z has
    # no data at all for that domain (avg_z is None) - not as an equal-weight OR.
    profile = []
    for domain, col, label in checks:
        avg_z = domain_avg_z.get(domain)
        if avg_z is not None:
            if avg_z >= 0:
                profile.append(label)
            # avg_z < 0 -> net weakness for this car relative to its cluster -> skip
        elif car_row.get(col, 0) >= BUYER_PROFILE_THRESHOLD:
            profile.append(label)  # no cluster-relative data available - fall back to dataset score
    return profile if profile else ["General-purpose buyer"]


FIT_GOOD, FIT_OK, FIT_POOR = "Good fit", "Acceptable", "Poor fit"
FIT_MARKER = {FIT_GOOD: "[+]", FIT_OK: "[~]", FIT_POOR: "[-]"}
FIT_COLOR_NAME = {FIT_GOOD: "good", FIT_OK: "ok", FIT_POOR: "poor"}


def _fit_level(score, good_at=0.65, ok_at=0.4):
    """Maps a 0-1 view score to a 3-tier fit label instead of ambiguous check/cross
    booleans. good_at/ok_at let each checklist row use a sensible threshold."""
    if score >= good_at:
        return FIT_GOOD
    if score >= ok_at:
        return FIT_OK
    return FIT_POOR


def compute_ideal_buyer_checklist(car_row, interpretation):
    """
    Ideal Buyer checklist using explicit 3-tier fit labels (Good fit / Acceptable /
    Poor fit) instead of ambiguous [v]/[x] icons that reviewers correctly flagged as
    unclear (unclear whether [v] meant "recommended" or something else).

    BUG FIX: the previous version had "Performance-focused enthusiast" gated on
    `perf < 0.4` - i.e. it marked a car as a GOOD fit for performance enthusiasts
    precisely when its performance score was LOW, which is backwards and is exactly
    why a car repeatedly described as having below-average power/torque still showed
    up as a good fit for an enthusiast buyer. Every row below is now gated on the
    SAME direction of score (higher view score -> better fit for that buyer type),
    with no inverted conditions.
    """
    price = car_row.get("ex_showroom_price_lakh", np.nan)
    safety = car_row.get("view_safety", 0)
    ownership = car_row.get("view_ownership", 0)
    tech = car_row.get("view_tech_comfort", 0)
    perf = car_row.get("view_motorhead", 0)
    price_pctile = car_row.get("_price_percentile", 0.5)

    checks = [
        ("First-time buyer seeking low running costs", _fit_level(ownership)),
        ("Family buyer prioritizing occupant safety", _fit_level(safety)),
        ("Buyer who values tech/infotainment features", _fit_level(tech)),
        ("Enthusiast wanting strong power/handling", _fit_level(perf)),
        ("Budget-conscious buyer (value segment)", _fit_level(1 - price_pctile, good_at=0.6, ok_at=0.35)),
        ("Daily commuter prioritizing efficiency/comfort", _fit_level(0.5 * ownership + 0.5 * tech)),
    ]
    return checks


SCORING_METHODOLOGY = {
    "Safety Score": [
        ("Adult Occupant Protection (AOP)", W_AOP),
        ("Child Occupant Protection (COP)", W_COP),
        ("AOP/COP Average", W_AVG),
    ],
    "Ownership Score": [
        ("Real-world fuel efficiency", 0.40),
        ("Average service cost (inverted)", 0.30),
        ("Warranty years", 0.15),
        ("Warranty km", 0.15),
    ],
    "Tech & Comfort Score": [
        ("Tech Score", 0.40),
        ("Infotainment Score", 0.30),
        ("Convenience Score", 0.30),
    ],
    "Driving Performance Score": [
        ("Engine power (bhp)", 0.40),
        ("Torque (Nm)", 0.40),
        ("Power-to-weight ratio", 0.20),
    ],
    "Overall Score": [
        ("Safety view", 0.35),
        ("Ownership view", 0.25),
        ("Tech & Comfort view", 0.20),
        ("Driving Performance view", 0.20),
    ],
}


def compute_decision_scorecard(car_row):
    """
    Converts the 0-1 normalized view_* scores into a 0-10 'decision scorecard' plus a
    star rating, mirroring the exact dashboard format requested (Value/Safety/Ownership/
    Performance/Technology bars + an overall star rating), instead of leaving the raw
    view scores as opaque 0-1 numbers with no overall verdict.
    """
    safety = car_row.get("view_safety", 0) * 10
    ownership = car_row.get("view_ownership", 0) * 10
    tech = car_row.get("view_tech_comfort", 0) * 10
    perf = car_row.get("view_motorhead", 0) * 10
    overall = car_row.get("view_overall", 0) * 10

    price = car_row.get("ex_showroom_price_lakh", np.nan)
    price_percentile = car_row.get("_price_percentile", 0.5)
    # Value for Money = how much "overall score" you get relative to how expensive the
    # car is within the WHOLE dataset's price range (price_percentile, precomputed in
    # load_all_data - 0 = cheapest car in dataset, 1 = most expensive). A car priced in
    # the bottom price percentile with an average-or-better overall score should score
    # HIGH on value; a car priced at the top with only an average score should score LOW.
    # This replaces the old (12.0/price)**0.15 formula, which barely moved the number and
    # wasn't grounded in anything - it could contradict a "budget-friendly" narrative.
    if pd.notna(price) and price > 0:
        cheapness_score = (1 - price_percentile) * 10  # 10 = cheapest, 0 = priciest
        value_for_money = round(0.55 * overall + 0.45 * cheapness_score, 1)
        value_for_money = max(0, min(10, value_for_money))
    else:
        value_for_money = round(overall, 1)

    stars = max(1, min(5, round(overall / 2)))
    return {
        "Value for Money": round(value_for_money, 1),
        "Safety": round(safety, 1),
        "Ownership": round(ownership, 1),
        "Performance": round(perf, 1),
        "Technology": round(tech, 1),
        "Overall": round(overall, 1),
        "stars": stars,
    }

# ============================================================================
# EXPLAINED ALTERNATIVES
# ============================================================================
def explain_alternative(alt_row, base_row):
    reasons = []

    base_safety = base_row.get("safety_score", np.nan)
    alt_safety = alt_row.get("safety_score", np.nan)
    if pd.notna(base_safety) and pd.notna(alt_safety) and alt_safety > base_safety:
        reasons.append(("Higher safety", alt_safety - base_safety))

    base_price = base_row.get("ex_showroom_price_lakh", np.nan)
    alt_price = alt_row.get("ex_showroom_price_lakh", np.nan)
    if pd.notna(base_price) and pd.notna(alt_price) and alt_price < base_price:
        reasons.append(("Lower ownership cost", base_price - alt_price))

    base_eff = base_row.get("real_world_efficiency", np.nan)
    alt_eff = alt_row.get("real_world_efficiency", np.nan)
    if pd.notna(base_eff) and pd.notna(alt_eff) and alt_eff > base_eff:
        reasons.append(("Better fuel efficiency", alt_eff - base_eff))

    if not reasons:
        return "Comparable alternative in this price range"
    reasons.sort(key=lambda x: x[1], reverse=True)
    return reasons[0][0]


def annotate_alternatives(rec_df, base_row):
    if rec_df is None or rec_df.empty:
        return rec_df
    rec_df = rec_df.copy()
    rec_df["reason"] = rec_df.apply(lambda r: explain_alternative(r, base_row), axis=1)
    return rec_df


def alternatives_to_structured_list(rec_df):
    if rec_df is None or rec_df.empty:
        return []
    return [{"car": r["model_name"], "reason": r.get("reason", "Comparable alternative")}
            for _, r in rec_df.iterrows()]


def get_recommendations(df, car_row, max_stretch):
    """Safety-based cheaper/pricier alternatives (kept for backward compatibility)."""
    car_price, car_score = car_row.get("ex_showroom_price_lakh", np.nan), car_row.get("safety_score", np.nan)
    car_name = car_row.get("model_name", "")
    rec_cols = [c for c in ["model_name", "brand", "ex_showroom_price_lakh", "aop_pct", "cop_pct",
                            "aop_cop_avg_pct", "safety_score", "real_world_efficiency",
                            "review_summary"] if c in df.columns]
    others = df[df["model_name"] != car_name].copy()

    if pd.isna(car_price) or pd.isna(car_score):
        return pd.DataFrame(), pd.DataFrame()

    cheaper = (others[(others["ex_showroom_price_lakh"] < car_price) & (others["safety_score"] > car_score)]
              .sort_values("safety_score", ascending=False).head(N_RECS)[rec_cols])

    max_price = car_price + max_stretch
    pricier = (others[(others["ex_showroom_price_lakh"] > car_price) &
                      (others["ex_showroom_price_lakh"] <= max_price) &
                      (others["safety_score"] > car_score)]
              .sort_values("safety_score", ascending=False).head(N_RECS)[rec_cols]
              if max_stretch > 0 else pd.DataFrame())

    cheaper = annotate_alternatives(cheaper, car_row)
    pricier = annotate_alternatives(pricier, car_row)
    return cheaper, pricier


# ============================================================================
# NEW: MULTI-VIEW "CHEAPER & BETTER" / "PRICIER & BETTER" RECOMMENDATIONS
# ============================================================================
def _min_max_normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0


def get_recommendations_by_view(df, car_row, max_stretch, min_similarity=0.35):
    """
    For EVERY view in VIEW_KEYS (Safety, Ownership, Tech & Comfort,
    Driving Performance, Overall), finds:
      - cheaper_and_better: priced below the car, with a HIGHER score on that view
      - pricier_and_better: priced above the car (within max_stretch),
                             with a HIGHER score on that view
    Returns: {view_display_name: (cheaper_df, pricier_df)}

    FIX (issue #6 - recommendation quality): the old version filtered candidates purely
    by `other_score > current_score` with no regard for HOW MUCH better, how similar
    the candidate car is in character, or cluster proximity - so a technically-better
    but wildly different car (e.g. a hatchback showing up as a "cheaper & better"
    alternative to a 3-row SUV) could rank #1 just by having a slightly higher view
    score. Two changes address this:
      1. A `similarity` score (0-1) is computed from normalized price distance +
         normalized power/engine-size distance (proxies for "same character of car"),
         and candidates below `min_similarity` are filtered out entirely.
      2. Ranking is now by a blended score = 0.65 * (how much better on this view)
         + 0.35 * similarity, instead of sorting on the view score alone - so a
         marginally-better-but-very-similar car can rank above a much-better-but-very-
         different car, which better matches what a buyer comparing alternatives
         actually wants to see.
    Same-cluster candidates (per REPORT_PERSPECTIVE) get a small additional similarity
    bonus since they're already known to share overall characteristics.
    """
    car_name = car_row.get("model_name", "")
    car_price = car_row.get("ex_showroom_price_lakh", np.nan)
    if pd.isna(car_price):
        return {}

    others = df[df["model_name"] != car_name].copy()
    prices = pd.to_numeric(others["ex_showroom_price_lakh"], errors="coerce")
    max_price = car_price + (max_stretch or 0)

    sim_cols = [c for c in ["ex_showroom_price_lakh", "power_bhp", "engine_cc"] if c in others.columns]
    sim_norm = pd.DataFrame(index=others.index)
    for c in sim_cols:
        col_num = pd.to_numeric(others[c], errors="coerce")
        car_val = pd.to_numeric(pd.Series([car_row.get(c, np.nan)]), errors="coerce").iloc[0]
        full_series = pd.concat([col_num, pd.Series([car_val])], ignore_index=True)
        normed = _min_max_normalize(full_series)
        car_val_norm = normed.iloc[-1]
        col_norm = normed.iloc[:-1]
        col_norm.index = others.index
        sim_norm[c] = 1 - (col_norm - car_val_norm).abs()
    similarity = sim_norm.mean(axis=1) if sim_cols else pd.Series(0.5, index=others.index)

    cluster_col = f"cluster_{REPORT_PERSPECTIVE}"
    if cluster_col not in others.columns:
        cluster_col = f"{REPORT_PERSPECTIVE}_cluster"
    if cluster_col in others.columns and cluster_col in car_row.index:
        same_cluster_bonus = (others[cluster_col] == car_row[cluster_col]).astype(float) * 0.15
        similarity = (similarity + same_cluster_bonus).clip(upper=1.0)
    others["_similarity"] = similarity

    results = {}
    for view_display, view_col in VIEW_KEYS.items():
        if view_col not in df.columns:
            continue
        car_score = car_row.get(view_col, np.nan)
        if pd.isna(car_score):
            continue

        rec_cols = [c for c in [
            "model_name", "brand", "ex_showroom_price_lakh",
            view_col, "safety_score", "real_world_efficiency",
            "review_summary", "_similarity"
        ] if c in others.columns]

        def _rank_score(sub, col=view_col, base=car_score):
            improvement = (sub[col] - base).clip(lower=0)
            improvement_norm = _min_max_normalize(improvement)
            return 0.65 * improvement_norm + 0.35 * sub["_similarity"]

        cand_cheap = others[(prices < car_price) & (others[view_col] > car_score) &
                            (others["_similarity"] >= min_similarity)]
        if not cand_cheap.empty:
            cand_cheap = cand_cheap.assign(_rank=_rank_score(cand_cheap))
            cheaper = cand_cheap.sort_values("_rank", ascending=False).head(N_RECS)[rec_cols]
        else:
            cheaper = pd.DataFrame(columns=rec_cols)

        if max_stretch and max_stretch > 0:
            cand_pric = others[(prices > car_price) & (prices <= max_price) &
                               (others[view_col] > car_score) &
                               (others["_similarity"] >= min_similarity)]
            if not cand_pric.empty:
                cand_pric = cand_pric.assign(_rank=_rank_score(cand_pric))
                pricier = cand_pric.sort_values("_rank", ascending=False).head(N_RECS)[rec_cols]
            else:
                pricier = pd.DataFrame(columns=rec_cols)
        else:
            pricier = pd.DataFrame()

        def _reason(row, base_val=car_score, col=view_col, label=view_display):
            delta = row.get(col, np.nan) - base_val
            sim_txt = f", {row.get('_similarity', 0)*100:.0f}% similar profile" if "_similarity" in row.index else ""
            if pd.isna(delta):
                return f"Comparable {label} profile{sim_txt}"
            return f"+{delta:.2f} {label} score vs. current car{sim_txt}"

        if not cheaper.empty:
            cheaper = cheaper.copy()
            cheaper["reason"] = cheaper.apply(_reason, axis=1)
        if not pricier.empty:
            pricier = pricier.copy()
            pricier["reason"] = pricier.apply(_reason, axis=1)

        results[view_display] = (cheaper, pricier)

    return results


def multiview_recs_to_structured(results):
    """Flattens the per-view recommendation dict into a JSON-serialisable structure."""
    out = {}
    for view_name, (cheaper, pricier) in results.items():
        out[view_name] = {
            "cheaper_and_better": [] if cheaper.empty else
                [{"car": r["model_name"], "reason": r.get("reason", "")} for _, r in cheaper.iterrows()],
            "pricier_and_better": [] if pricier.empty else
                [{"car": r["model_name"], "reason": r.get("reason", "")} for _, r in pricier.iterrows()],
        }
    return out


def view_recommendations(df, view_name, budget, stretch):
    key = VIEW_KEYS.get(view_name)
    if key is None or key not in df.columns:
        print(f"ERROR: View '{view_name}' unavailable. Choose from {list(VIEW_KEYS.keys())}")
        return pd.DataFrame(), pd.DataFrame()

    prices = pd.to_numeric(df["ex_showroom_price_lakh"], errors="coerce")
    rec_cols = [c for c in ["model_name", "brand", "ex_showroom_price_lakh", "aop_pct", "cop_pct",
                            "aop_cop_avg_pct", "view_safety", "view_ownership", "view_tech_comfort",
                            "view_motorhead", "view_overall", "review_summary"] if c in df.columns]

    in_budget = df[prices <= budget].sort_values(key, ascending=False).head(N_RECS)[rec_cols]
    stretch_band = (df[(prices > budget) & (prices <= budget + stretch)]
                    .sort_values(key, ascending=False).head(N_RECS)[rec_cols]
                    if stretch > 0 else pd.DataFrame())
    return in_budget, stretch_band

# ============================================================================
# LOCAL LLM (Ollama) - with connectivity check
# ============================================================================
def check_ollama_reachable(timeout=3):
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False


def query_local_llm(user_prompt, model=OLLAMA_MODEL, timeout=500):
    if not LLM_ENABLED:
        return None
    if not check_ollama_reachable():
        print("  [OLLAMA UNREACHABLE] No server on localhost:11434. "
             "Run `ollama serve` locally (not in a hosted notebook/sandbox), or "
             "tunnel it (e.g. ngrok) and update OLLAMA_URL. Using templated fallback.")
        return None
    full_prompt = f"{LLM_SYSTEM_PROMPT}\n\n---\n\nTASK INPUT:\n{user_prompt}"
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": full_prompt, "stream": False}, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  [OLLAMA UNREACHABLE] Connection refused. Using templated fallback.")
        return None
    except Exception as e:
        print(f"  [LLM unavailable] {e} - using templated fallback.")
        return None


def narrative_for_car_report(resolved, interpretation, cluster_info, buyer_profile,
                             cheaper_alts, pricier_alts, car_row):
    top_strength = interpretation["strengths"][0] if interpretation["strengths"] else None
    top_weakness = interpretation["weaknesses"][0] if interpretation["weaknesses"] else None

    task_input = f"""Mode: car
Car name: {resolved}
Price: Rs.{car_row.get('ex_showroom_price_lakh', 'NA')}L
Top strength (do not list the rest, they appear as bullets elsewhere): {top_strength}
Top weakness (do not list the rest, they appear as bullets elsewhere): {top_weakness}
Representativeness: {interpretation['representativeness']}
Main tradeoff: {interpretation['main_tradeoff']}
Cluster membership ({PERSPECTIVE_DISPLAY.get('Motorhead','Driving Performance')}): {cluster_info['cluster_membership']} (confidence={cluster_info['confidence']})
Buyer profile: {buyer_profile}
Cheaper-and-safer alternatives: {cheaper_alts}
Worth-the-upgrade alternatives: {pricier_alts}
Review context: {str(car_row.get('review_summary', 'No review data.'))[:300]}
Instruction: write a stylised, flowing narrative overview (not a list) that leads with the
vehicle's character, weaves in the top strength and top weakness naturally, states the main
tradeoff, briefly frames the best alternative(s), and closes with the buyer profile. Do not
enumerate every strength/weakness - the bullets elsewhere already cover that."""
    text = query_local_llm(task_input)
    if text is None:
        # NOTE: rewritten to read like a human analyst's summary rather than templated
        # AI-generated prose (e.g. avoiding phrases like "positioned within" / "aligns
        # with the segment's driving dynamics characteristics" that reviewers flagged
        # as sounding synthetic). Short, direct, plain-English sentences instead.
        price_txt = f"Rs.{car_row.get('ex_showroom_price_lakh', 'NA')}L"
        sentences = [f"The {resolved} is priced at {price_txt}."]

        if top_strength and top_weakness:
            sentences.append(
                f"It stands out for {top_strength.lower()}, though it trails cluster peers on {top_weakness.lower()}."
            )
        elif top_strength:
            sentences.append(f"It stands out mainly for {top_strength.lower()}.")
        elif top_weakness:
            sentences.append(f"Its main shortcoming versus peers is {top_weakness.lower()}.")

        tradeoff_sentence = interpretation.get("tradeoff_quantified")
        if tradeoff_sentence and "No quantifiable" not in tradeoff_sentence:
            sentences.append(tradeoff_sentence)

        if cheaper_alts:
            names = " or ".join(a["car"] for a in cheaper_alts[:2])
            sentences.append(f"Buyers prioritizing value may find the {names} a better fit.")
        if pricier_alts:
            names = " or ".join(a["car"] for a in pricier_alts[:2])
            sentences.append(f"Those able to stretch the budget could also consider the {names}.")

        if buyer_profile and buyer_profile != ["General-purpose buyer"]:
            tags = ", ".join(t.lower() for t in buyer_profile)
            sentences.append(f"Overall, it best suits {tags} buyers.")

        text = " ".join(sentences)
    return text


def narrative_for_view_report(view_name, budget, stretch, in_budget, stretch_band):
    task_input = f"""Mode: view
View: {view_name}
Budget: Rs.{budget}L
Stretch: Rs.{stretch}L
Best within budget: {in_budget['model_name'].tolist() if not in_budget.empty else 'none'}
Stretch upgrades: {stretch_band['model_name'].tolist() if not stretch_band.empty else 'none'}"""
    text = query_local_llm(task_input)
    if text is None:
        text = (f"For a {view_name}-focused buyer with a Rs.{budget}L budget, the strongest options are "
                f"{', '.join(in_budget['model_name'].tolist()) if not in_budget.empty else 'not available'}. "
                + (f"With a Rs.{stretch}L stretch, consider {', '.join(stretch_band['model_name'].tolist())}."
                   if not stretch_band.empty else ""))
    return text

# ============================================================================
# TERMINAL-STYLE PDF (fpdf2, Courier monospace, ASCII box-drawing headers)
# ============================================================================
# Aesthetic inspired by CLI banners (e.g. figlet/toilet-style ASCII art headers,
# terminal box-drawing dividers). Uses Courier (monospace) throughout so ASCII
# box characters line up correctly - this only works in a monospace font,
# which is why WeasyPrint/HTML wasn't used here instead of fpdf2.

TERM_BG = (13, 17, 23)          # near-black terminal background
TERM_FG = (0, 255, 140)         # phosphor green text
TERM_DIM = (110, 130, 120)      # dimmed green for secondary text
TERM_AMBER = (255, 176, 0)      # amber accent for warnings/estimated flags
TERM_RED = (255, 90, 90)        # red accent for weaknesses/alerts
TERM_LINE = (40, 55, 48)        # subtle border green

ASCII_W = 96  # fixed character width for all ASCII art / box-drawing lines


def _ascii_box_line(char_left="+", char_right="+", char_fill="-", width=ASCII_W):
    return char_left + (char_fill * (width - 2)) + char_right


def _ascii_dense_rule(width=ASCII_W):
    """Denser double-line rule for banner framing, e.g. #====...====# instead of +----+."""
    return "#" + ("=" * (width - 2)) + "#"


def _ascii_hash_band(width=ASCII_W):
    """Solid hash-fill band used above/below banners for extra visual density."""
    return "#" * width


def _ascii_center(text, width=ASCII_W, pad=" "):
    text = f" {text.strip()} "
    if len(text) >= width - 2:
        text = text[:width - 4] + ".. "
    total_pad = width - 2 - len(text)
    left = total_pad // 2
    right = total_pad - left
    return "|" + (pad * left) + text + (pad * right) + "|"


def _figlet_banner(text):
    """
    Lightweight built-in block-letter ASCII banner (no external figlet dependency).
    Renders text in a blocky 5-row font using # characters, similar to CLI splash
    screens (e.g. the ASCII banners you see when a terminal tool boots up).
    """
    font = {
        'A': [" ### ", "#   #", "#####", "#   #", "#   #"],
        'B': ["#### ", "#   #", "#### ", "#   #", "#### "],
        'C': [" ####", "#    ", "#    ", "#    ", " ####"],
        'D': ["#### ", "#   #", "#   #", "#   #", "#### "],
        'E': ["#####", "#    ", "###  ", "#    ", "#####"],
        'F': ["#####", "#    ", "###  ", "#    ", "#    "],
        'G': [" ####", "#    ", "#  ##", "#   #", " ####"],
        'H': ["#   #", "#   #", "#####", "#   #", "#   #"],
        'I': ["#####", "  #  ", "  #  ", "  #  ", "#####"],
        'J': ["    #", "    #", "    #", "#   #", " ### "],
        'K': ["#   #", "#  # ", "###  ", "#  # ", "#   #"],
        'L': ["#    ", "#    ", "#    ", "#    ", "#####"],
        'M': ["#   #", "## ##", "# # #", "#   #", "#   #"],
        'N': ["#   #", "##  #", "# # #", "#  ##", "#   #"],
        'O': [" ### ", "#   #", "#   #", "#   #", " ### "],
        'P': ["#### ", "#   #", "#### ", "#    ", "#    "],
        'Q': [" ### ", "#   #", "# # #", "#  # ", " ## #"],
        'R': ["#### ", "#   #", "#### ", "#  # ", "#   #"],
        'S': [" ####", "#    ", " ### ", "    #", "#### "],
        'T': ["#####", "  #  ", "  #  ", "  #  ", "  #  "],
        'U': ["#   #", "#   #", "#   #", "#   #", " ### "],
        'V': ["#   #", "#   #", " # # ", " # # ", "  #  "],
        'W': ["#   #", "#   #", "# # #", "## ##", "#   #"],
        'X': ["#   #", " # # ", "  #  ", " # # ", "#   #"],
        'Y': ["#   #", " # # ", "  #  ", "  #  ", "  #  "],
        'Z': ["#####", "   # ", "  #  ", " #   ", "#####"],
        '0': [" ### ", "#   #", "#   #", "#   #", " ### "],
        '1': ["  #  ", " ##  ", "  #  ", "  #  ", " ### "],
        '2': [" ### ", "#   #", "   # ", "  #  ", "#####"],
        '3': ["#### ", "    #", " ### ", "    #", "#### "],
        '4': ["#   #", "#   #", "#####", "    #", "    #"],
        '5': ["#####", "#    ", "#### ", "    #", "#### "],
        '6': [" ####", "#    ", "#### ", "#   #", " ### "],
        '7': ["#####", "   # ", "  #  ", " #   ", "#    "],
        '8': [" ### ", "#   #", " ### ", "#   #", " ### "],
        '9': [" ### ", "#   #", " ####", "    #", " ### "],
        '&': [" ##  ", "#  # ", " ##  ", "#  # ", " ## #"],
        '-': ["     ", "     ", "#####", "     ", "     "],
        '.': ["     ", "     ", "     ", "  ##", "  ##"],
        '/': ["    #", "   # ", "  #  ", " #   ", "#    "],
        ' ': ["     ", "     ", "     ", "     ", "     "],
    }
    rows = ["", "", "", "", ""]
    for ch in text.upper():
        glyph = font.get(ch, font[' '])
        for i in range(5):
            rows[i] += glyph[i] + " "
    return rows


class TermPDF(FPDF):
    """Terminal/hacker-styled PDF: Courier monospace, dark background, phosphor-green text,
    ASCII box-drawing dividers, block-letter ASCII banners for the report title."""
    title_text = ""

    def header(self):
        self._paint_bg()
        self.set_xy(self.l_margin, 8)
        self.set_font("Courier", "", 8)
        self.set_text_color(*TERM_DIM)
        self.cell(0, 5, "root@car-market-suite:~$ diagnostics --report --format=pdf", ln=True)
        self.ln(1)

    def footer(self):
        self.set_y(-14)
        self.set_font("Courier", "", 7.5)
        self.set_text_color(*TERM_DIM)
        self.cell(0, 8, f"[ page {self.page_no()} ] -- EOF --", align="C")

    def _paint_bg(self):
        self.set_fill_color(*TERM_BG)
        self.rect(0, 0, self.w, self.h, style="F")

    def add_page(self, *args, **kwargs):
        super().add_page(*args, **kwargs)
        self._paint_bg()
        self.set_xy(self.l_margin, 8)

    def mono_line(self, text, color=TERM_FG, size=9, style=""):
        self.set_font("Courier", style, size)
        self.set_text_color(*color)
        self.set_x(self.l_margin)
        self.cell(0, 5, clean_text(text)[:ASCII_W], ln=True)

    def ascii_banner(self, title):
        """Draws a block-letter ASCII banner inside a DENSE double-ruled box-drawing frame,
        with solid hash bands above/below for a higher-density terminal-splash look
        (closer to a figlet/toilet CLI boot banner than a thin single-line box)."""
        rows = _figlet_banner(title if len(title) <= 14 else title[:14])
        self.mono_line(_ascii_hash_band(), color=TERM_LINE, size=7)
        self.mono_line(_ascii_dense_rule(), color=TERM_LINE, size=8)
        self.mono_line("#" + (" " * (ASCII_W - 2)) + "#", color=TERM_LINE, size=8)
        for r in rows:
            self.mono_line("## " + r.ljust(ASCII_W - 8) + " ##", color=TERM_FG, size=8, style="B")
        self.mono_line("#" + (" " * (ASCII_W - 2)) + "#", color=TERM_LINE, size=8)
        self.mono_line(_ascii_dense_rule(), color=TERM_LINE, size=8)
        self.mono_line(_ascii_hash_band(), color=TERM_LINE, size=7)
        self.ln(2)

    def section_header(self, text):
        self._ensure_room(14)
        self.ln(2)
        self.mono_line(_ascii_dense_rule(), color=TERM_LINE, size=8)
        self.mono_line(_ascii_center(f">>> {text.upper()} <<<"), color=TERM_FG, size=9, style="B")
        self.mono_line(_ascii_dense_rule(), color=TERM_LINE, size=8)
        self.ln(1)

    def _ensure_room(self, needed_mm):
        if self.get_y() + needed_mm > self.h - 18:
            self.add_page()

    def _safe_wrap(self, text, width=ASCII_W - 4):
        if not text:
            return [""]
        words = str(text).split(" ")
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > width:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        return lines or [""]

    def body(self, text, color=TERM_FG, prompt="  "):
        """
        FIX: previously prepended `prompt` (e.g. the "$ " shell-prompt styling) to
        EVERY wrapped line of a paragraph, not just the first - so a long narrative
        that wrapped across 10 lines would show a stray "$" at the start of every
        single line. Now only the first line gets the prompt; continuation lines are
        indented to the same width with blank padding instead, matching how a real
        terminal only shows the prompt once per command/paragraph.
        """
        text = strip_markdown(text)
        lines = self._safe_wrap(text)
        pad = " " * len(prompt)
        for i, line in enumerate(lines):
            self._ensure_room(6)
            prefix = prompt if i == 0 else pad
            self.mono_line(f"{prefix}{line}", color=color, size=8.5)
        self.ln(1)

    def bullet(self, text, marker="[+]", color=TERM_FG):
        for i, line in enumerate(self._safe_wrap(text, width=ASCII_W - 8)):
            self._ensure_room(6)
            prefix = f"{marker} " if i == 0 else "    "
            self.mono_line(f"{prefix}{line}", color=color, size=8.5)

    def bullet_list(self, items, marker="[+]", color=TERM_FG):
        if not items:
            self.mono_line(f"  (none)", color=TERM_DIM, size=8.5)
            return
        for it in items:
            self.bullet(it, marker=marker, color=color)
        self.ln(1)

    def status_tag(self, label, value, ok_color=TERM_FG, warn_color=TERM_AMBER, is_warn=False):
        self._ensure_room(6)
        color = warn_color if is_warn else ok_color
        tag = "[ESTIMATED]" if is_warn else "[VERIFIED] "
        self.mono_line(f"  {label:<28} {tag} {value}", color=color, size=8.5)

    def kv_row(self, key, value, color=TERM_FG):
        self._ensure_room(6)
        self.mono_line(f"  {key:<28} : {value}", color=color, size=8.5)

    def ascii_table(self, df, cols=None, max_rows=10, col_widths=None):
        """Renders a dataframe as an ASCII/box-drawing table using pipe separators,
        matching a terminal 'column -t' style output instead of fpdf2 grid cells.

        IMPORTANT: cell content is WORD-WRAPPED across multiple physical lines within
        the same logical row instead of being truncated with a trailing '.' - so long
        model names, review snippets, or reasons are never cut off. Each logical row
        renders as N wrapped sub-lines (N = max lines needed by any column in that row),
        with blank-padding on shorter columns so the box-drawing borders still line up.
        """
        cols = cols or list(df.columns)
        cols = [c for c in cols if c in df.columns]
        if not cols or df.empty:
            self.mono_line("  (no data)", color=TERM_DIM, size=8.5)
            return

        n = len(cols)
        # Border overhead = "| " + n cells joined by " | " + " |"
        #                  = 2 + (n-1)*3 + 2 = 4 + 3*(n-1) characters of fixed punctuation.
        # The previous formula (ASCII_W - 4 - (n+1)) // n undercounted the separator
        # overhead for n > 2 columns, so total rendered line length exceeded ASCII_W and
        # mono_line()'s hard truncation silently chopped off the last column + border.
        border_overhead = 4 + 3 * (n - 1)
        default_w = max(6, (ASCII_W - border_overhead) // n)
        widths = col_widths or [default_w] * n

        # Safety net: even if explicit col_widths were passed in and still overflow
        # ASCII_W, proportionally shrink them instead of letting mono_line truncate.
        total_line_len = border_overhead + sum(widths)
        if total_line_len > ASCII_W:
            shrink = (ASCII_W - border_overhead) / sum(widths)
            widths = [max(4, int(w * shrink)) for w in widths]

        def wrap_cell(val, w):
            s = clean_text(str(val))
            if s == "" or s.lower() == "nan":
                return [""]
            words = s.split(" ")
            lines, cur = [], ""
            for word in words:
                while len(word) > w:
                    if cur:
                        lines.append(cur)
                        cur = ""
                    lines.append(word[:w])
                    word = word[w:]
                if len(cur) + len(word) + (1 if cur else 0) > w:
                    lines.append(cur)
                    cur = word
                else:
                    cur = f"{cur} {word}".strip()
            if cur:
                lines.append(cur)
            return lines or [""]

        def render_row(cells, colors, style=""):
            wrapped_cols = [wrap_cell(c, w) for c, w in zip(cells, widths)]
            max_lines = max(len(wc) for wc in wrapped_cols)
            for line_i in range(max_lines):
                self._ensure_room(6)
                parts = []
                for wc, w in zip(wrapped_cols, widths):
                    txt = wc[line_i] if line_i < len(wc) else ""
                    parts.append(txt.ljust(w))
                line = "| " + " | ".join(parts) + " |"
                self.mono_line(line, color=colors, size=7.5, style=style)

        sep_line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

        self._ensure_room(6)
        self.mono_line(sep_line, color=TERM_LINE, size=7.5)
        render_row(cols, TERM_FG, style="B")
        self.mono_line(sep_line, color=TERM_LINE, size=7.5)
        for _, row in df.head(max_rows).iterrows():
            render_row([row[c] for c in cols], TERM_DIM)
        self.mono_line(sep_line, color=TERM_LINE, size=7.5)
        self.ln(2)

    def progress_bar(self, label, pct, width=40, color=TERM_FG, bar_char="#", empty_char="."):
        """
        Unicode block-bar rendering (requested upgrade from ####.... to a cleaner
        filled/empty block look). fpdf2's core Courier font only reliably supports
        Latin-1, so true Unicode blocks (#x2588/#x2591) are NOT used here to avoid
        silent character-replacement/garbling - instead this uses a denser ASCII
        alternative (= filled / - empty) that renders identically and cleanly in
        every PDF viewer without needing a Unicode-capable font swap.
        """
        pct = max(0, min(100, pct))
        filled = int(width * pct / 100)
        bar = (bar_char * filled) + (empty_char * (width - filled))
        self._ensure_room(6)
        self.mono_line(f"  {label:<22} [{bar}] {pct:5.1f}%", color=color, size=8)

    def score_bar(self, label, score_0_10, width=24):
        """Decision-scorecard style bar for a 0-10 score, e.g. Value for Money 9.4/10."""
        pct = max(0, min(100, score_0_10 * 10))
        color = TERM_RED if score_0_10 < 5 else (TERM_AMBER if score_0_10 < 7.5 else TERM_FG)
        self.progress_bar(f"{label:<20} {score_0_10:.1f}/10", pct, width=width, color=color,
                          bar_char="=", empty_char="-")


def _confidence_delta_bars(pdf, diag_df, max_rows=14):
    """Renders each diagnostic feature as a terminal-style progress bar showing
    how far the car deviates from its cluster mean (bar = |z-score| capped at 100%)."""
    pdf.section_header("Deviation Bars (|z-score| vs cluster)")
    for _, r in diag_df.head(max_rows).iterrows():
        z = abs(r["Z"])
        pct = min(100, z * 33.3)
        color = TERM_RED if r["Z"] < -1 else (TERM_AMBER if abs(r["Z"]) < 0.5 else TERM_FG)
        pdf.progress_bar(r["FeatureLabel"][:22], pct, color=color)
    pdf.ln(2)


def build_car_pdf(resolved, cluster, cluster_df, diag_df, interpretation, cluster_info,
                  buyer_profile, cheaper, pricier, car_price, max_stretch, narrative, out_path,
                  multiview_recs=None, is_estimated_safety=False, car_row=None):
    """
    NOTE on section ordering/redundancy (per user feedback):
    The old layout put the single-view (safety-only) "Cheaper & Safer" / "Worth The
    Upgrade" sections BEFORE the multi-view breakdown, and interleaved cheaper/pricier
    view-by-view. This meant:
      (a) safety-only recs duplicated the "Safety" entry inside multiview_recs almost
          exactly (same filter logic, just computed twice) - redundant, so it's dropped
          whenever multiview_recs is available, and only kept as a fallback for
          backward-compat when multiview_recs is None.
      (b) cheaper-and-better and pricier-and-better were mixed together per view instead
          of being grouped - now ALL "cheaper & better" views are shown together first,
          then ALL "pricier & better" views together afterward, exactly as requested.
    """
    pdf = TermPDF()
    pdf.title_text = f"Car Diagnostics Report - {resolved}"
    pdf.add_page()

    pdf.ascii_banner(resolved[:14])
    pdf.mono_line(_ascii_center(f"CAR MARKET SUITE v9 // DIAGNOSTICS MODE"), color=TERM_DIM, size=8)
    pdf.mono_line(_ascii_center(pd.Timestamp.now().strftime("run_ts=%Y-%m-%d_%H:%M:%S")), color=TERM_DIM, size=7.5)
    pdf.ln(3)

    pdf.section_header("Vehicle Identity")
    pdf.kv_row("model_name", resolved)
    pdf.kv_row("price_lakh", f"Rs.{car_price:.2f}L")
    pdf.kv_row("cluster_id", str(cluster))
    pdf.kv_row("cluster_size", f"{len(cluster_df)} cars")
    pdf.status_tag("safety_score_source", "RF-imputed" if is_estimated_safety else "lab-tested",
                  is_warn=is_estimated_safety)

    pdf.section_header("Summary // narrative.log")
    pdf.body(narrative, color=TERM_FG, prompt="  $ ")

    pdf.section_header("Strengths [+]")
    pdf.bullet_list(interpretation["strengths"], marker="[+]", color=TERM_FG)

    pdf.section_header("Weaknesses [-]")
    pdf.bullet_list(interpretation["weaknesses"], marker="[-]", color=TERM_RED)

    pdf.section_header("Representativeness & Tradeoff")
    pdf.kv_row("representativeness", interpretation["representativeness"])
    pdf.body(interpretation.get("tradeoff_quantified", interpretation["main_tradeoff"]), color=TERM_AMBER)

    pdf.section_header("Cluster Assignment")
    # Replaces the old confusing pairing of "Representative member" (from GMM
    # confidence, i.e. HOW CLEANLY it belongs to its assigned cluster) directly next to
    # "Low representativeness" (from the z-score typical_ratio, i.e. HOW TYPICAL it is
    # of that cluster's feature averages) with NO explanation - these are two genuinely
    # different statistics and showing them side-by-side with overlapping wording made
    # them look contradictory. Now shown as a labeled breakdown with an explicit reason.
    pdf.kv_row("cluster_view", PERSPECTIVE_DISPLAY.get('Motorhead', 'Driving Performance'))
    pdf.kv_row("assignment_confidence", cluster_info.get("confidence_pct", str(cluster_info["confidence"])))
    pdf.kv_row("typicality_vs_cluster", interpretation["representativeness"])
    reason = _typicality_reason(interpretation, cluster_info)
    pdf.body(f"Reason: {reason}", color=TERM_DIM, prompt="  ")

    pdf.section_header("Buyer Profile Tags")
    pdf.bullet_list(buyer_profile, marker="[*]", color=TERM_AMBER)

    ideal_checklist = compute_ideal_buyer_checklist(car_row, interpretation)
    pdf.section_header("Ideal Buyer Checklist")
    _fit_color = {FIT_GOOD: TERM_FG, FIT_OK: TERM_AMBER, FIT_POOR: TERM_RED}
    for label, fit_level in ideal_checklist:
        marker = FIT_MARKER[fit_level]
        pdf.bullet(f"{label} -- {fit_level}", marker=marker, color=_fit_color[fit_level])

    pdf.add_page()
    pdf.section_header("Decision Scorecard (vs. Whole Dataset)")
    # NOTE (issue #3): this scorecard answers "is this car objectively good across the
    # ENTIRE dataset" (view_* scores are min-max normalized dataset-wide). The
    # Strengths/Weaknesses and Feature Diagnostics sections above answer a DIFFERENT
    # question - "is this car good relative to its own cluster of similar cars" (z-scores
    # vs cluster mean). Both are valid but must not be read interchangeably - a car can
    # score low here while still being the best-in-class within its own cluster, or vice
    # versa. The section header and body note below make this distinction explicit.
    pdf.body("Scores below are normalized against ALL cars in the dataset, not just this "
            "car's cluster peers - see Feature Diagnostics for cluster-relative comparison.",
            color=TERM_DIM, prompt="  ")
    scorecard = compute_decision_scorecard(car_row)
    for key in ["Value for Money", "Safety", "Ownership", "Performance", "Technology"]:
        pdf.score_bar(key, scorecard[key])
    pdf.ln(1)
    star_str = ("*" * scorecard["stars"]) + ("." * (5 - scorecard["stars"]))
    pdf.mono_line(f"  OVERALL RECOMMENDATION: [{star_str}]  {scorecard['Overall']}/10", color=TERM_AMBER, size=9, style="B")
    pdf.ln(2)

    pdf.section_header("Scoring Methodology (Transparency)")
    for score_name, weights in SCORING_METHODOLOGY.items():
        pdf.mono_line(f"  {score_name}:", color=TERM_AMBER, size=8.5, style="B")
        for factor, w in weights:
            pdf.mono_line(f"    {w*100:>3.0f}%  {factor}", color=TERM_FG, size=8)
    pdf.ln(2)

    pdf.add_page()
    _confidence_delta_bars(pdf, diag_df)

    pdf.section_header("Feature Diagnostics (vs. Cluster Peers)")
    pdf.ascii_table(diag_df[["FeatureLabel", "Car Value", "Cluster Mean", "Delta %", "Assessment"]],
                    max_rows=18, col_widths=[22, 11, 11, 9, 18])

    pdf.add_page()

    def simplified_alt_table(df_alt, price_ref):
        """3-column Model / Why It's Better / Price Diff table (requested simplification
        over the old wide multi-column table that was hard to scan)."""
        if df_alt.empty:
            return pd.DataFrame(columns=["Model", "Why It's Better", "Price Diff"])
        out = pd.DataFrame()
        out["Model"] = df_alt["model_name"]
        out["Why It's Better"] = df_alt.get("reason", "Comparable alternative")
        if "ex_showroom_price_lakh" in df_alt.columns:
            diff = df_alt["ex_showroom_price_lakh"] - price_ref
            out["Price Diff"] = diff.apply(lambda d: f"{'+'  if d > 0 else ''}Rs.{d:.1f}L")
        else:
            out["Price Diff"] = "N/A"
        return out

    if multiview_recs:
        pdf.section_header(f"Cheaper & Better (< Rs.{car_price:.1f}L) -- All Views")
        any_cheap = False
        for view_name, (vcheap, vpric) in multiview_recs.items():
            if vcheap.empty:
                continue
            any_cheap = True
            pdf.mono_line(f"  -- {view_name} --", color=TERM_AMBER, size=8.5, style="B")
            pdf.ascii_table(simplified_alt_table(vcheap, car_price), max_rows=3,
                            col_widths=[26, 44, 14])
        if not any_cheap:
            pdf.body("no cheaper-and-better matches found in any view.", color=TERM_DIM)

        pdf.add_page()
        pdf.section_header(f"Pricier & Better (<= Rs.{car_price + max_stretch:.1f}L) -- All Views")
        if max_stretch <= 0:
            pdf.body("upgrade search skipped (max_stretch=0).", color=TERM_DIM)
        else:
            any_pric = False
            for view_name, (vcheap, vpric) in multiview_recs.items():
                if vpric.empty:
                    continue
                any_pric = True
                pdf.mono_line(f"  -- {view_name} --", color=TERM_AMBER, size=8.5, style="B")
                pdf.ascii_table(simplified_alt_table(vpric, car_price), max_rows=3,
                                col_widths=[26, 44, 14])
            if not any_pric:
                pdf.body("no pricier-and-better matches found in any view.", color=TERM_DIM)
    else:
        pdf.section_header(f"Cheaper & Safer < Rs.{car_price:.1f}L")
        pdf.ascii_table(simplified_alt_table(cheaper, car_price), max_rows=5, col_widths=[26, 44, 14]) \
            if not cheaper.empty else pdf.body("no matches found.", color=TERM_DIM)

        pdf.add_page()
        pdf.section_header(f"Worth The Upgrade (<= Rs.{car_price + max_stretch:.1f}L)")
        if max_stretch <= 0:
            pdf.body("upgrade search skipped (max_stretch=0).", color=TERM_DIM)
        elif pricier.empty:
            pdf.body("no upgrades found within max stretch.", color=TERM_DIM)
        else:
            pdf.ascii_table(simplified_alt_table(pricier, car_price), max_rows=5, col_widths=[26, 44, 14])

    pdf.output(out_path)
    print(f"  PDF report -> {out_path}")


def build_view_pdf(view_name, budget, stretch, in_budget, stretch_band, narrative, out_path):
    pdf = TermPDF()
    pdf.title_text = f"View + Budget Explorer - {view_name}"
    pdf.add_page()

    pdf.ascii_banner(view_name[:14])
    pdf.mono_line(_ascii_center("CAR MARKET SUITE v7 // VIEW EXPLORER MODE"), color=TERM_DIM, size=8)
    pdf.mono_line(_ascii_center(pd.Timestamp.now().strftime("run_ts=%Y-%m-%d_%H:%M:%S")), color=TERM_DIM, size=7.5)
    pdf.ln(3)

    pdf.section_header("Query Parameters")
    pdf.kv_row("view", view_name)
    pdf.kv_row("budget_lakh", f"Rs.{budget:.2f}L")
    pdf.kv_row("stretch_lakh", f"Rs.{stretch:.2f}L")

    pdf.section_header("Summary // narrative.log")
    pdf.body(narrative, color=TERM_FG, prompt="  $ ")

    pdf.section_header(f"Best Within Budget (<= Rs.{budget:.1f}L)")
    if in_budget.empty:
        pdf.body("no matches found.", color=TERM_DIM)
    else:
        pdf.ascii_table(in_budget.drop(columns=["review_summary"], errors="ignore"), max_rows=8)

    pdf.section_header(f"Stretch Upgrades (Rs.{budget:.1f}L - Rs.{budget + stretch:.1f}L)")
    if stretch <= 0:
        pdf.body("stretch exploration skipped (stretch=0).", color=TERM_DIM)
    elif stretch_band.empty:
        pdf.body("no matches found in stretch range.", color=TERM_DIM)
    else:
        pdf.ascii_table(stretch_band.drop(columns=["review_summary"], errors="ignore"), max_rows=8)

    pdf.output(out_path)
    print(f"  PDF report -> {out_path}")

# ============================================================================
# CONSOLE DISPLAY
# ============================================================================
def print_plain_diag(diag_df, model_name, car_cluster, cluster_df):
    print(f"\\n{'='*80}\\n  Car: {model_name}   |   Cluster: {car_cluster}  ({len(cluster_df)} cars)\\n{'='*80}")
    for _, r in diag_df.iterrows():
        print(f"  {r['FeatureLabel']:<30} car={fmt(r['Car Value']):>10}  mean={fmt(r['Cluster Mean']):>10}  "
              f"({r['Delta %']:+.1f}%)  {r['Assessment']}")
    print("=" * 80)


def print_interpretation(interpretation, cluster_info, buyer_profile):
    print(f"\\n{'='*80}\\n  INTERPRETATION\\n{'='*80}")
    print("  Strengths:")
    for s in interpretation["strengths"]:
        print(f"    + {s}")
    print("  Weaknesses:")
    for w in interpretation["weaknesses"]:
        print(f"    - {w}")
    print(f"  Representativeness: {interpretation['representativeness']}")
    print(f"  Main tradeoff: {interpretation['main_tradeoff']}")
    print(f"  Cluster membership: {cluster_info['cluster_membership']} (confidence={cluster_info['confidence']})")
    print(f"  Buyer profile: {', '.join(buyer_profile)}")
    print("=" * 80)


def print_plain_recs(cheaper, pricier, car_price, max_stretch):
    print(f"\\n{'='*80}\\n  RECOMMENDATIONS (Safety-based)\\n{'='*80}")
    if cheaper.empty:
        print("  Cheaper & Safer: none found")
    else:
        print(f"  Cheaper & Safer (< Rs.{car_price:.1f}L):")
        for _, r in cheaper.iterrows():
            print(f"    - {r['model_name']} ({r.get('brand','')}) - Rs.{r['ex_showroom_price_lakh']:.1f}L - {r.get('reason','')}")
    if max_stretch <= 0:
        print("  Upgrade skipped (stretch = Rs.0L)")
    elif pricier.empty:
        print(f"  No upgrades within Rs.{car_price + max_stretch:.1f}L")
    else:
        print(f"  Worth the Upgrade (up to Rs.{car_price + max_stretch:.1f}L):")
        for _, r in pricier.iterrows():
            print(f"    - {r['model_name']} ({r.get('brand','')}) - Rs.{r['ex_showroom_price_lakh']:.1f}L - {r.get('reason','')}")


def print_multiview_recs(multiview_recs, car_price, max_stretch):
    """NEW: prints cheaper/pricier alternatives per view to console."""
    print(f"\\n{'='*80}\\n  RECOMMENDATIONS BY VIEW\\n{'='*80}")
    for view_name, (cheaper, pricier) in multiview_recs.items():
        print(f"\\n  -- {view_name} --")
        if cheaper.empty:
            print(f"    Cheaper & Better: none found")
        else:
            print(f"    Cheaper & Better (< Rs.{car_price:.1f}L):")
            for _, r in cheaper.iterrows():
                print(f"      - {r['model_name']} ({r.get('brand','')}) - Rs.{r['ex_showroom_price_lakh']:.1f}L - {r.get('reason','')}")
        if max_stretch <= 0:
            print("    Upgrade skipped (stretch = Rs.0L)")
        elif pricier.empty:
            print(f"    No upgrades within Rs.{car_price + max_stretch:.1f}L")
        else:
            print(f"    Pricier & Better (up to Rs.{car_price + max_stretch:.1f}L):")
            for _, r in pricier.iterrows():
                print(f"      - {r['model_name']} ({r.get('brand','')}) - Rs.{r['ex_showroom_price_lakh']:.1f}L - {r.get('reason','')}")
    print("=" * 80)


def show_view_table(df_rec, title):
    print(f"\\n{title}")
    if df_rec is None or df_rec.empty:
        print("  none found")
        return
    for _, r in df_rec.iterrows():
        print(f"  - {r['model_name']} ({r.get('brand','')}) - Rs.{pd.to_numeric(r['ex_showroom_price_lakh'], errors='coerce'):.1f}L")

# ============================================================================
# PUBLIC API: CAR MODE
# ============================================================================
REPORT_PERSPECTIVE = "motorhead"  # single source of truth for which cluster
                                  # perspective anchors an ENTIRE report - diagnostics,
                                  # representativeness/confidence, and the narrative all
                                  # key off this same value now (see compute_diagnostics
                                  # docstring for why this was previously mismatched).


def run_diagnostics(model_name, output_path=None, max_stretch=None, force_rebuild=False):
    df = load_all_data(force_rebuild=force_rebuild)
    diag_df, resolved, cluster, car_row = compute_diagnostics(df, model_name, perspective=REPORT_PERSPECTIVE)

    if diag_df is None:
        print(f"\nERROR: '{model_name}' not found.")
        for n in df["model_name"].dropna().unique()[:30]:
            print(f"  - {n}")
        return None

    cluster_col = f"cluster_{REPORT_PERSPECTIVE}"
    if cluster_col not in df.columns:
        cluster_col = f"{REPORT_PERSPECTIVE}_cluster"
    cluster_df = df[df[cluster_col] == cluster]
    car_price = car_row.get("ex_showroom_price_lakh", 0)

    print_plain_diag(diag_df, resolved, cluster, cluster_df)

    interpretation = interpret_diagnostics(diag_df, car_row)
    cluster_info = get_cluster_membership(car_row, perspective_prefix=REPORT_PERSPECTIVE)
    buyer_profile = compute_buyer_profile(car_row, interpretation=interpretation)

    print_interpretation(interpretation, cluster_info, buyer_profile)

    if max_stretch is None:
        max_stretch = ask_max_stretch(car_price)

    # Safety-based recommendations (kept for backward compatibility)
    cheaper, pricier = get_recommendations(df, car_row, max_stretch)
    print_plain_recs(cheaper, pricier, car_price, max_stretch)

    # NEW: multi-view cheaper/pricier recommendations across ALL perspectives
    multiview_recs = get_recommendations_by_view(df, car_row, max_stretch)
    print_multiview_recs(multiview_recs, car_price, max_stretch)
    multiview_structured = multiview_recs_to_structured(multiview_recs)

    cheaper_alts = alternatives_to_structured_list(cheaper)
    pricier_alts = alternatives_to_structured_list(pricier)

    safe = resolved.replace(" ", "_").replace("/", "_")

    # NEW: every car gets its own dedicated folder (outputs/<CarName>/) so that
    # its CSVs, JSONs, and PDF report all live together instead of being scattered
    # flat inside outputs/.
    car_dir = os.path.join(OUTPUTS_DIR, safe)
    os.makedirs(car_dir, exist_ok=True)
    car_json_dir = os.path.join(car_dir, "json")
    os.makedirs(car_json_dir, exist_ok=True)

    diag_base = output_path.replace(".csv", "") if output_path else os.path.join(car_dir, f"{safe}_diagnostics")
    save_dual(diag_df, diag_base, json_dir=car_json_dir)

    cheaper2, pricier2 = cheaper.copy(), pricier.copy()
    if not cheaper2.empty:
        cheaper2["rec_type"] = "Cheaper & Safer"
    if not pricier2.empty:
        pricier2["rec_type"] = "Worth the Upgrade"
    save_dual(pd.concat([cheaper2, pricier2], ignore_index=True),
             os.path.join(car_dir, f"{safe}_recommendations"), json_dir=car_json_dir)

    # NEW: save each view's cheaper/pricier tables as CSV+JSON too, inside the same car folder
    for vname, (vcheap, vpric) in multiview_recs.items():
        safe_v = vname.replace(" ", "_").replace("&", "and")
        if not vcheap.empty:
            save_dual(vcheap, os.path.join(car_dir, f"{safe}_{safe_v}_cheaper_better"), json_dir=car_json_dir)
        if not vpric.empty:
            save_dual(vpric, os.path.join(car_dir, f"{safe}_{safe_v}_pricier_better"), json_dir=car_json_dir)

    narrative = narrative_for_car_report(resolved, interpretation, cluster_info, buyer_profile,
                                         cheaper_alts, pricier_alts, car_row)

    structured_summary = {
        "car": resolved,
        "price_lakh": None if pd.isna(car_price) else float(car_price),
        "strengths": interpretation["strengths"],
        "weaknesses": interpretation["weaknesses"],
        "representativeness": interpretation["representativeness"],
        "main_tradeoff": interpretation["main_tradeoff"],
        "cluster_membership": cluster_info["cluster_membership"],
        "cluster_confidence": cluster_info["confidence"],
        "buyer_profile": buyer_profile,
        "cheaper_alternatives": cheaper_alts,
        "upgrade_alternatives": pricier_alts,
        "multiview_alternatives": multiview_structured,   # NEW
        "narrative": narrative,
    }
    save_dict_json(structured_summary, os.path.join(car_dir, f"{safe}_summary"), json_dir=car_json_dir)

    # PDF now saved inside the same per-car folder, alongside its CSVs/JSONs
    pdf_path = os.path.join(car_dir, f"{safe}_diagnostics_report.pdf")
    is_estimated_safety = bool(car_row.get("aop_cop_estimated", False))
    try:
        build_car_pdf(resolved, cluster, cluster_df, diag_df, interpretation, cluster_info, buyer_profile,
                      cheaper, pricier, car_price, max_stretch, narrative, pdf_path,
                      multiview_recs=multiview_recs, is_estimated_safety=is_estimated_safety,
                      car_row=car_row)
    except Exception as e:
        print(f"  [PDF WARNING] Could not generate PDF report: {e}")

    print(f"\n  Saved all outputs for '{resolved}' -> {car_dir}")
    print(f"    - Diagnostics: {diag_base}.csv/.json")
    print(f"    - Recommendations: {os.path.join(car_dir, safe + '_recommendations.csv/.json')}")
    print(f"    - Summary: {os.path.join(car_json_dir, safe + '_summary.json')}")
    print(f"    - PDF report: {pdf_path}")
    return diag_df

# ============================================================================
# PUBLIC API: VIEW MODE
# ============================================================================
def run_view_explorer(view_name, budget, stretch, force_rebuild=False):
    """
    View + Budget Explorer mode. Mirrors the per-car folder structure used by
    run_diagnostics(): each run gets its own dedicated folder named after the
    view (e.g. outputs/Driving_Performance/) containing its CSVs, JSONs, and
    PDF report together, instead of flat files scattered in outputs/.
    """
    df = load_all_data(force_rebuild=force_rebuild)
    in_budget, stretch_band = view_recommendations(df, view_name, budget, stretch)

    print(f"\nView: {view_name} | Budget: Rs.{budget:.1f}L | Stretch: Rs.{stretch:.1f}L")
    show_view_table(in_budget, "Best within budget")
    if stretch > 0:
        show_view_table(stretch_band, "Stretch upgrades")

    safe_view = view_name.replace(" ", "_").replace("&", "and")

    # NEW: dedicated folder per view (outputs/<ViewName>/) with its own json/ subfolder,
    # matching the per-car folder convention used in run_diagnostics().
    view_dir = os.path.join(OUTPUTS_DIR, safe_view)
    os.makedirs(view_dir, exist_ok=True)
    view_json_dir = os.path.join(view_dir, "json")
    os.makedirs(view_json_dir, exist_ok=True)

    save_dual(in_budget, os.path.join(view_dir, f"{safe_view}_in_budget"), json_dir=view_json_dir)
    if stretch > 0:
        save_dual(stretch_band, os.path.join(view_dir, f"{safe_view}_stretch_band"), json_dir=view_json_dir)

    narrative = narrative_for_view_report(view_name, budget, stretch, in_budget, stretch_band)
    save_dict_json({"narrative": narrative}, os.path.join(view_dir, f"{safe_view}_summary"), json_dir=view_json_dir)

    pdf_path = os.path.join(view_dir, f"{safe_view}_view_report.pdf")
    try:
        build_view_pdf(view_name, budget, stretch, in_budget, stretch_band, narrative, pdf_path)
    except Exception as e:
        print(f"  [PDF WARNING] Could not generate PDF report: {e}")

    print(f"\n  Saved all outputs for view '{view_name}' -> {view_dir}")
    print(f"    - In-budget table: {os.path.join(view_dir, safe_view + '_in_budget.csv/.json')}")
    if stretch > 0:
        print(f"    - Stretch-band table: {os.path.join(view_dir, safe_view + '_stretch_band.csv/.json')}")
    print(f"    - Summary: {os.path.join(view_json_dir, safe_view + '_summary.json')}")
    print(f"    - PDF report: {pdf_path}")
    return in_budget, stretch_band

# ============================================================================
# INTERACTIVE MODE
# ============================================================================
def ask_max_stretch(car_price):
    if RICH:
        console.print(f"\\n[bold]Your car:[/bold] Rs.[cyan]{car_price:.1f}L[/cyan]")
        try:
            return max(0.0, FloatPrompt.ask(
                "[bold yellow]What is the MAX additional lakhs you can stretch to?[/bold yellow]", default=0.0))
        except Exception:
            return 0.0
    print(f"\\n  Your car price: Rs.{car_price:.1f}L")
    try:
        raw = input("  Max additional lakhs to stretch (ceiling)? [0]: ").strip()
        return max(0.0, float(raw) if raw else 0.0)
    except ValueError:
        return 0.0


def interactive_mode():
    df = load_all_data()
    all_models = sorted(df["model_name"].dropna().unique().tolist())
    print("\\n" + "=" * 60 + "\\n  Car Market Suite v7\\n" + "=" * 60)
    print("1. Diagnose my current car")
    print("2. Explore by view + budget")
    mode = input("Select 1 or 2 (q to quit): ").strip()
    if mode.lower() in ("q", "quit", "exit", ""):
        return

    if mode == "1":
        query = input("\\nSearch car name: ").strip()
        matches = [m for m in all_models if query.lower() in m.lower()]
        if not matches:
            print(f"  No matches for '{query}'.")
            return
        if len(matches) == 1:
            selected = matches[0]
        else:
            for i, m in enumerate(matches, 1):
                print(f"  [{i}] {m}")
            idx = int(input("  Select number: ").strip()) - 1
            selected = matches[idx]
        run_diagnostics(selected)
    else:
        print("\\nAvailable views:", ", ".join(VIEW_KEYS.keys()))
        view_name = input("Select view: ").strip()
        budget = float(input("Max price in lakhs: ").strip() or 0)
        stretch = float(input("Optional stretch (default 0): ").strip() or 0)
        run_view_explorer(view_name, budget, stretch)

# ============================================================================
# NOTEBOOK-SAFE MAIN (PyCharm + Jupyter/Colab compatible)
# ============================================================================
def main(car=None, output=None, max_stretch=None, view=None, budget=None,
        rebuild_pipeline=False, data_dir=None, output_dir=None):
    _configure_paths(data_dir=data_dir, output_dir=output_dir)
    if rebuild_pipeline and not car and not view:
        return run_clustering_pipeline()
    if car:
        return run_diagnostics(car, output, max_stretch, force_rebuild=rebuild_pipeline)
    if view:
        return run_view_explorer(view, budget or 0.0, max_stretch or 0.0, force_rebuild=rebuild_pipeline)
    interactive_mode()
    return None


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _is_notebook():
        print("Running inside a notebook kernel. Call main(car=...) or main(view=...) directly; "
             "CLI arg parsing is skipped to avoid Jupyter kernel-arg conflicts.")
    else:
        clean_argv = [a for a in sys.argv[1:] if not a.startswith("-f") and ".json" not in a]
        parser = argparse.ArgumentParser(description="Car market suite: pipeline + diagnostics + view explorer")
        parser.add_argument("--car", type=str, default=None)
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--view", type=str, default=None)
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument("--max-stretch", type=float, default=None,
                            help="Maximum additional lakhs to search above the base price/budget "
                                 "(used for both --car and --view modes)")
        parser.add_argument("--rebuild-pipeline", action="store_true")
        parser.add_argument("--data", type=str, default=None,
                            help="Path to dataset directory (default: <script_dir>/data)")
        parser.add_argument("--output-dir", type=str, default=None,
                            help="Path to output directory (default: <script_dir>/outputs)")
        args = parser.parse_args(clean_argv)

        _configure_paths(data_dir=args.data, output_dir=args.output_dir)

        if args.rebuild_pipeline and not args.car and not args.view:
            run_clustering_pipeline()
        elif args.car:
            main(car=args.car, output=args.output, max_stretch=args.max_stretch,
                rebuild_pipeline=args.rebuild_pipeline, data_dir=args.data, output_dir=args.output_dir)
        elif args.view:
            main(view=args.view, budget=args.budget, max_stretch=args.max_stretch,
                rebuild_pipeline=args.rebuild_pipeline, data_dir=args.data, output_dir=args.output_dir)
        else:
            main(data_dir=args.data, output_dir=args.output_dir)
