# Car Market Suite v7

**End-to-End Vehicle Analytics Pipeline** — Clustering + Diagnostics + Reviews + Structured LLM Interpretation + Local LLM Narration + PDF Reports

A single-file Python system that turns raw Indian car market datasets (safety ratings, features, ownership costs, reviews) into personalized diagnostic reports and budget-aware recommendations, narrated by a local LLM and exported as polished PDFs.

---

## Overview

Car Market Suite runs in two stages:

- **Stage 1 — Clustering Pipeline** (auto-runs if needed): Merges safety, features, feature-matrix, and ownership CSVs, then runs unsupervised clustering (KMeans / K-Prototypes + Gaussian Mixture Models) across five perspectives — Safety, Ownership, Tech & Comfort, Driving Performance, and Overall — producing `outputs/master_clustered.csv`.
- **Stage 2 — Report Generation**: Runs in one of two modes:
  - **Car mode** — diagnoses a specific car against its cluster peers.
  - **View mode** — explores the best cars for a chosen perspective (e.g. Safety) within a budget, plus a "stretch" upgrade band.

Each report is saved as CSV + JSON and rendered into a Unicode-safe PDF, narrated by a local LLM (Ollama) when reachable, with a deterministic templated fallback otherwise.

## Key Features

- **Multi-perspective clustering** — Safety, Ownership, Tech & Comfort, Driving Performance, Overall, using KMeans, optional K-Prototypes (mixed numeric/categorical), and GMM for soft cluster-confidence scoring.
- **Structured pre-interpretation layer** — precomputed strengths, weaknesses, representativeness, and main tradeoff so the LLM barely has to reason.
- **Fuzzy review matching** — bridges general model names in reviews (e.g. "Swift") to specific trims in the car dataset (e.g. "Swift LXi") via regex + fuzzy matching cascade.
- **Deviation categories** — z-score-based labels (Typical / Slightly / Moderately / Significantly above-or-below average) instead of raw deltas.
- **Buyer profile inference** — classifies cars into Safety-focused, Urban commuter, Budget-conscious, or Driving enthusiast segments.
- **Explained alternatives** — every recommended car carries a plain-English reason (higher safety, lower cost, better efficiency).
- **Local LLM narration** — integrates with Ollama (`qwen3:4b` by default) for natural-language summaries, with automatic fallback if unreachable.
- **Crash-hardened PDF export** — custom `fpdf2` wrapper prevents "not enough horizontal space" rendering errors via safe-wrapping and drawable-area checks.
- **Dual CSV + JSON outputs** for every report, machine-readable and human-readable.

## Requirements

```bash
pip install numpy pandas scipy scikit-learn fpdf2 requests matplotlib
pip install kmodes      # optional, enables K-Prototypes mixed clustering
pip install rich        # optional, enables styled CLI prompts
```

For LLM narration, install and run [Ollama](https://ollama.com) locally:

```bash
ollama serve
ollama pull qwen3:4b
```

If Ollama isn't reachable, the script automatically falls back to templated narratives — no functionality is lost.

## Input Data

Place these CSV files in your working directory (update `PROJECT_DIR` in the config section if needed):

| File | Purpose |
|---|---|
| `safety_ratings_v2.csv` | Crash-test / safety scores (AOP, COP, airbags, ESC) |
| `car_features_v2.csv` | Core specs (power, torque, engine, price) |
| `India_Car_Features_Matrix.csv` | Optional detailed feature matrix (tech/comfort) |
| `car_ownership_costs.csv` | Fuel efficiency, service cost, warranty |
| `reviews_cars.csv` | Expert/owner review text (general model names) |

## Usage

### Terminal / PyCharm

```bash
# Diagnose a specific car
python car_market_suite.py --car "Nexon Smart"

# With an upgrade budget stretch (in lakhs)
python car_market_suite.py --car "Swift LXi" --stretch 3.0

# Explore best cars for a view within a budget
python car_market_suite.py --view "Driving Performance" --budget 12 --stretch 3

# Force-rebuild the clustering pipeline
python car_market_suite.py --rebuild-pipeline

# Interactive menu (no args)
python car_market_suite.py
```

### Jupyter / Google Colab

```python
from car_market_suite import main

main(car="Nexon Smart")
main(view="Safety", budget=12.0, stretch=3.0)
main(rebuild_pipeline=True)
```

## Available Views

`Safety` · `Ownership` · `Tech & Comfort` · `Driving Performance` · `Overall`

## Output Structure

```
outputs/
├── master_clustered.csv          # Stage 1 master table
├── <car>_diagnostics.csv/.json   # Feature-vs-cluster-mean diagnostics
├── <car>_recommendations.csv     # Cheaper & pricier alternatives
├── <car>_diagnostics_report.pdf  # Full PDF report
├── <view>_in_budget.csv          # View mode: best-in-budget cars
├── <view>_stretch_band.csv       # View mode: upgrade options
├── <view>_view_report.pdf
└── json/                         # JSON mirror of all CSV outputs
```

## How Diagnostics Work

1. The target car is matched (exact or fuzzy) against the master dataset.
2. It's compared against its own cluster peers using z-scores across price, safety, efficiency, and performance features.
3. Deviations are labeled (Typical / Slightly / Moderately / Significantly above-or-below average) and rolled up into strengths, weaknesses, and a single main tradeoff.
4. GMM cluster-confidence indicates how representative the car is of its cluster (Representative member → Boundary vehicle).
5. Cheaper-and-safer and worth-the-upgrade alternatives are computed and explained.
6. A local LLM (or templated fallback) weaves the top strength, top weakness, and tradeoff into a narrative.

## Configuration

Key tunables live at the top of the script:

| Variable | Purpose |
|---|---|
| `PREFERRED_K` | Default number of clusters per perspective |
| `PREFER_THRESHOLD` | Silhouette-score margin to override preferred K |
| `N_RECS` | Number of recommendations per category |
| `OLLAMA_MODEL` | Local LLM model name |
| `LLM_ENABLED` | Toggle LLM narration on/off |
| `FUZZY_REVIEW_THRESHOLD` | Minimum similarity score for review matching |

## Project Structure

Single-file design (`car_market_suite.py`) organized into clear sections: universal helpers, fuzzy matching, clustering pipeline, review/view-score loaders, diagnostics logic, interpretation layer, LLM integration, PDF rendering, console display, and public API (`run_diagnostics`, `run_view_explorer`, `main`).

## License

Add your preferred license here (e.g. MIT).
