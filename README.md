# 🚗 Car Market Suite v7

> **An end-to-end automotive intelligence platform that combines machine learning, explainable AI, recommendation systems, and local LLMs to help buyers make data-driven vehicle decisions.**

---

# Overview

Car Market Suite is an AI-powered analytics platform that transforms raw automotive datasets into meaningful buyer insights.

Instead of relying on isolated specifications or subjective reviews, the system combines:

* Machine Learning
* Explainable AI
* Vehicle Clustering
* Recommendation Systems
* Statistical Diagnostics
* Local Large Language Models
* Automated Report Generation

to create an interpretable, end-to-end vehicle analysis pipeline.

---

# Features

## 1. Data Integration Pipeline

The system automatically combines information from multiple independent datasets into a unified master dataset.

Integrated sources include:

* Safety ratings
* Vehicle specifications
* Ownership costs
* Feature matrices
* Expert reviews

The pipeline automatically:

* Cleans data
* Merges datasets
* Handles missing values
* Engineers additional features
* Produces a unified dataset for downstream analysis

---

## 2. Automatic Feature Engineering

The project derives several high-level metrics from raw data, including:

* Power-to-weight ratio
* Combined safety score
* View-specific scores
* AWD detection
* Price normalization
* Fuel-efficiency metrics

These engineered features improve clustering quality and recommendation accuracy.

---

## 3. Multi-Perspective Vehicle Clustering

Rather than clustering every vehicle into a single category, the platform creates independent clusters from different buyer perspectives.

### Safety

Clusters vehicles according to:

* Adult Occupant Protection
* Child Occupant Protection
* Airbags
* ESC
* Crash-test performance

---

### Ownership

Groups cars by ownership experience using:

* Fuel efficiency
* Service costs
* Warranty
* Purchase price

---

### Tech & Comfort

Clusters vehicles using:

* Infotainment
* Convenience
* Driver assistance
* Interior technology
* Comfort features

---

### Driving Performance

Groups vehicles based on:

* Horsepower
* Torque
* Power-to-weight ratio
* Engine size
* Ground clearance
* AWD availability

---

### Overall Market Position

Creates a holistic cluster using representative features from all perspectives.

---

# Intelligent Cluster Selection

Instead of using a fixed number of clusters, the platform automatically evaluates:

* Silhouette Score
* Information Entropy
* Mutual Information

to determine whether additional clusters provide meaningful separation.

---

# Hybrid Clustering

Depending on the available data, the pipeline automatically switches between:

* K-Means
* K-Prototypes

allowing both numerical and categorical features to be clustered effectively.

---

# Gaussian Mixture Diagnostics

Each vehicle receives:

* Cluster probability
* Membership confidence
* Boundary detection

Rather than simply assigning a cluster, the model quantifies how strongly each vehicle belongs to it.

---

# Explainable Vehicle Diagnostics

For every selected vehicle, the platform compares it against similar vehicles in its cluster.

The report includes:

* Cluster averages
* Standard deviations
* Z-scores
* Percentage differences
* Statistical deviation categories

Every feature is interpreted as:

* Typical
* Slightly Above Average
* Slightly Below Average
* Moderately Above Average
* Moderately Below Average
* Significantly Above Average
* Significantly Below Average

making statistical outputs understandable for non-technical users.

---

# Automated Strength & Weakness Detection

Instead of requiring an LLM to infer insights, the platform computes them directly.

Examples include:

### Strengths

* Above-average safety
* Better fuel efficiency
* Higher power
* Lower ownership cost

### Weaknesses

* High purchase price
* Lower warranty
* Poor efficiency
* Weak crash-test performance

This significantly reduces hallucination in generated reports.

---

# Trade-off Analysis

The system automatically identifies the primary trade-off of every vehicle.

Example:

Driving Performance vs Ownership

or

Technology vs Safety

rather than presenting isolated statistics.

---

# Buyer Profile Generation

Each vehicle is automatically matched to buyer personas such as:

* Safety-focused buyers
* Budget-conscious buyers
* Driving enthusiasts
* Urban commuters

using normalized scoring across multiple perspectives.

---

# Intelligent Recommendation Engine

The recommendation engine identifies vehicles that offer better value than the selected vehicle.

Two recommendation categories are generated.

## Cheaper & Better

Vehicles costing less while outperforming the selected car.

---

## Worth the Upgrade

Vehicles within a configurable stretch budget that provide meaningful improvements.

Each recommendation also explains **why** it was selected.

Example:

* Better safety
* Better mileage
* Lower ownership cost
* Higher performance

---

# Review Matching Engine

Vehicle review datasets often store only generic model names while datasets contain specific trims.

The platform bridges this gap through:

* Regex normalization
* Trim removal
* Fuzzy string matching
* Similarity scoring

allowing reviews to be attached even when naming conventions differ.

---

# View-Based Vehicle Explorer

Instead of searching for one specific vehicle, users can explore vehicles based on priorities.

Available views include:

* Safety
* Ownership
* Tech & Comfort
* Driving Performance
* Overall

Users specify:

* Budget
* Stretch budget

and receive ranked recommendations.

---

# Local LLM Integration

The platform integrates with Ollama for fully local inference.

The LLM is responsible for generating:

* Narrative summaries
* Buyer-friendly explanations
* Market positioning
* Recommendation context

If the LLM is unavailable, the application automatically falls back to deterministic template generation.

---

# Structured AI Prompting

Rather than sending raw numerical data to the LLM, the platform constructs a structured interpretation layer containing:

* Strengths
* Weaknesses
* Buyer profile
* Trade-offs
* Cluster confidence
* Alternative recommendations

This dramatically reduces hallucination while improving consistency.

---

# Automated Report Generation

Every analysis produces multiple outputs.

### CSV

Machine-readable tables.

### JSON

Structured outputs for APIs or downstream applications.

### PDF

Professionally formatted reports containing:

* Executive summary
* Vehicle diagnostics
* Statistical comparisons
* Recommendations
* Buyer profile
* AI-generated narrative

---

# Interactive Console

The application supports:

* Interactive vehicle search
* Vehicle diagnostics
* Budget explorer
* Command-line execution
* Notebook execution

making it usable in both development and production environments.

---

# Technology Stack

### Languages

* Python

### Machine Learning

* Scikit-learn
* Gaussian Mixture Models
* K-Means
* K-Prototypes

### Data Processing

* Pandas
* NumPy
* SciPy

### Visualization

* Matplotlib

### Report Generation

* FPDF

### Local AI

* Ollama
* Qwen

---

# Repository Structure

```
data/
outputs/
outputs/json/

car_market_suite.py

safety_ratings_v2.csv
car_features_v2.csv
India_Car_Features_Matrix.csv
car_ownership_costs.csv
reviews_cars.csv

llm_instructions.md
```

---

# Future Improvements

* Streamlit Web Dashboard
* FastAPI Backend
* RAG-based automotive knowledge retrieval
* Multi-LLM support
* Real-time vehicle pricing APIs
* Dealer inventory integration
* Image-based vehicle comparison
* Personalized recommendation learning
* Interactive visual analytics
* Cloud deployment

