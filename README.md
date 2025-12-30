# Advanced Causal Impact Framework

## Overview
This project implements a robust Causal Impact framework designed to automatically source and filter the best control series from a library of candidates. The core philosophy is to prioritize **behavioral synchronization** over simple trend matching, ensuring that selected control markets react to external shocks in the same way as the target market.

## Objective
Build a Python-based pipeline that:
1.  Ingests time-series data for a Target city and multiple Control cities.
2.  Applies rigorous stationarity transformations to reveal true behavioral correlations.
3.  Selects control markets based on strict statistical criteria (Correlation, Variance Ratio, Granger Causality).
4.  Estimates the causal effect of an intervention using the `pycausalimpact` library.

---

## Workflow

### 1. Data Simulation
The framework operates on simulated sales data for 21 cities:
*   **Target:** `Barcelona` (Linear trend + Seasonality + Shared Noise + Intervention).
*   **Controls:** 20 cities with varying degrees of correlation (High/Med/Low) and noise properties.
*   **Key Feature:** A "Shared Noise" component is included to simulate synchronous behavioral "shocks" between the target and high-correlation controls.
*   **Output:** The data is expected in a long-format CSV file named `sales_data.csv`.

### 2. Ingestion & Stationarity Pipeline
The pipeline ingests the raw data and performs a "Rainy Tuesday" test to find the optimal stationarity transformation for each series.
*   **Transformations Tested:** Raw -> Log -> Difference -> Seasonal Difference.
*   **Goal:** To interpret the "dips and spikes" (shocks) rather than just the long-term trend.

### 3. Behavioral Selection Logic
Controls are selected using a **Tiered Strategy** to ensure structural integrity:

*   **Tier 1 (Strict):**
    *   **Transformed Correlation** > 0.8: Ensures strong behavioral alignment.
    *   **Variance Ratio** (0.5 - 2.0): Ensures the control isn't too volatile or too flat compared to the target.
*   **Tier 2 (Fallback):** used if no Tier 1 matches are found.
    *   Raw Correlation > 0.6 AND Transformed Correlation > 0.6.

**Expert Layer (Granger Causality):**
A final check is performed on stationary data to verify if the control series carries predictive intent (p < 0.05) for the target, effectively acting as a leading indicator.

### 4. Analysis & Results
*   **Selection Matrix:** A matrix displaying Raw/Transformed correlations, Granger p-values, and Variance Ratios for all candidates.
*   **Visualizations:** Side-by-side plots with status icons (✅ PASS, 🆗 ELIGIBLE, ⚠️ VOLATILE, 📉 TIER 2, ❌ FAILED).
*   **Causal Model:** The `pycausalimpact` model is run using the **Raw Data** of the selected best controls to estimate the intervention effect.

---

## Usage
1.  Ensure `sales_data.csv` is present in the project directory.
2.  Run the Jupyter Notebook `causal_impact.ipynb`.
3.  The notebook will:
    *   Load and preprocess the data.
    *   Run the selection pipeline.
    *   Display the selection results and diagnostics.
    *   Generate the Causal Impact analysis and plots.

## Constraints & Requirements
*   **Libraries:** `statsmodels` (for AdFuller, Granger), `pycausalimpact` (for Causal Impact model).
*   **Data Leakage:** All selection tests are strictly confined to the pre-intervention period.
*   **Philosophy:** Prioritize movement synchronization over simple trend alignment.
