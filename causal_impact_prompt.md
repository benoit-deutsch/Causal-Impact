# Prompt to Recreate the Advanced Causal Impact Framework

**Role:** You are a Senior Data Scientist specializing in Causal Inference and Time Series Econometrics.

**Objective:** Build a robust Python-based Causal Impact framework that automatically sources and filters the best control series from a library of candidates using behavioral synchronization instead of just trend matching.

---

## Step 1: Data Simulation (`generate_data.py`)
Create a script to simulate sales data for 21 cities (1 Target 'Barcelona', 20 Controls).
- **Target (Barcelona):** Linear trend + Seasonality + Shared Noise + Intervention (+20 units at day 150).
- **Controls:** Randomly assigned traits (High/Med/Low correlation).
- **CRITICAL:** Include a **Shared Noise** component that Barcelona and high-corr cities share. This represents synchronous behavioral "shocks."
- **Educational Edge:** Include a `City_Spurious` that matches Barcelona's trend/seasonality perfectly but has *independent* noise.
- **Output:** SAVE in **LONG format** CSV (`sales_data.csv`) with columns: `Date`, `City`, `Value`.

## Step 2: Ingestion & Stationarity Pipeline
In `causal_impact_demo.ipynb`:
1. **Load & Pivot:** Ingest the long-format CSV and pivot it to a wide matrix.
2. **Stationarity Pipeline:** Implement a function `get_stationary_transform` that tries (Raw -> Log -> Diff -> Seasonal Diff) on the **pre-intervention period only**.
3. **Return Function:** The pipeline must return both the name of the successful step and a `transform_func` (lambda) that can be applied to other series.

## Step 3: Behavioral Selection Logic
Implement `select_best_controls(df, target, pre_beg, pre_end)` using a **Tiered Strategy**:
- **The "Rainy Tuesday" Test:** For each city, find the transformation that makes it stationary. Apply that **exact same transformation** to both the City and Barcelona.
- **Tiered Selection Strategy:**
    - **Tier 1 (Strict):** Select cities where **Transformed Correlation > 0.8** AND **Variance Ratio (0.5 to 2.0)**.
    - **Tier 2 (Fallback):** If Tier 1 is empty, select cities where **both Raw Correlation > 0.6 AND Transformed Correlation > 0.6**.
- **Structural Integrity:** Always calculate the Transformed Correlation after applying the same stationarity-inducing transformation to both control and target.
- **Granger Causality (Expert Layer):** Perform a `grangercausalitytests` on the stationary data to verify predictive intent (p < 0.05).

## Step 4: Analysis & Final Polish
- **Selection Matrix:** Display a dataframe with Raw vs. Transformed correlations, Granger p-values, Variance Ratios, and selection status.
- **Visualization:** Create a side-by-side bar plot with clear **Status Icons**:
    - ✅ **PASS (All Tests)**
    - 🆗 **ELIGIBLE (No Granger)**
    - ⚠️ **VOLATILE (Failed Var)**
    - 📉 **TIER 2 (Weak Match)**
    - ❌ **FAILED**
- **Causal Model:** Run `pycausalimpact` using the **Raw Data** for selected controls.
- **Documentation:** Add a detailed markdown introduction explaining the rationale behind each test:
    - **Stationarity:** The "Rainy Tuesday" test (Behavioral Shocks).
    - **Granger Causality:** The "Leading Indicator" test (Predictive Information).
    - **Variance Ratio:** The "Store Scaling" test (Volatility Matching).

---

**Constraints:**
- Use `statsmodels.tsa.stattools.adfuller` for tests.
- Use `google/causalimpact` (`pycausalimpact`) for the model.
- No data leakage: All selection tests MUST be confined to `[pre_beg : pre_end]`.
- Always prioritize movement synchronization over simple trend alignment.
