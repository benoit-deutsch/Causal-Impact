import json
import os

notebook_path = r'c:\Users\benoi\Desktop\AI Studio\Data Science\Causal Impact\candidate_selection.ipynb'

# Load the current state
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper functions for our senior selection
helper_funcs = [
    "def select_sparse_portfolio(df_pre, target):\n",
    "    \"\"\"TRICK 1: LASSO selection for a sparse 'optimal' control group.\"\"\"\n",
    "    from sklearn.linear_model import LassoCV\n",
    "    X = df_pre.drop(columns=[target]).fillna(0)\n",
    "    y = df_pre[target].fillna(0)\n",
    "    lasso = LassoCV(cv=5, random_state=42).fit(X, y)\n",
    "    import pandas as pd\n",
    "    weights = pd.Series(lasso.coef_, index=X.columns)\n",
    "    return weights[weights != 0].sort_values(ascending=False)\n",
    "\n",
    "def calculate_dtw_distance(s1, s2):\n",
    "    \"\"\"TRICK 3: Dynamic Time Warping for shape-based matching.\"\"\"\n",
    "    from fastdtw import fastdtw\n",
    "    distance, path = fastdtw(s1.fillna(0).values, s2.fillna(0).values, dist=lambda a, b: abs(a - b))\n",
    "    return distance\n",
    "\n",
    "def get_mutual_info(s1, s2):\n",
    "    \"\"\"TRICK 4: Non-linear dependency detection.\"\"\"\n",
    "    from sklearn.feature_selection import mutual_info_regression\n",
    "    mi = mutual_info_regression(s1.values.reshape(-1, 1), s2)\n",
    "    return mi[0]\n",
    "\n",
    "def get_stationary_transform(series, seasonal_period=7):\n",
    "    \"\"\"Returns (step_name, transform_func) or (None, None).\"\"\"\n",
    "    from statsmodels.tsa.stattools import adfuller\n",
    "    def is_stationary(s):\n",
    "        try: return adfuller(s.dropna())[1] < 0.05\n",
    "        except: return False\n",
    "    if is_stationary(series): return \"Raw\", lambda s: s\n",
    "    try:\n",
    "        s_log = np.log(series)\n",
    "        if is_stationary(s_log): return \"Log\", lambda s: np.log(s)\n",
    "        s_diff = s_log.diff()\n",
    "        if is_stationary(s_diff): return \"Log+Diff\", lambda s: np.log(s).diff()\n",
    "        s_seasonal = s_diff.diff(seasonal_period)\n",
    "        if is_stationary(s_seasonal): return \"Log+Diff+Seasonal\", lambda s: np.log(s).diff().diff(seasonal_period)\n",
    "    except: pass\n",
    "    return None, None\n"
]

senior_selection_func = [
    "def select_best_controls(df_full, target, pre_beg, pre_end, t1_thresh=0.8, t2_thresh=0.6):\n",
    "    \"\"\"SENIOR SELECTION: Multi-metric ranking using Correlation, LASSO, DTW, and Mutual Information.\"\"\"\n",
    "    import pandas as pd\n",
    "    df_pre = df_full.loc[pre_beg:pre_end]\n",
    "    potential_controls = [c for c in df_full.columns if c != target]\n",
    "    \n",
    "    print(f\"--- Senior Selection for {target} ---\")\n",
    "    \n",
    "    lasso_weights = select_sparse_portfolio(df_pre, target)\n",
    "    \n",
    "    results = []\n",
    "    for city in potential_controls:\n",
    "        corr_raw = df_pre[target].corr(df_pre[city])\n",
    "        step_name, transform_func = get_stationary_transform(df_pre[city])\n",
    "        \n",
    "        corr_trans = 0\n",
    "        var_ratio = 0\n",
    "        if step_name:\n",
    "            s_city_trans = transform_func(df_pre[city]).dropna()\n",
    "            s_target_trans = transform_func(df_pre[target]).dropna()\n",
    "            joined = pd.concat([s_city_trans, s_target_trans], axis=1).dropna()\n",
    "            corr_trans = joined.iloc[:, 0].corr(joined.iloc[:, 1])\n",
    "            var_ratio = s_city_trans.std() / s_target_trans.std() if s_target_trans.std() != 0 else 0\n",
    "        \n",
    "        dtw_dist = calculate_dtw_distance(df_pre[target], df_pre[city])\n",
    "        mi_score = get_mutual_info(df_pre[city], df_pre[target])\n",
    "        lasso_w = lasso_weights.get(city, 0.0)\n",
    "        \n",
    "        tier = \"None\"\n",
    "        if corr_trans > t1_thresh and 0.5 < var_ratio < 2.0:\n",
    "            if lasso_w > 0 or mi_score > 0.6:\n",
    "                tier = \"Tier 1 (Elite)\"\n",
    "            else:\n",
    "                tier = \"Tier 2 (Robust)\"\n",
    "        elif corr_raw > t2_thresh:\n",
    "            tier = \"Tier 3 (Baseline)\"\n",
    "            \n",
    "        results.append({\n",
    "            'City': city,\n",
    "            'Corr_Transformed': corr_trans,\n",
    "            'Variance_Ratio': var_ratio,\n",
    "            'DTW_Distance': dtw_dist,\n",
    "            'Mutual_Info': mi_score,\n",
    "            'LASSO_Weight': lasso_w,\n",
    "            'Selection_Tier': tier\n",
    "        })\n",
    "        \n",
    "    matrix = pd.DataFrame(results).sort_values(['Selection_Tier', 'DTW_Distance'])\n",
    "    selected = matrix[matrix['Selection_Tier'].isin([\"Tier 1 (Elite)\", \"Tier 2 (Robust)\"])]['City'].tolist()\n",
    "    \n",
    "    if not selected:\n",
    "        selected = matrix[matrix['Selection_Tier'] == \"Tier 3 (Baseline)\"]['City'].tolist()\n",
    "        status = \"Falling back to Baseline candidates.\"\n",
    "    else:\n",
    "        status = f\"Successfully selected {len(selected)} high-purity candidates.\"\n",
    "        \n",
    "    return matrix, selected, status"
]

power_funcs = [
    "def run_power_simulation(df_full, target, controls, lift_percent, pre_period, post_period, num_sims=10):\n",
    "    \"\"\"Simulates a synthetic lift and returns detection power.\"\"\"\n",
    "    data_base = df_full[[target] + controls].fillna(method='ffill').fillna(0)\n",
    "    hits = 0\n",
    "    for _ in range(num_sims):\n",
    "        sim_data = data_base.copy()\n",
    "        pre_mean = data_base.loc[pre_period[0]:pre_period[1], target].mean()\n",
    "        lift_amount = pre_mean * (lift_percent / 100)\n",
    "        sim_data.loc[post_period[0]:post_period[1], target] += lift_amount\n",
    "        try:\n",
    "            ci = CausalImpact(sim_data, pre_period, post_period)\n",
    "            if ci.p_value < 0.05: hits += 1\n",
    "        except: pass\n",
    "    return hits / num_sims\n",
    "\n",
    "def get_volume_requirements(df_full, target, controls, pre_period, post_period, lift_range=[2, 5, 10, 15, 20]):\n",
    "    results = []\n",
    "    for lift in lift_range:\n",
    "        power = run_power_simulation(df_full, target, controls, lift, pre_period, post_period)\n",
    "        results.append({'Lift_%': lift, 'Power': power})\n",
    "    return pd.DataFrame(results)"
]

# Clean up notebook
new_cells = []
for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    c_id = str(cell.get('id', ''))
    
    # Remove any existing selection or power cells
    if any(k in src for k in ['def select_best_controls', 'def run_power_simulation', 'def select_sparse_portfolio', 'def calculate_dtw_distance', 'get_volume_requirements']):
        continue
    if any(k in c_id for k in ['senior', 'power', 'integrated', 'advanced']):
        continue
        
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
    new_cells.append(cell)

# Re-insert Senior Selection block at the right place (index 1 is usually imports)
selection_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "senior_selection_core",
    "metadata": {},
    "outputs": [],
    "source": helper_funcs + ["\n"] + senior_selection_func
}
new_cells.insert(2, selection_cell)

# Re-insert Power Analysis section at the end
new_cells.append({"cell_type": "markdown", "id": "mde_header", "metadata": {}, "source": ["## experiment Design: Power & Volume Analysis"]})
new_cells.append({"cell_type": "code", "execution_count": None, "id": "power_funcs_core", "metadata": {}, "outputs": [], "source": power_funcs})
new_cells.append({"cell_type": "code", "execution_count": None, "id": "power_demo_cell_final", "metadata": {}, "outputs": [], "source": [
    "target_city = 'City_1'\n",
    "matrix, best_selection, status = select_best_controls(df, target_city, pre_beg, pre_end)\n",
    "print(status)\n",
    "display(matrix[matrix['Selection_Tier'] != 'None'].head(10))\n",
    "if best_selection:\n",
    "    power_df = get_volume_requirements(df, target_city, best_selection[:3], pre_period, post_period)\n",
    "    print(power_df)\n"
]})

nb['cells'] = new_cells
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Robust integration of Senior Selection criteria complete.')
