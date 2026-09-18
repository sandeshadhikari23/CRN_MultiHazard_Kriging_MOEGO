"""
main.py

Top-level orchestration with log(Y+1) transformation and one PLS component.
- Loads Combined_EQ.xlsx and Combined_F.xlsx (expected columns: x0,x1,...,xd-1 and 'Y')
- Loads Edge_length.xlsx (first column used)
- Runs EGO-MOGA iterations (configurable number)
- Calls simulation oracle: either 'external' or 'mock'
- Appends evaluated infill results to Combined.xlsx
"""
import os
import sys
from pathlib import Path
import time
import uuid
import re
import subprocess
import numpy as np
import pandas as pd

from kpls_moga_ego import ego_moga_iteration, nondominated_sort

# === CONFIG ===
# Repo-relative paths (standalone-repo convention). Combined_EQ.xlsx / Combined_F.xlsx
# (the ~1000-LHS-policy training dataset, ~12-23MB) belong at the repo root -- see the
# top-level README's "Required data" section for their schema. Place your own copies
# at the repo root (next to this file's parent's parent) before running.
BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
OUTPUT_DIR = Path(__file__).resolve().parent  # this folder -- all exports go here
EQ_PATH = BASE_DIR / "Combined_EQ.xlsx"
FL_PATH = BASE_DIR / "Combined_F.xlsx"
LENGTH_PATH = BASE_DIR / "Edge_length.xlsx"  # kept for load_edge_lengths() / notebook reuse; not used by main()
COST_INFO_PATH = OUTPUT_DIR / "Network_info" / "Edge_Info_Miles_DMS_Updated.xlsx"
RESULTS_PATH = OUTPUT_DIR / "Combined_Result.xlsx"
SIMULATION_ORACLE_DIR = BASE_DIR / "02_HighFidelity_Simulation_Engine"
SIMULATION_MODE = "external"  # "external" or "mock"
MAX_ITER = 5
REHAB_PERCENT = np.float64(0.05)  # 5% of total network rehab cost
FAIL_TOLERANCE_N = 2
TRUNCATION_PERCENTILE = np.float64(92.5)  # one-sided upper-tail truncation, per diagnostic

# Unit conversions for the cost model
METERS_TO_FEET = np.float64(3.280839895)
MILES_TO_FEET = np.float64(5280.0)
CUFT_PER_CY = np.float64(27.0)

# Rehab unit cost rates
BRIDGE_RATE_PER_SQFT = np.float64(465.0)          # $/sq.ft of deck area (width x length) -- 2024
PAVEMENT_RATE_PER_CY = np.float64(882.00)          # $/CY, Joint Plain Concrete Pavement (RSC)
PAVEMENT_DEPTH_FT = np.float64(9.0) / np.float64(12.0)  # 9" slab depth
CULVERT_RATE_PER_FT_WIDTH = np.float64(352.69)     # $ per ft of road width, 24" Alternative Pipe Culvert

np.random.seed(0)

# === Utilities ===
def load_edge_costs(path):
    """
    Per-edge rehabilitation cost ($), built from Edge_Info_Miles_DMS_Updated.xlsx.

    For each edge:
      - Has_Bridge == 'Y': bridge deck cost = BRIDGE_RATE_PER_SQFT x (width_ft x length_ft)
      - Has_Culvert == 'Y': culvert cost = CULVERT_RATE_PER_FT_WIDTH x width_ft
        (a per-crossing cost -- the 24" pipe spans the road width, independent of edge length)
      - Neither bridge nor culvert: pavement cost = PAVEMENT_RATE_PER_CY x
        (width_ft x length_ft x PAVEMENT_DEPTH_FT) / CUFT_PER_CY
      - Both bridge AND culvert (both structures physically present): bridge cost + culvert cost
    'Total width' in the source file is in meters (SUMO lane widths); 'Length (miles)' is in miles.
    """
    df = pd.read_excel(path)
    required = {'Length (miles)', 'Total width', 'Has_Bridge', 'Has_Culvert'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Edge cost info file {path} missing columns: {missing}")

    length_ft = df['Length (miles)'].values.astype(np.float64) * MILES_TO_FEET
    width_ft = df['Total width'].values.astype(np.float64) * METERS_TO_FEET
    is_bridge = (df['Has_Bridge'].astype(str).str.strip().str.upper() == 'Y').values
    is_culvert = (df['Has_Culvert'].astype(str).str.strip().str.upper() == 'Y').values
    is_pavement = ~is_bridge & ~is_culvert

    bridge_cost = np.where(is_bridge, BRIDGE_RATE_PER_SQFT * width_ft * length_ft, 0.0)
    culvert_cost = np.where(is_culvert, CULVERT_RATE_PER_FT_WIDTH * width_ft, 0.0)
    pavement_volume_cy = (width_ft * length_ft * PAVEMENT_DEPTH_FT) / CUFT_PER_CY
    pavement_cost = np.where(is_pavement, PAVEMENT_RATE_PER_CY * pavement_volume_cy, 0.0)

    return (bridge_cost + culvert_cost + pavement_cost).astype(np.float64)

def truncate_mask(Y, percentile):
    """
    Boolean mask keeping only points where Y <= the given percentile of Y
    (one-sided, upper-tail truncation). Returns (mask, threshold).
    """
    thresh = np.percentile(Y, percentile)
    mask = Y <= thresh
    return mask, thresh

def load_design_and_y(path):
    df = pd.read_excel(path)

    # detect all columns starting with x or Selected_
    X_cols = [c for c in df.columns if c.lower().startswith('x') or c.startswith('Selected_')]
    if len(X_cols) == 0:
        X_cols = [c for c in df.columns if c.startswith('Selected_')]
    if len(X_cols) == 0:
        raise RuntimeError(f"No binary design columns found in {path}")

    X = df[X_cols].values.astype(np.int64)

    # find Y column
    if 'Y' in df.columns:
        Y = df['Y'].values.astype(np.float64)
    else:
        alt = [c for c in df.columns if 'loss' in c.lower()]
        if len(alt) == 0:
            raise RuntimeError(f"No 'Y' column (or alternative) found in {path}")
        Y = df[alt[0]].values.astype(np.float64)

    # remove rows where Y == 0
    mask = Y != 0
    X = X[mask]
    Y = Y[mask]

    # Apply log(Y+1) transformation
    Y = np.log(Y + 1.0)

    return X, Y

def load_edge_lengths(path):
    df = pd.read_excel(path)
    if 'Length (miles)' not in df.columns:
        raise ValueError("Expected column 'Length (miles)' in edge length Excel file.")
    lengths = df['Length (miles)'].values.astype(np.float64)
    return lengths

# === Simulation oracle ===
def simulation_oracle_external(x_infill):
    """
    Runs both Earthquake and Flood simulation pipelines.
    Returns original losses (not yet transformed).
    """
    sim_dir = Path(SIMULATION_ORACLE_DIR)

    eq_script1 = sim_dir / 'Earthquake Damage Creation Considering Rehabilitation Policy.py'
    eq_script2 = sim_dir / 'Earthquake Hazard Modeling Damaged Network with Rehabilitation Policy and Additional Punishment.py'
    fl_script1 = sim_dir / 'Flood Damage Scenario Creation Considering Rehabilitation Policy.py'
    fl_script2 = sim_dir / 'Flood Hazrad Modeling Damaged Network with Rehabilitation Policy and Additional Punishment.py'

    for s in [eq_script1, eq_script2, fl_script1, fl_script2]:
        if not s.exists():
            raise FileNotFoundError(f"Missing simulation script: {s}")

    python_exec = sys.executable

    def _extract_var(text, varname):
        m = re.search(rf"{varname}\s*=\s*r?[\'\"](.+?)[\'\"]", text)
        return m.group(1) if m else None

    def _run_pipeline(script1, script2, label):
        script1_text = script1.read_text(encoding='utf-8')
        policy_folder = _extract_var(script1_text, 'policy_folder')
        input_file = _extract_var(script1_text, 'input_path') or _extract_var(script1_text, 'pgv_file')
        output_root = _extract_var(script1_text, 'output_root')

        if policy_folder is None or input_file is None or output_root is None:
            raise RuntimeError(f"[{label}] Missing required paths in script1 ({script1.name})")

        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"[{label}] Input file not found: {input_file}")

        df_input = pd.read_excel(input_file)
        if 'Edge ID' not in df_input.columns:
            raise RuntimeError(f"[{label}] Input file missing 'Edge ID' column")

        edge_ids = df_input['Edge ID'].values
        x_arr = np.asarray(x_infill).astype(np.int64).flatten()
        if len(edge_ids) != len(x_arr):
            raise ValueError(f"[{label}] Length mismatch: candidate length {len(x_arr)} != edges {len(edge_ids)}")

        os.makedirs(policy_folder, exist_ok=True)
        policy_name = f'policy_temp_{label}_{int(time.time())}_{uuid.uuid4().hex[:6]}'
        policy_file = Path(policy_folder) / f"{policy_name}.xlsx"
        df_policy = pd.DataFrame({'Edge ID': edge_ids, 'Selected': x_arr})
        df_policy.to_excel(policy_file, index=False)

        def _run_and_stream(cmd, cwd, tag):
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, text=True)
            try:
                for line in proc.stdout:
                    if line:
                        print(f"[{tag}] {line.strip()}")
            finally:
                proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"{tag} failed with code {proc.returncode}")

        _run_and_stream([python_exec, "-u", str(script1)], str(sim_dir), f"{label}_script1")
        _run_and_stream([python_exec, "-u", str(script2)], str(sim_dir), f"{label}_script2")

        policy_dir = Path(output_root) / policy_name
        summary_file = policy_dir / 'scenario_summary.csv'
        if not summary_file.exists():
            for root, _, files in os.walk(output_root):
                for f in files:
                    if f == 'scenario_summary.csv' and policy_name in root:
                        summary_file = Path(root) / f
                        break

        if not summary_file.exists():
            raise RuntimeError(f"[{label}] scenario_summary.csv not found for {policy_name}")

        df_summary = pd.read_csv(summary_file)
        if not {'Serviceability Loss ($)', 'Additional Loss ($)'}.issubset(df_summary.columns):
            raise RuntimeError(f"[{label}] scenario_summary.csv missing required loss columns")

        total_loss = float((df_summary['Serviceability Loss ($)'] + df_summary['Additional Loss ($)']).mean())

        try:
            policy_file.unlink()
        except Exception:
            pass

        return total_loss

    eq_loss = _run_pipeline(eq_script1, eq_script2, "EQ")
    fl_loss = _run_pipeline(fl_script1, fl_script2, "FL")

    return eq_loss, fl_loss

def simulation_oracle_mock(x_infill):
    """
    Mock oracle for testing: returns reproducible pseudo-random losses.
    """
    x = np.asarray(x_infill).astype(np.int64).flatten()
    seed = int(np.sum(x.astype(np.int64) * (np.arange(x.size, dtype=np.int64) + 1)))
    rng = np.random.RandomState(seed)
    eq_loss = float(rng.rand() * 1000.0)
    fl_loss = float(rng.rand() * 1000.0)
    return eq_loss, fl_loss

# === Main EGO loop ===
def main():
    # Load datasets
    if not EQ_PATH.exists():
        raise FileNotFoundError(f"{EQ_PATH} not found.")
    if not FL_PATH.exists():
        raise FileNotFoundError(f"{FL_PATH} not found.")
    if not COST_INFO_PATH.exists():
        raise FileNotFoundError(f"{COST_INFO_PATH} not found.")

    df_eq = pd.read_excel(EQ_PATH)
    df_fl = pd.read_excel(FL_PATH)

    X_cols_eq = [c for c in df_eq.columns if c.lower().startswith('x') or c.startswith('Selected_')]
    X_cols_fl = [c for c in df_fl.columns if c.lower().startswith('x') or c.startswith('Selected_')]

    Y_col_eq = 'Y' if 'Y' in df_eq.columns else [c for c in df_eq.columns if 'loss' in c.lower()][0]
    Y_col_fl = 'Y' if 'Y' in df_fl.columns else [c for c in df_fl.columns if 'loss' in c.lower()][0]

    if not np.allclose(df_eq[X_cols_eq].values, df_fl[X_cols_fl].values):
        raise RuntimeError("Designs in EQ and FL files are not aligned. Check Combined_EQ.xlsx and Combined_F.xlsx.")

    mask_valid = (df_eq[Y_col_eq] != 0) & (df_fl[Y_col_fl] != 0)

    X_obs_eq = df_eq.loc[mask_valid, X_cols_eq].values.astype(np.int64)
    Y_obs_eq_orig = df_eq.loc[mask_valid, Y_col_eq].values.astype(np.float64)
    X_obs_fl = df_fl.loc[mask_valid, X_cols_fl].values.astype(np.int64)
    Y_obs_fl_orig = df_fl.loc[mask_valid, Y_col_fl].values.astype(np.float64)

    print(f"Removed {(~mask_valid).sum()} rows where Y=0 in either EQ or FL dataset.")

    # Truncate at the 92.5th percentile (one-sided, upper-tail only) per hazard,
    # then intersect the two masks so X_obs_eq/X_obs_fl stay row-aligned (the
    # rest of this file assumes the same design vector at the same index in
    # both arrays).
    mask_trunc_eq, thresh_eq = truncate_mask(Y_obs_eq_orig, TRUNCATION_PERCENTILE)
    mask_trunc_fl, thresh_fl = truncate_mask(Y_obs_fl_orig, TRUNCATION_PERCENTILE)
    mask_trunc = mask_trunc_eq & mask_trunc_fl

    X_obs_eq = X_obs_eq[mask_trunc]
    Y_obs_eq_orig = Y_obs_eq_orig[mask_trunc]
    X_obs_fl = X_obs_fl[mask_trunc]
    Y_obs_fl_orig = Y_obs_fl_orig[mask_trunc]

    print(f"Truncated at {TRUNCATION_PERCENTILE}th percentile "
          f"(EQ threshold=${thresh_eq:,.2f}, FL threshold=${thresh_fl:,.2f}): "
          f"removed {int((~mask_trunc).sum())} rows, {int(mask_trunc.sum())} remaining.")

    # Apply log(Y+1) transformation to the truncated data
    Y_obs_eq = np.log(Y_obs_eq_orig + 1.0)
    Y_obs_fl = np.log(Y_obs_fl_orig + 1.0)

    costs = load_edge_costs(COST_INFO_PATH)
    total_cost = float(np.sum(costs))
    rehab_budget = np.float64(REHAB_PERCENT) * np.float64(total_cost)
    print(f"Rehab budget: {float(REHAB_PERCENT*100):.2f}% of total network rehab cost "
          f"(${total_cost:,.2f}) -> ${rehab_budget:,.2f}")

    X_eq = X_obs_eq.copy().astype(np.int64)
    Y_eq = Y_obs_eq.copy().astype(np.float64)
    X_fl = X_obs_fl.copy().astype(np.float64)
    Y_fl = Y_obs_fl.copy().astype(np.float64)

    if SIMULATION_MODE == "external":
        sim_oracle = simulation_oracle_external
        print("Using external simulation oracle (will attempt to run scripts).")
    else:
        sim_oracle = simulation_oracle_mock
        print("Using mock simulation oracle (random deterministic outputs).")

    appended_results = []
    consecutive_failures = 0

    initial_rehab_costs = (X_eq.astype(np.int64) @ costs.astype(np.float64)).astype(np.float64)
    feasible_mask = initial_rehab_costs <= float(rehab_budget)

    if np.any(feasible_mask):
        initial_objs = np.vstack([-Y_eq[feasible_mask], -Y_fl[feasible_mask]]).T
        initial_fronts = nondominated_sort(initial_objs)
        historical_pareto_size = len(initial_fronts[0])
    else:
        historical_pareto_size = 0

    print(f"Initial historical feasible Pareto front size (Y_eq, Y_fl): {historical_pareto_size}")

    for iteration in range(int(MAX_ITER)):
        print(f"\n=== EGO Iteration {iteration+1}/{MAX_ITER} ===")

        selected = ego_moga_iteration(
            X_eq, Y_eq, X_fl, Y_fl,
            costs, rehab_budget,
            simulation_oracle_func=sim_oracle,
            n_neighbors_single=50, n_neighbors_double=50,
            moga_pop=200, moga_gens=100,
            n_pls_components=1   # use one PLS component for both
        )

        new_policies_added_this_iter = 0

        keys = ['ei_eq_best', 'ei_fl_best', 'midpoint']
        for key in keys:
            x_bin, ei_eq, ei_fl = selected[key]

            tup = tuple(map(int, x_bin.tolist()))
            already_eq = any((X_eq.astype(np.int64) == x_bin).all(axis=1))
            already_fl = any((X_fl.astype(np.int64) == x_bin).all(axis=1))
            if already_eq and already_fl:
                print(f"Candidate {tup} already observed in both datasets; skipping simulation.")
                continue

            current_cost = float(np.dot(x_bin.astype(np.float64), costs.astype(np.float64)))
            if current_cost > float(rehab_budget):
                print(f"Candidate {key} violates budget; skipping simulation.")
                continue

            try:
                eq_loss_orig, fl_loss_orig = sim_oracle(x_bin)
            except Exception as e:
                print(f"Simulation oracle failed for candidate {tup}: {e}")
                eq_loss_orig, fl_loss_orig = simulation_oracle_mock(x_bin)
                print(f"Falling back to mock results: EQ={eq_loss_orig:.3f}, FL={fl_loss_orig:.3f}")

            # Transform for storage in surrogate models
            eq_loss = np.log(eq_loss_orig + 1.0)
            fl_loss = np.log(fl_loss_orig + 1.0)

            new_x = x_bin.reshape(1, -1).astype(np.int64)
            new_y_eq = np.array([eq_loss], dtype=np.float64)
            new_y_fl = np.array([fl_loss], dtype=np.float64)

            X_eq = np.vstack([X_eq, new_x])
            Y_eq = np.hstack([Y_eq, new_y_eq])
            X_fl = np.vstack([X_fl, new_x])
            Y_fl = np.hstack([Y_fl, new_y_fl])

            # Update Pareto front size (using transformed values)
            rehab_costs_current = (X_eq.astype(np.int64) @ costs.astype(np.float64)).astype(np.float64)
            feasible_mask_current = rehab_costs_current <= float(rehab_budget)
            if np.any(feasible_mask_current):
                current_objs = np.vstack([-Y_eq[feasible_mask_current], -Y_fl[feasible_mask_current]]).T
                current_fronts = nondominated_sort(current_objs)
                current_pareto_size = len(current_fronts[0])
            else:
                current_pareto_size = 0

            if current_pareto_size > historical_pareto_size:
                new_policies_added_this_iter += 1
                historical_pareto_size = current_pareto_size

            # Store original losses for human-readable output
            appended_results.append({
                'Iteration': iteration + 1,
                'Type': key,
                'Binary': ''.join(map(str, x_bin.tolist())),
                'EI_EQ': float(ei_eq),
                'EI_FL': float(ei_fl),
                'Eq_loss_orig': float(eq_loss_orig),
                'Fl_loss_orig': float(fl_loss_orig)
            })
            print(f"Appended EQ_loss_orig={eq_loss_orig:.3f} | FL_loss_orig={fl_loss_orig:.3f} | EI_EQ={ei_eq:.6e} | EI_FL={ei_fl:.6e}")

        if new_policies_added_this_iter == 0:
            consecutive_failures += 1
            print(f"⚠️ No new non-dominated policies added. Consecutive failures: {consecutive_failures}/{FAIL_TOLERANCE_N}")
        else:
            consecutive_failures = 0
            print(f"✅ {new_policies_added_this_iter} new non-dominated policies added. New size: {historical_pareto_size}")

        if consecutive_failures >= FAIL_TOLERANCE_N:
            print(f"\n🚨 **Early Stopping Triggered**: {FAIL_TOLERANCE_N} consecutive iterations failed to add a new non-dominated policy.")
            break

    df_appended = pd.DataFrame(appended_results)
    if RESULTS_PATH.exists():
        df_existing = pd.read_excel(RESULTS_PATH)
        df_out = pd.concat([df_existing, df_appended], ignore_index=True)
    else:
        df_out = df_appended
    df_out.to_excel(RESULTS_PATH, index=False)
    print(f"\nDone. Results written to {RESULTS_PATH}")

if __name__ == "__main__":
    main()