import os
from pathlib import Path
import numpy as np
import pandas as pd
import shutil, os, stat
import time

# Repo-relative paths (standalone-repo convention). RandomPolicies/ holds the input policy
# files for this script; Flood_MonteCarlo_withPolicy/ is created here to hold its per-policy
# Monte Carlo output.
_HERE = Path(__file__).resolve().parent
input_path = str(_HERE / "Flood Depth in inch with traffic stop mark - Cleaned.xlsx")
policy_folder = str(_HERE / "RandomPolicies")
output_root = str(_HERE / "Flood_MonteCarlo_withPolicy")
os.makedirs(output_root, exist_ok=True)


# === CLEAR OUTPUT ROOT FIRST ===
def remove_readonly(func, path, exc_info):
    """Clear read-only attribute and retry deletion."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

if os.path.exists(output_root):
    # Try up to 3 times in case the directory is in use
    for attempt in range(3):
        try:
            shutil.rmtree(output_root, onerror=remove_readonly)
            break
        except PermissionError as e:
            print(f"Attempt {attempt+1}: PermissionError while deleting {output_root} — {e}")
            time.sleep(1)
    else:
        raise PermissionError(f"Failed to delete {output_root} after multiple attempts.")

# Recreate the output directory
os.makedirs(output_root, exist_ok=True)

n_runs_per_policy = 130  # Monte Carlo runs per policy (aligned to the same N=130 convergence-study
# parameter as the earthquake side; manuscript: N=130 MCRs "through a convergence study of ADC
# under both earthquake and flood hazards" -- NOTE: this flood-side value of 130 is an inference
# by the repo curator, not an explicit user-confirmed number; the user only explicitly confirmed
# the earthquake-side 20 realizations x 130 MC. Double-check against Figure 4 / Convergence_Study
# before treating flood N=130 as final.
mean_rain = 1.58
conf_low, conf_high = 1.23, 2.04
std_rain = (conf_high - conf_low) / (2 * 1.645)

df_master = pd.read_excel(input_path)
depth_columns = df_master.columns[1:9].tolist()
hour_columns = ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3']
col_pairs = [
    (depth_columns[0], depth_columns[1]),  # Hour 0: B & C
    (depth_columns[2], depth_columns[3]),  # Hour 1: D & E
    (depth_columns[4], depth_columns[5]),  # Hour 2: F & G
    (depth_columns[6], depth_columns[7]),  # Hour 3: H & I
]

policy_files = [f for f in os.listdir(policy_folder) if f.endswith('.xlsx')]

for policy_file in policy_files:
    # Read policy file
    policy_path = os.path.join(policy_folder, policy_file)
    df_policy = pd.read_excel(policy_path)
    # Merge policy with flood depth on 'Edge ID'
    df_merge = pd.merge(df_master, df_policy[['Edge ID', 'Selected']], on='Edge ID', how='left').fillna({'Selected': 0})

    policy_name = os.path.splitext(policy_file)[0]
    policy_outdir = os.path.join(output_root, policy_name)
    os.makedirs(policy_outdir, exist_ok=True)

    for run in range(1, n_runs_per_policy + 1):
        df_run = df_merge.copy()
        rain = np.random.normal(mean_rain, std_rain)
        scaling_factor = rain / mean_rain
        for col in depth_columns:
            df_run[col] = df_run[col] * scaling_factor

        for j, hour in enumerate(hour_columns):
            col1, col2 = col_pairs[j]
            max_depth = df_run[[col1, col2]].max(axis=1)
            # Apply flood logic
            df_run[hour] = 1
            df_run[hour] = df_run[hour].astype(float)
            df_run.loc[(max_depth > 1) & (max_depth < 6), hour] = 0.5
            df_run.loc[max_depth >= 6, hour] = 0
            # **Override by Selected: force open if Selected==1**
            df_run.loc[df_run['Selected'] == 1, hour] = 1

        out_path = os.path.join(policy_outdir, f"{policy_name}_MC_{run}.xlsx")
        df_run.to_excel(out_path, index=False)
        print(f"{policy_name}: Saved run {run}/{n_runs_per_policy}")

print("Done! All policy Monte Carlo runs created.")
