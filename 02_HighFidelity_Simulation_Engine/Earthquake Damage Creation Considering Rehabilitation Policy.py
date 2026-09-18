import pandas as pd
import numpy as np
import os
import shutil, os, stat
import time
from pathlib import Path
from scipy.stats import norm

# === SETTINGS ===
# Repo-relative paths (standalone-repo convention). RandomPolicies/ holds the input policy
# files for this script; PolicyScenarios/ is created here to hold its per-policy Monte Carlo
# damage output.
_HERE = Path(__file__).resolve().parent
pgv_file = str(_HERE / "Edge_with_Avg_PGV.xlsx")
policy_folder = str(_HERE / "RandomPolicies")
output_root = str(_HERE / "PolicyScenarios")

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

# === PARAMETERS ===
phi = 0.45  # intraevent std
tau = 0.40  # interevent std

rho1, gamma1 = 3.178, 0.7   # minor
rho2, gamma2 = 3.689, 0.7   # moderate
rho3, gamma3 = 4.994, 0.7   # extensive

num_realizations = 20  # PGV field realizations (Adhikari et al., manuscript: "20 random PGV fields")
num_mc = 130  # Monte Carlo damage-state draws per PGV field realization (manuscript: "N = 130" MCRs, from the convergence study in Figure 4)

# === FUNCTIONS ===
def fragility(PGV, rho, gamma):
    return norm.cdf((np.log(PGV) - rho) / gamma)

def assign_damage_state(PGV, selected):
    if selected == 1:
        return 1.0, 1.0, 1.0, -1, "intact (policy)", 0
    f1 = fragility(PGV, rho1, gamma1)
    f2 = fragility(PGV, rho2, gamma2)
    f3 = fragility(PGV, rho3, gamma3)
    U = np.random.uniform()
    if U >= f1:
        cond = "intact"
        dmg = 0
    elif U >= f2:
        cond = "minor"
        dmg = 0.25
    elif U >= f3:
        cond = "moderate"
        dmg = 0.50
    else:
        cond = "extensive"
        dmg = 1
    return f1, f2, f3, U, cond, dmg

# === MAIN LOOP ===
df_pgv = pd.read_excel(pgv_file)[['Edge ID', 'Edge Avg PGV']]
policy_files = [f for f in os.listdir(policy_folder) if f.endswith('.xlsx')]

for policy_file in policy_files:
    df_policy = pd.read_excel(os.path.join(policy_folder, policy_file))[['Edge ID', 'Selected']]
    df = pd.merge(df_pgv, df_policy, on='Edge ID', how='left').fillna({'Selected': 0})
    edge_ids = df['Edge ID'].values
    pgv_mean = df['Edge Avg PGV'].values
    selected = df['Selected'].values
    policy_name = os.path.splitext(policy_file)[0]
    policy_dir = os.path.join(output_root, policy_name)
    os.makedirs(policy_dir, exist_ok=True)

    for r in range(1, num_realizations + 1):
        eta = np.random.normal(0, tau)
        epsilon = np.random.normal(0, phi, size=len(df))
        ln_pgv_sim = np.log(pgv_mean) + epsilon + eta
        pgv_sim = np.exp(ln_pgv_sim)
        realization_dir = os.path.join(policy_dir, f"Realization_{r:02d}")
        os.makedirs(realization_dir, exist_ok=True)

        for mc in range(1, num_mc + 1):
            data = []
            for eid, mean_pgv, sel, pgv in zip(edge_ids, pgv_mean, selected, pgv_sim):
                f1, f2, f3, U, cond, dmg = assign_damage_state(pgv, sel)
                data.append([eid, mean_pgv, pgv, sel, f1, f2, f3, U, cond, dmg])
            out_df = pd.DataFrame(data, columns=[
                'Edge ID', 'Edge Avg PGV', 'PGV_sim', 'Selected',
                'f1_minor', 'f2_moderate', 'f3_extensive', 'U', 'Condition', 'Damage State'
            ])
            out_file = os.path.join(realization_dir, f"damage_{mc:03d}.xlsx")
            out_df.to_excel(out_file, index=False)
        print(f"{policy_name}: Finished Realization {r:02d} with {num_mc} MC runs.")

    print(f"{policy_name}: All {num_realizations} realizations with MC runs completed!")

print("All policy-based earthquake scenarios are ready!")
