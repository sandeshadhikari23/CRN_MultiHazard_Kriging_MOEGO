import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path

# ----------------------------
# Academic Plot Settings (Journal Standard)
# ----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["figure.titlesize"] = 12

# Professional Journal Color Palette
COLOR_BASE_NET = "#7F8C8D"      # Muted slate gray for background infrastructure
COLOR_REHAB = "#E67E22"         # Academic amber/orange for rehabilitated edges
COLOR_ALL_SOL = "#BDC3C7"       # Light gray for non-dominated background solutions
COLOR_PARETO = "#2C3E50"        # Deep Navy/Charcoal for Pareto optimal points

# ----------------------------
# Helper: convert DMS to decimal degrees
# ----------------------------
def dms_to_dd(dms_str):
    dms_str = str(dms_str).strip()
    match = re.match(r"(\d+)°(\d+)'([\d\.]+)\"?([NSEW])", dms_str)
    if not match:
        return None
    degrees, minutes, seconds, direction = match.groups()
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ["S", "W"]:
        dd *= -1
    return dd

# ----------------------------
# Proper Pareto Front Extraction
# ----------------------------
def pareto_front(df, obj1, obj2, minimize_obj1=True, minimize_obj2=True):
    data = df[[obj1, obj2]].values.copy()
    if not minimize_obj1:
        data[:, 0] = -data[:, 0]
    if not minimize_obj2:
        data[:, 1] = -data[:, 1]
    
    n_points = data.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            if np.all(data[j] <= data[i]) and np.any(data[j] < data[i]):
                is_pareto[i] = False
                break
    return df[is_pareto].copy()

# ----------------------------
# 1. Load Excel files
# ----------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
network_file = str(_REPO_ROOT / "Network_info" / "Edge_Info_Miles_DMS.xlsx")
moga_result = str(_REPO_ROOT / "Results" / "SanAndres+6in" / "Result_5%" / "ego_moga_results_20260822_162449.xlsx")

df_network = pd.read_excel(network_file)
df_all_fronts = pd.read_excel(moga_result, sheet_name="All_Fronts")

# ----------------------------
# 2. Clean invalid data
# ----------------------------
df_all_fronts = df_all_fronts[
    (df_all_fronts["Y_eq_original"] >= 0) & 
    (df_all_fronts["Y_fl_original"] >= 0)
].copy()
df_all_fronts.reset_index(drop=True, inplace=True)

# ----------------------------
# 3. Convert DMS coordinates
# ----------------------------
df_network["Start_X"] = df_network["Start Node X (DMS)"].apply(dms_to_dd)
df_network["Start_Y"] = df_network["Start Node Y (DMS)"].apply(dms_to_dd)
df_network["End_X"] = df_network["End Node X (DMS)"].apply(dms_to_dd)
df_network["End_Y"] = df_network["End Node Y (DMS)"].apply(dms_to_dd)

# ----------------------------
# 4. Compute Pareto Front
# ----------------------------
MIN_EQ = True
MIN_FL = True

df_pareto = pareto_front(
    df_all_fronts, obj1="Y_eq_original", obj2="Y_fl_original",
    minimize_obj1=MIN_EQ, minimize_obj2=MIN_FL
)
df_pareto.reset_index(drop=True, inplace=True)

# ----------------------------
# 5. Create output folder
# ----------------------------
output_folder = os.path.join(os.path.dirname(moga_result), "Pareto_Network_Plots")
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# 6. Plot each Pareto solution network
# ----------------------------
for i, row in df_pareto.iterrows():
    decision_vector = str(row["Decision_Vector"]).strip()
    decisions = [int(x) for x in decision_vector]

    # Academic single-column width layout (approx 3.5 inches / 9 cm width)
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Plot edges efficiently using line collections or simple iterations
    for j, edge in df_network.iterrows():
        is_rehab = (j < len(decisions) and decisions[j] == 1)
        color = COLOR_REHAB if is_rehab else COLOR_BASE_NET
        linewidth = 1.8 if is_rehab else 0.8
        zorder = 3 if is_rehab else 2
        
        ax.plot(
            [edge["Start_X"], edge["End_X"]],
            [edge["Start_Y"], edge["End_Y"]],
            color=color,
            linewidth=linewidth,
            alpha=0.9 if is_rehab else 0.4,
            zorder=zorder
        )

    ax.set_aspect("equal")
    ax.axis("off")
    
    # Clean, unobtrusive title/annotation structure
    ax.set_title(f"Rehabilitation Strategy Portfolio {i+1}", fontsize=10, fontweight="bold", pad=4)
    
    # Metadata string positioned professionally inside/below the frame bounds
    info_text = f"$ADC_{{EQ}}$: {row['Y_eq_original']:.2f}  |  $ADC_{{FL}}$: {row['Y_fl_original']:.2f}"
    plt.figtext(0.5, 0.02, info_text, wrap=True, ha="center", fontsize=9, fontweight="normal")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(output_folder, f"pareto_front_{i+1}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------
# 7. Enhanced Scatter Plot
# ----------------------------
# 1-column standard academic sizing (width=5.5 inches)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.set_facecolor("white")

# Plot Non-dominated Background Space
ax.scatter(
    df_all_fronts["Y_eq_original"],
    df_all_fronts["Y_fl_original"],
    color=COLOR_ALL_SOL,
    alpha=0.35,
    s=20,
    edgecolors="none",
    label="Feasible Objective Space",
    zorder=2
)

# Sort Pareto points to cleanly draw boundary
df_plot = df_pareto.sort_values(by="Y_eq_original")

# Draw crisp step function for discrete trade-offs (Standard for multi-objective problems)
ax.plot(
    df_plot["Y_eq_original"],
    df_plot["Y_fl_original"],
    color=COLOR_PARETO,
    linewidth=1.2,
    linestyle="-",
    drawstyle="steps-post", 
    alpha=0.7,
    zorder=3
)

# Plot Pareto Front Points
ax.scatter(
    df_pareto["Y_eq_original"],
    df_pareto["Y_fl_original"],
    color=COLOR_REHAB,
    edgecolor=COLOR_PARETO,
    linewidth=0.8,
    s=45,
    alpha=1.0,
    label="Identified Pareto Frontier",
    zorder=4
)

# Journal Label Standardizations (Using subscripts properly via LaTeX formatting)
ax.set_xlabel("Average Disruption Cost under Earthquake ($ADC_{EQ}$)", labelpad=6)
ax.set_ylabel("Average Disruption Cost under Flood ($ADC_{FL}$)", labelpad=6)

# Minimalistic, Clean Axis Frame
ax.grid(True, linestyle=":", color="#BDC3C7", alpha=0.6, zorder=1)

# Full closed frame
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(0.7)
    ax.spines[spine].set_color("#333333")

# Academic Legend formatting
ax.legend(
    frameon=True, 
    facecolor="#FFFFFF", 
    edgecolor="none", 
    loc="upper right", 
    shadow=False
)

plt.tight_layout()
scatter_path = os.path.join(output_folder, "Pareto_All_Fronts_Scatter.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------
# 8. Save Pareto solutions
# ----------------------------
pareto_excel = os.path.join(output_folder, "Pareto_Front_Solutions.xlsx")
df_pareto.to_excel(pareto_excel, index=False)

print("\n📊 Process Complete. Visualizations converted successfully to Academic Specifications.")