"""
Compare_policies_Cost_MapEnhanced.py

Copy of Compare_policies_Cost.py with enhanced network maps: scale bars, north
arrows, a clearer basemap, and road-class information.

Everything up through the summary tables / bar / Venn / heatmap plots is unchanged from
Compare_policies_Cost.py (the old dead-code experiments at the top of that file were
dropped here since they don't affect behavior). What's new:

  1. Roads are bucketed into a rough functional class (Arterial/Highway, Collector, Local)
     from the posted speed limit, and the basemap is drawn with that class hierarchy
     (dark/thick for arterials down to light/thin for local roads) instead of one flat
     gray -- this gives the basemap clearer road-class information.
  2. Every network map gets a scale bar (built from a local miles-per-degree
     approximation -- no tile/CRS dependency needed) and a north arrow.
  3. A new composite 2x2 figure (Section 7) reproduces the (a)-(d) Earthquake/Flood/
     Balanced/Common panel layout from the base figure, now with the elements above
     plus a shared legend that also documents the road classes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_venn import venn3
import seaborn as sns
import re
import os

# Reuse the exact same per-edge rehab cost formula used by the optimizer
# (kpls_main.py, one folder up) instead of duplicating it here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kpls_main import load_edge_costs, COST_INFO_PATH

# ----------------------------
# GLOBAL PLOT SETTINGS
# ----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# ----------------------------
# HELPER: Convert DMS -> Decimal Degrees
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

# ============================================================
# 1️⃣ LOAD INPUT FILES
# ============================================================
policies_file = str(Path(__file__).resolve().parent.parent / "Results" / "SanAndres+6in" / "Result_5%" / "3-policies.xlsx")
network_file  = str(COST_INFO_PATH)  # Edge_Info_Miles_DMS_Updated.xlsx -- has the Total width column the cost formula needs

df = pd.read_excel(policies_file)
df_network = pd.read_excel(network_file)

# Per-edge rehab cost ($), row-aligned with df_network / the Decision_Vector bits
edge_costs = load_edge_costs(COST_INFO_PATH)
total_network_cost = float(edge_costs.sum())

# Original output folder
output_folder = os.path.join(os.path.dirname(policies_file), "Scenario_Comparison_Plots")
os.makedirs(output_folder, exist_ok=True)
print(f"📁 Output folder: {output_folder}")

# Output folder for the enhanced (scale bar / north arrow / road-class basemap) maps
enhanced_output_folder = os.path.join(os.path.dirname(policies_file), "Scenario_Comparison_Plots_EnhancedMaps")
os.makedirs(enhanced_output_folder, exist_ok=True)
print(f"📁 Enhanced-map plots folder: {enhanced_output_folder}")

# ============================================================
# 2️⃣ PROCESS DECISION VECTORS
# ============================================================
df['Decision_Vector'] = df['Decision_Vector'].apply(lambda x: [int(ch) for ch in str(x).strip()])

decision_df = pd.DataFrame({row.Rehab_Scenario: row.Decision_Vector for _, row in df.iterrows()})
total_sections = len(decision_df)

# Summary
summary = (decision_df.sum() / total_sections * 100).round(2)
summary_df = pd.DataFrame({'Scenario': summary.index, 'Critical (%)': summary.values})

# Overlap
all_critical = (decision_df.sum(axis=1) == len(decision_df.columns)).sum()
any_critical = (decision_df.sum(axis=1) >= 1).sum()
unique_critical = {
    col: ((decision_df[col] == 1) & (decision_df.drop(columns=[col]).sum(axis=1) == 0)).sum()
    for col in decision_df.columns
}

pairwise_overlap = pd.DataFrame(index=decision_df.columns, columns=decision_df.columns)
for s1 in decision_df.columns:
    for s2 in decision_df.columns:
        both = ((decision_df[s1] == 1) & (decision_df[s2] == 1)).sum()
        pairwise_overlap.loc[s1, s2] = round(both / total_sections * 100, 2)

# ============================================================
# 2️⃣B LENGTH-BASED CRITICALITY ANALYSIS
# ============================================================
length_col = "Length (miles)"
total_network_length = df_network[length_col].sum()

length_summary = {}
for scenario in decision_df.columns:
    critical_mask = decision_df[scenario] == 1
    critical_length = df_network.loc[critical_mask, length_col].sum()
    length_summary[scenario] = round(critical_length / total_network_length * 100, 2)

length_summary_df = pd.DataFrame({
    "Scenario": list(length_summary.keys()),
    "Critical_Length (%)": list(length_summary.values())
})

pairwise_overlap_length = pd.DataFrame(index=decision_df.columns, columns=decision_df.columns)
for s1 in decision_df.columns:
    for s2 in decision_df.columns:
        overlap_mask = (decision_df[s1] == 1) & (decision_df[s2] == 1)
        overlap_length = df_network.loc[overlap_mask, length_col].sum()
        pairwise_overlap_length.loc[s1, s2] = round(overlap_length / total_network_length * 100, 2)

print("\n📏 Length-based overlap analysis completed.")

# ============================================================
# 2️⃣C COST-BASED CRITICALITY ANALYSIS
# ============================================================
cost_summary = {}
for scenario in decision_df.columns:
    critical_mask = (decision_df[scenario] == 1).values
    critical_cost = edge_costs[critical_mask].sum()
    cost_summary[scenario] = round(critical_cost / total_network_cost * 100, 2)

cost_summary_df = pd.DataFrame({
    "Scenario": list(cost_summary.keys()),
    "Critical_Cost (%)": list(cost_summary.values())
})

pairwise_overlap_cost = pd.DataFrame(index=decision_df.columns, columns=decision_df.columns)
for s1 in decision_df.columns:
    for s2 in decision_df.columns:
        overlap_mask = ((decision_df[s1] == 1) & (decision_df[s2] == 1)).values
        overlap_cost = edge_costs[overlap_mask].sum()
        pairwise_overlap_cost.loc[s1, s2] = round(overlap_cost / total_network_cost * 100, 2)

print("💲 Cost-based overlap analysis completed.")

# ============================================================
# 3️⃣ SAVE SUMMARY TABLES
# ============================================================
excel_output = os.path.join(output_folder, "Scenario_Comparison_Summary.xlsx")
with pd.ExcelWriter(excel_output) as writer:
    summary_df.to_excel(writer, sheet_name="Criticality_Percentage", index=False)
    pairwise_overlap.to_excel(writer, sheet_name="Pairwise_Overlap")
    length_summary_df.to_excel(writer, sheet_name="Criticality_Length_Percent", index=False)
    pairwise_overlap_length.to_excel(writer, sheet_name="Pairwise_Overlap_Length")
    cost_summary_df.to_excel(writer, sheet_name="Criticality_Cost_Percent", index=False)
    pairwise_overlap_cost.to_excel(writer, sheet_name="Pairwise_Overlap_Cost")
    pd.DataFrame(
        [{"Type": "All Critical", "Count": all_critical},
         {"Type": "Any Critical", "Count": any_critical}] +
        [{"Type": f"Unique to {k}", "Count": v} for k, v in unique_critical.items()]
    ).to_excel(writer, sheet_name="Overlap_Summary", index=False)

print(f"📑 Excel summary saved: {excel_output}")

# ============================================================
# 4️⃣ VISUALIZATIONS (Bar, Venn, Heatmap)
# ============================================================
plt.figure(figsize=(8, 5))
plt.bar(summary_df['Scenario'], summary_df['Critical (%)'])
plt.ylabel("Critical Sections (%)")
plt.title("Percentage of Critical Sections")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_folder, "Critical_Percentage_BarChart.png"), dpi=300)
plt.close()

if len(decision_df.columns) == 3:
    s1, s2, s3 = decision_df.columns
    set1 = set(decision_df.index[decision_df[s1] == 1])
    set2 = set(decision_df.index[decision_df[s2] == 1])
    set3 = set(decision_df.index[decision_df[s3] == 1])

    plt.figure(figsize=(6, 6))
    venn3([set1, set2, set3], set_labels=(s1, s2, s3))
    plt.title("Critical Road Section Overlap")
    plt.savefig(os.path.join(output_folder, "Critical_Overlap_Venn.png"), dpi=300)
    plt.close()

plt.figure(figsize=(7, 6))
sns.heatmap(pairwise_overlap.astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
plt.xlabel("Scenario")
plt.ylabel("Scenario")
plt.savefig(os.path.join(output_folder, "Pairwise_Overlap_Heatmap.png"), dpi=300)
plt.close()

# LENGTH-BASED BAR CHART
plt.figure(figsize=(8, 5))
plt.bar(length_summary_df["Scenario"], length_summary_df["Critical_Length (%)"])
plt.ylabel("Critical Roadway Length (%)")
plt.xlabel("Scenario")
plt.title("Percentage of Critical Roadway Length")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_folder, "Critical_Length_Percentage_BarChart.png"), dpi=300, bbox_inches='tight')
plt.close()
print("📊 Length-based bar chart saved.")

# LENGTH-BASED OVERLAP HEATMAP
plt.figure(figsize=(7, 6))
sns.heatmap(pairwise_overlap_length.astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
plt.xlabel("Scenario")
plt.ylabel("Scenario")
plt.title("Pairwise Overlap Based on Roadway Length (%)")
plt.savefig(os.path.join(output_folder, "Pairwise_Overlap_Length_Heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("🔥 Length-based overlap heatmap saved.")

# COST-BASED BAR CHART
plt.figure(figsize=(8, 5))
plt.bar(cost_summary_df["Scenario"], cost_summary_df["Critical_Cost (%)"])
plt.ylabel("Critical Rehab Cost (%)")
plt.xlabel("Scenario")
plt.title("Percentage of Total Network Rehab Cost Used")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_folder, "Critical_Cost_Percentage_BarChart.png"), dpi=300, bbox_inches='tight')
plt.close()
print("📊 Cost-based bar chart saved.")

# COST-BASED OVERLAP HEATMAP
plt.figure(figsize=(7, 6))
sns.heatmap(pairwise_overlap_cost.astype(float), annot=True, fmt=".2f", cmap="YlOrRd")
plt.xlabel("Scenario")
plt.ylabel("Scenario")
plt.title("Pairwise Overlap Based on Rehab Cost (%)")
plt.savefig(os.path.join(output_folder, "Pairwise_Overlap_Cost_Heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("🔥 Cost-based overlap heatmap saved.")

# ============================================================
# 5️⃣ MAP CARTOGRAPHY HELPERS (scale bar, north arrow, road-class basemap)
# ============================================================
df_network["Start_X"] = df_network["Start Node X (DMS)"].apply(dms_to_dd)
df_network["Start_Y"] = df_network["Start Node Y (DMS)"].apply(dms_to_dd)
df_network["End_X"]   = df_network["End Node X (DMS)"].apply(dms_to_dd)
df_network["End_Y"]   = df_network["End Node Y (DMS)"].apply(dms_to_dd)

scenarios = list(decision_df.columns)
vectors = {s: decision_df[s].tolist() for s in scenarios}

# Common and unique edges
all_common_edges = [i for i in range(len(df_network)) if all(vectors[s][i] == 1 for s in scenarios)]
unique_edges = {
    s: [i for i in range(len(df_network)) if vectors[s][i] == 1 and sum(vectors[o][i] for o in scenarios if o != s) == 0]
    for s in scenarios
}

# Scenario-specific critical bridges (Includes links with BOTH bridge and culvert)
critical_bridges = {
    s: [i for i, row in df_network.iterrows() if row.get("Has_Bridge", "N") == "Y" and vectors[s][i] == 1]
    for s in scenarios
}

# Scenario-specific critical culverts (Strictly culverts that are NOT bridges)
critical_culverts = {
    s: [i for i, row in df_network.iterrows() if row.get("Has_Culvert", "N") == "Y" and row.get("Has_Bridge", "N") == "N" and vectors[s][i] == 1]
    for s in scenarios
}

color_map = {scenarios[0]: "orange", scenarios[1]: "blue", scenarios[2]: "green"}

# ---- Road classification (functional class proxy from posted speed limit) ----
def classify_road(speed_limit):
    if pd.isna(speed_limit):
        return "Local"
    if speed_limit >= 50:
        return "Arterial / Highway"
    if speed_limit >= 35:
        return "Collector"
    return "Local"

df_network["Road_Class"] = df_network["Speed Limit (mph)"].apply(classify_road)

ROAD_CLASS_ORDER = ["Local", "Collector", "Arterial / Highway"]  # draw local first, arterials on top
ROAD_CLASS_STYLE = {
    "Arterial / Highway": dict(color="#595959", linewidth=1.6, zorder=1.3),
    "Collector":          dict(color="#9e9e9e", linewidth=1.0, zorder=1.2),
    "Local":              dict(color="#d4d4d4", linewidth=0.5, zorder=1.1),
}

# ---- Shared map extent / local-distance scale so every panel lines up ----
all_x = pd.concat([df_network["Start_X"], df_network["End_X"]])
all_y = pd.concat([df_network["Start_Y"], df_network["End_Y"]])
X_MIN, X_MAX = all_x.min(), all_x.max()
Y_MIN, Y_MAX = all_y.min(), all_y.max()
X_PAD = (X_MAX - X_MIN) * 0.04
Y_PAD = (Y_MAX - Y_MIN) * 0.04
MAP_XLIM = (X_MIN - X_PAD, X_MAX + X_PAD)
# extra headroom on top only, so the north arrow has clear white space and doesn't sit on top of the road network
MAP_YLIM = (Y_MIN - Y_PAD, Y_MAX + Y_PAD * 3.5)

MEAN_LAT = float(all_y.mean())
MILES_PER_DEG_LAT = 69.0
MILES_PER_DEG_LON = 69.172 * np.cos(np.radians(MEAN_LAT))  # local flat-earth approximation, fine at this extent

def draw_basemap(ax):
    """Full road network as context, styled by road class (LineCollection = fast)."""
    for cls in ROAD_CLASS_ORDER:
        subset = df_network[df_network["Road_Class"] == cls]
        if subset.empty:
            continue
        style = ROAD_CLASS_STYLE[cls]
        segments = list(zip(
            zip(subset["Start_X"], subset["Start_Y"]),
            zip(subset["End_X"], subset["End_Y"]),
        ))
        lc = LineCollection(segments, colors=style["color"], linewidths=style["linewidth"],
                             zorder=style["zorder"], alpha=0.9, capstyle="round")
        ax.add_collection(lc)

def _nice_scale_length(width_miles):
    candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10]
    target = width_miles * 0.2
    return min(candidates, key=lambda v: abs(v - target))

def add_scale_bar(ax):
    """Bottom-left scale bar in miles, from the local miles-per-degree-longitude estimate."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
    width_miles = x_span * MILES_PER_DEG_LON
    bar_miles = _nice_scale_length(width_miles)
    bar_deg = bar_miles / MILES_PER_DEG_LON

    x0 = xlim[0] + 0.05 * x_span
    y0 = ylim[0] + 0.015 * y_span
    x1 = x0 + bar_deg
    tick_h = 0.01 * y_span

    ax.plot([x0, x1], [y0, y0], color="black", lw=2, solid_capstyle="butt", zorder=6)
    for x in (x0, x1):
        ax.plot([x, x], [y0 - tick_h, y0 + tick_h], color="black", lw=1.5, zorder=6)
    ax.text((x0 + x1) / 2, y0 + 0.018 * y_span, f"{bar_miles:g} mi",
            ha="center", va="bottom", fontsize=9, zorder=6)
    ax.text(x0, y0 - 0.022 * y_span, "0", ha="center", va="top", fontsize=8, zorder=6)

def add_north_arrow(ax):
    """Standard cartographic north arrow: a solid arrowhead topped with 'N'.

    Drawn in a small fixed-size (physical-inches) inset axes -- placing it straight in
    `ax`'s own axes-fraction coords would size it using `ax`'s data aspect ratio and it
    would come out stretched.
    """
    arrow_ax = inset_axes(ax, width=0.36, height=0.36, loc="upper right", borderpad=0.4)
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis("off")
    arrow_ax.patch.set_alpha(0)

    # Solid arrowhead with a concave notch at the base (classic north-arrow silhouette)
    arrow_pts = [(0.5, 1.0), (0.85, 0.15), (0.5, 0.4), (0.15, 0.15)]
    arrow_ax.add_patch(Polygon(arrow_pts, closed=True, facecolor="black", edgecolor="black"))
    arrow_ax.text(0.5, 1.0, "N", transform=arrow_ax.transAxes,
                  ha="center", va="bottom", fontsize=12, fontweight="bold")

def road_class_legend_handles():
    return [
        Line2D([0], [0], color=ROAD_CLASS_STYLE[cls]["color"],
               lw=ROAD_CLASS_STYLE[cls]["linewidth"] + 1.2, label=cls)
        for cls in ["Arterial / Highway", "Collector", "Local"]
    ]

def finalize_map_axes(ax, add_scale=True, add_north=False):
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    if add_scale:
        add_scale_bar(ax)

    if add_north:
        add_north_arrow(ax)

# ============================================================
# 6️⃣ NETWORK PLOT WITH SCENARIO-SPECIFIC BRIDGE HIGHLIGHT (enhanced)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Scenario Comparison – Critical Sections with Bridge & Culvert Highlight", fontweight="bold")
draw_basemap(ax)

for i, edge in df_network.iterrows():
    if i in critical_bridges[scenarios[0]] or i in critical_bridges[scenarios[1]] or i in critical_bridges[scenarios[2]]:
        color, lw = "red", 3
    elif i in critical_culverts[scenarios[0]] or i in critical_culverts[scenarios[1]] or i in critical_culverts[scenarios[2]]:
        color, lw = "cyan", 3
    elif i in all_common_edges:
        color, lw = "black", 3
    elif i in unique_edges[scenarios[0]]:
        color, lw = color_map[scenarios[0]], 2
    elif i in unique_edges[scenarios[1]]:
        color, lw = color_map[scenarios[1]], 2
    elif i in unique_edges[scenarios[2]]:
        color, lw = color_map[scenarios[2]], 2
    elif sum(vectors[s][i] for s in scenarios) >= 2:
        color, lw = "purple", 2
    else:
        continue  # not critical -- already shown by the road-class basemap

    ax.plot([edge["Start_X"], edge["End_X"]], [edge["Start_Y"], edge["End_Y"]],
            color=color, linewidth=lw, alpha=0.9, zorder=3)

finalize_map_axes(ax)

legend_items = [
    Line2D([0], [0], color=color_map[scenarios[0]], lw=3, label=f"Unique: {scenarios[0]}"),
    Line2D([0], [0], color=color_map[scenarios[1]], lw=3, label=f"Unique: {scenarios[1]}"),
    Line2D([0], [0], color=color_map[scenarios[2]], lw=3, label=f"Unique: {scenarios[2]}"),
    Line2D([0], [0], color="purple", lw=3, label="Shared by 2 Scenarios"),
    Line2D([0], [0], color="black", lw=4, label="Common to ALL Scenarios"),
    Line2D([0], [0], color="red", lw=3, label="Critical Bridge (incl. Bridge+Culvert)"),
    Line2D([0], [0], color="cyan", lw=3, label="Critical Culvert (Culvert only)"),
] + road_class_legend_handles()
ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.32, 1.0), frameon=True)

network_bridge_path = os.path.join(enhanced_output_folder, "Scenario_Comparison_Network_BridgeHighlight.png")
fig.savefig(network_bridge_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"🛰 Enhanced network plot with bridge/culvert highlight saved: {network_bridge_path}")

# ============================================================
# 7️⃣ FOUR SEPARATE SPECIALIZED MAPS (enhanced, scenario-specific bridges/culverts red/cyan)
# ============================================================
EQ, FL, BAL = scenarios[0], scenarios[1], scenarios[2]
vEQ, vFL, vBAL = vectors[EQ], vectors[FL], vectors[BAL]

groups = {
    "EQ": [i for i in range(len(df_network)) if vEQ[i] == 1],
    "FL": [i for i in range(len(df_network)) if vFL[i] == 1],
    "BAL": [i for i in range(len(df_network)) if vBAL[i] == 1],
    "COMMON": [i for i in range(len(df_network)) if vEQ[i] == 1 and vFL[i] == 1 and vBAL[i] == 1],
}

def scenario_legend_handles(scenario_label, base_color):
    return [
        Line2D([0], [0], color=base_color, lw=3, label=scenario_label),
        Line2D([0], [0], color="red", lw=3, label="Critical Bridge"),
        Line2D([0], [0], color="cyan", lw=3, label="Critical Culvert"),
    ] + road_class_legend_handles()

def plot_group(ax, edge_list, scenario, base_color):
    draw_basemap(ax)
    for i, edge in df_network.iterrows():
        if i in critical_bridges.get(scenario, []):
            c, lw = "red", 3
        elif i in critical_culverts.get(scenario, []):
            c, lw = "cyan", 3
        elif i in edge_list:
            c, lw = base_color, 3
        else:
            continue  # not critical -- basemap already covers it
        ax.plot([edge["Start_X"], edge["End_X"]], [edge["Start_Y"], edge["End_Y"]],
                color=c, linewidth=lw, alpha=0.9, zorder=3)
    finalize_map_axes(ax)

def save_single_group_map(edge_list, scenario, title, filename, base_color):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title, fontweight="bold")
    plot_group(ax, edge_list, scenario, base_color)
    ax.legend(handles=scenario_legend_handles(title, base_color), loc="upper right",
              bbox_to_anchor=(1.32, 1.0), frameon=True)
    fig.savefig(os.path.join(enhanced_output_folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ Saved: {filename}")

save_single_group_map(groups["EQ"], EQ, "Earthquake-Centric Critical Links", "EQ_Centric_Critical_Network.png", "orange")
save_single_group_map(groups["FL"], FL, "Flood-Centric Critical Links", "FL_Centric_Critical_Network.png", "blue")
save_single_group_map(groups["BAL"], BAL, "Balanced-Centric Critical Links", "BAL_Centric_Critical_Network.png", "green")

# COMMON map: bridge/culvert priority checked across all scenarios
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Common Critical Links (EQ + FL + BAL)", fontweight="bold")
draw_basemap(ax)
for i, edge in df_network.iterrows():
    if i in groups["COMMON"]:
        if any(i in critical_bridges[s] for s in scenarios):
            c, lw = "red", 3
        elif any(i in critical_culverts[s] for s in scenarios):
            c, lw = "cyan", 3
        else:
            c, lw = "black", 3
    else:
        continue
    ax.plot([edge["Start_X"], edge["End_X"]], [edge["Start_Y"], edge["End_Y"]],
            color=c, linewidth=lw, alpha=0.9, zorder=3)
finalize_map_axes(ax)
ax.legend(handles=scenario_legend_handles("Common to ALL", "black"), loc="upper right",
          bbox_to_anchor=(1.32, 1.0), frameon=True)
fig.savefig(os.path.join(enhanced_output_folder, "Common_Critical_Network.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("   ✔ Saved: Common_Critical_Network.png")

# ============================================================
# 8️⃣ COMPOSITE 2x2 FIGURE -- (a) EQ / (b) FL / (c) Balanced / (d) Common
# Reproduces the reviewed figure's panel layout with scale bars, north arrows,
# a road-class basemap, and a legend that documents the road classes.
# ============================================================
panels = [
    ("(a)", groups["EQ"], EQ, "orange"),
    ("(b)", groups["FL"], FL, "blue"),
    ("(c)", groups["BAL"], BAL, "green"),
    ("(d)", groups["COMMON"], "COMMON", "black"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 12))
for (label, edge_list, scenario_key, base_color), ax in zip(panels, axes.flat):

    if scenario_key == "COMMON":
        draw_basemap(ax)
        for i, edge in df_network.iterrows():
            if i in edge_list:
                if any(i in critical_bridges[s] for s in scenarios):
                    c, lw = "red", 3
                elif any(i in critical_culverts[s] for s in scenarios):
                    c, lw = "cyan", 3
                else:
                    c, lw = "black", 3

                ax.plot(
                    [edge["Start_X"], edge["End_X"]],
                    [edge["Start_Y"], edge["End_Y"]],
                    color=c, linewidth=lw, alpha=0.9, zorder=3
                )

        finalize_map_axes(ax, add_scale=True, add_north=(label == "(b)"))

    else:
        draw_basemap(ax)

        for i, edge in df_network.iterrows():
            if i in critical_bridges.get(scenario_key, []):
                c, lw = "red", 3
            elif i in critical_culverts.get(scenario_key, []):
                c, lw = "cyan", 3
            elif i in edge_list:
                c, lw = base_color, 3
            else:
                continue

            ax.plot(
                [df_network.loc[i, "Start_X"], df_network.loc[i, "End_X"]],
                [df_network.loc[i, "Start_Y"], df_network.loc[i, "End_Y"]],
                color=c, linewidth=lw, alpha=0.9, zorder=3
            )

        finalize_map_axes(ax, add_scale=True, add_north=(label == "(b)"))

    ax.text(
    0.50, 0.97, label,
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    ha="center",
    va="top")

fig.subplots_adjust(wspace=0.12, hspace=0.03, bottom=0.09)

legend_handles = [
    Line2D([0], [0], color="orange", lw=6, label="Earthquake-centric"),
    Line2D([0], [0], color="blue", lw=6, label="Flood-centric"),
    Line2D([0], [0], color="red", lw=6, label="Bridge"),
    Line2D([0], [0], color="green", lw=6, label="Balanced"),
    Line2D([0], [0], color="black", lw=6, label="Common"),
    Line2D([0], [0], color="cyan", lw=6, label="Culvert"),
] + road_class_legend_handles()

fig.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.005),
           frameon=True, fontsize=10, handlelength=1.6, columnspacing=1.0,
           labelspacing=0.4, handletextpad=0.5, borderaxespad=0.4, borderpad=0.6)

composite_path = os.path.join(enhanced_output_folder, "Composite_2x2_Network_Comparison.png")
fig.savefig(composite_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"🗺 Composite (a)-(d) network comparison figure saved: {composite_path}")

print("\n🎉 ALL processing and enhanced-map plotting completed successfully!")
print(f"➡ Enhanced maps saved in: {enhanced_output_folder}")
