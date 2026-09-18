import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Academic Plot Settings (Journal Standard)
# ----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

def get_pareto_front(df, x_col, y_col):
    """
    Extracts the non-dominated solutions (Pareto front) for minimization objectives.
    """
    vals = df[[x_col, y_col]].values
    is_efficient = np.ones(vals.shape[0], dtype=bool)
    for i, c in enumerate(vals):
        if is_efficient[i]:
            # Keep solutions that are better in at least one objective and not worse in both
            is_efficient[is_efficient] = np.any(vals[is_efficient] < c, axis=1) | np.all(vals[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return df[is_efficient].sort_values(by=x_col)

def pick_snapshot_generations(gens, n_snapshots):
    """Evenly spaced generation indices (endpoints always kept).

    Pareto-front evolution figures get visually congested when every generation's
    front is drawn -- with 100+ generations that's 100+ overlapping curves. Instead
    we plot a small, legible set of snapshots spanning the same range; the colorbar
    still documents the full generation range.
    """
    gens = sorted(gens)
    if len(gens) <= n_snapshots:
        return gens
    idx = sorted(set(int(round(i)) for i in np.linspace(0, len(gens) - 1, n_snapshots)))
    return [gens[i] for i in idx]

def main():
    # =========================================================================
    # 1. USER CONFIGURATION BLOCK
    # =========================================================================
    # Input file location
    # ego_moga_generations_*.xlsx (the full per-generation population history) is produced by
    # running kpls_main.py (one directory up from this script). This example points at the
    # SanAndres+6in/Result_10% scenario; update file_path below to point at your own run.
    _repo_root = Path(__file__).resolve().parent.parent
    file_path = str(_repo_root / "Results" / "SanAndres+6in" / "Result_10%" / "ego_moga_generations_20260902_161022.xlsx")
    output_dir = str(Path(__file__).resolve().parent / "GA_Progress_Decluttered")

    # -------------------------------------------------------------------------
    # GENERATION RANGE SELECTION FOR PLOTTING
    # Set to None to automatically use all available generations,
    # or specify explicit integer limits (e.g., start_gen=80, end_gen=100)
    # -------------------------------------------------------------------------
    start_gen = 10  # e.g., 0 or 80
    end_gen = None    # e.g., 50 or 100

    # How many generation fronts Plot 1 actually draws (endpoints always included).
    # This is the knob that keeps the progression figure legible -- see
    # pick_snapshot_generations() above.
    n_snapshots = 15

    # Objective and column names
    x_col = 'Y_eq_original'
    y_col = 'Y_fl_original'
    budget_col = 'Budget_Used'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 2. DATA LOADING & FILTERING
    # =========================================================================
    print("Reading Excel spreadsheet sheets...")
    xl = pd.ExcelFile(file_path)

    # Strategy A: Check if a unified 'All_Generations' sheet exists
    if 'All_Generations' in xl.sheet_names:
        print("Found consolidated 'All_Generations' worksheet. Loading data...")
        df_all = pd.read_excel(xl, sheet_name='All_Generations')
    else:
        # Strategy B: If individual worksheets are used instead, compile them dynamically
        print("Consolidated sheet not found. Compiling data from individual sheet iterations...")
        gen_sheets = [s for s in xl.sheet_names if 'Generation_' in s]
        gen_sheets.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        dfs = []
        for sheet in gen_sheets:
            g_num = int(re.search(r'\d+', sheet).group())
            df_s = pd.read_excel(xl, sheet_name=sheet)
            df_s['Generation'] = g_num
            dfs.append(df_s)
        df_all = pd.concat(dfs, ignore_index=True)

    # Deduce active limits if none are explicitly specified
    if start_gen is None:
        start_gen = int(df_all['Generation'].min())
    if end_gen is None:
        end_gen = int(df_all['Generation'].max())

    print(f"Applying generation range filter: Generations [{start_gen} to {end_gen}]")
    df_filtered = df_all[(df_all['Generation'] >= start_gen) & (df_all['Generation'] <= end_gen)].copy()

    if df_filtered.empty:
        print("Error: Filtered range returned an empty dataset. Check your generation limits.")
        return

    # Extract global and Pareto subsets across active range
    unique_gens = sorted(df_filtered['Generation'].unique())
    pareto_list = []
    for g in unique_gens:
        df_g = df_filtered[df_filtered['Generation'] == g]
        front_g = get_pareto_front(df_g, x_col, y_col).copy()
        front_g['Generation'] = g
        pareto_list.append(front_g)
    df_pareto = pd.concat(pareto_list, ignore_index=True)

    # Extract Pareto across ALL generations (for Plot 2 only)
    unique_gens_all = sorted(df_all['Generation'].unique())

    pareto_list_all = []
    for g in unique_gens_all:
        df_g = df_all[df_all['Generation'] == g]
        front_g = get_pareto_front(df_g, x_col, y_col).copy()
        front_g['Generation'] = g
        pareto_list_all.append(front_g)

    df_pareto_all = pd.concat(pareto_list_all, ignore_index=True)


    # =========================================================================
    # PLOT 1: Unified Continuous Pareto Front Progression (Gradient View)
    # =========================================================================
    print("Generating Plot 1: Unified Pareto Frontier Line Progression...")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(start_gen, end_gen)

    # Drawing every generation's front (often 100+) stacks near-identical curves on
    # top of each other, producing a visually congested figure. Instead, draw a
    # small set of evenly spaced snapshots -- the colorbar still spans the full
    # [start_gen, end_gen] range for context.
    snapshot_gens = pick_snapshot_generations(unique_gens, n_snapshots)
    print(f"   Plotting {len(snapshot_gens)} snapshot generations (of {len(unique_gens)} available): {snapshot_gens}")

    for g in snapshot_gens:
        df_g = df_pareto[df_pareto['Generation'] == g]
        if df_g.empty:
            continue
        color = cmap(norm(g))
        is_final = (g == snapshot_gens[-1])
        lw = 3.0 if is_final else 1.6
        alpha = 1.0 if is_final else 0.8
        marker_size = 36 if is_final else 22
        ax.plot(df_g[x_col], df_g[y_col], color=color, linewidth=lw, alpha=alpha, zorder=3)
        ax.scatter(df_g[x_col], df_g[y_col], color=color, s=marker_size, alpha=alpha,
                   zorder=4, edgecolor='white', linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Generation Index', rotation=270, labelpad=15, fontsize=16, weight='bold')

    ax.set_xlabel('Average Disruption Cost under Earthquake ($ADC_{EQ}$)', fontsize=16, labelpad=8)
    ax.set_ylabel('Average Disruption Cost under Flood ($ADC_{FL}$)', fontsize=16, labelpad=8)
    ax.set_title(f'Pareto Front Progression (Generations {start_gen} to {end_gen})', fontsize=16, weight='bold', pad=15)
    ax.text(0.99, 0.02, f'{len(snapshot_gens)} representative generations shown',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, style='italic', color='#555555')
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'pareto_front_progression_unified.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"-> Saved: {plot1_path}")

    # =========================================================================
    # PLOT 2: Continuous Range Shaded Envelopes (Metrics over Frontiers)
    # =========================================================================
    print("Generating Plot 2: Pareto Distribution Envelopes...")
    plt.figure(figsize=(14, 6))

    # Subplot A: Earthquake Loss
    plt.subplot(1, 2, 1)
    stats_eq = df_pareto_all.groupby('Generation')[x_col].agg(['min', 'median', 'max', 'mean']).reset_index()
    plt.plot(stats_eq['Generation'], stats_eq['median'], label='Median', color='#1f77b4', linewidth=2)
    plt.plot(stats_eq['Generation'], stats_eq['mean'], label='Mean', color='#1f77b4', linestyle=':', linewidth=1.5)
    plt.fill_between(stats_eq['Generation'], stats_eq['min'], stats_eq['max'], alpha=0.15, color='#1f77b4', label='Min-Max Range')
    plt.xlabel('Generation Index', fontsize=18)
    plt.ylabel('$ADC_{EQ}$', fontsize=18)
    plt.title('Distribution of $ADC_{EQ}$ on Pareto Fronts', fontsize=20, weight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # Subplot B: Flood Loss
    plt.subplot(1, 2, 2)
    stats_fl = df_pareto_all.groupby('Generation')[y_col].agg(['min', 'median', 'max', 'mean']).reset_index()
    plt.plot(stats_fl['Generation'], stats_fl['median'], label='Median', color='#ff7f0e', linewidth=2)
    plt.plot(stats_fl['Generation'], stats_fl['mean'], label='Mean', color='#ff7f0e', linestyle=':', linewidth=1.5)
    plt.fill_between(stats_fl['Generation'], stats_fl['min'], stats_fl['max'], alpha=0.15, color='#ff7f0e', label='Min-Max Range')
    plt.xlabel('Generation Index', fontsize=18)
    plt.ylabel('$ADC_{FL}$', fontsize=18)
    plt.title('Distribution of $ADC_{FL}$ on Pareto Fronts', fontsize=20, weight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'pareto_front_range_progression.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"-> Saved: {plot2_path}")

    # =========================================================================
    # PLOT 3: Generation Snapshot Interval Boxplots
    # =========================================================================
    print("Generating Plot 3: Discrete Snapshot Boxplots...")
    # Determine intervals dynamically based on user selection range
    span = end_gen - start_gen
    step = 5 if span <= 50 else 10

    selected_gens = list(range(start_gen, end_gen + 1, step))
    if end_gen not in selected_gens:
        selected_gens.append(end_gen)

    df_sampled = df_pareto[df_pareto['Generation'].isin(selected_gens)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # EQ Boxplot
    sns.boxplot(ax=axes[0], data=df_sampled, x='Generation', y=x_col, color='#1f77b4', width=0.5)
    axes[0].set_xlabel('Generation Snapshot', fontsize=16)
    axes[0].set_ylabel('EQ Loss ($Y_{eq}$)', fontsize=16)
    axes[0].set_title('Pareto Front Distribution Shift Profile: $Y_{eq}$', weight='bold', fontsize=13)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # FL Boxplot
    sns.boxplot(ax=axes[1], data=df_sampled, x='Generation', y=y_col, color='#ff7f0e', width=0.5)
    axes[1].set_xlabel('Generation Snapshot', fontsize=16)
    axes[1].set_ylabel('FL Loss ($Y_{fl}$)', fontsize=16)
    axes[1].set_title('Pareto Front Distribution Shift Profile: $Y_{fl}$', weight='bold', fontsize=13)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'pareto_front_boxplots_intervals.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"-> Saved: {plot3_path}")

    # =========================================================================
    # PLOT 4: Total Budget Consumption Metric Tracking
    # =========================================================================
    if budget_col in df_filtered.columns:
        print("Generating Plot 4: Mean Budget Tracking...")
        stats_budget = df_filtered.groupby('Generation')[budget_col].mean().reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(stats_budget['Generation'], stats_budget[budget_col], color='teal', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Generation Index', fontsize=16)
        ax.set_ylabel('Mean Budget Used', fontsize=16)
        ax.set_title('Budget Consumption over Generations', fontsize=16, weight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plot4_path = os.path.join(output_dir, 'budget_consumption_progression.png')
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        print(f"-> Saved: {plot4_path}")

    print("\nProcessing completed successfully. All figures exported.")

if __name__ == "__main__":
    main()
