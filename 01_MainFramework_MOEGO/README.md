# 01 — Main Framework: Kriging-based MO-EGO

Supports: **Table 4** (algorithm settings), **Table 5 / Table 22** (Earthquake-Centric,
Flood-Centric, and Balanced policies across San Andreas +4in/+5in/+6in and Rodgers Creek +6in
scenarios, each at 5%/10%/15% rehab budgets), and **Figures 6, 7, 8, 9**.

## Contents

- `kpls_main.py`, `kpls_moga_ego.py`, `kpls_kriging.py` — the core MO-EGO optimizer: Kriging
  surrogate (KPLS) fit on the earthquake/flood training data, MOGA search with expected-
  improvement (EI) infill, iterated to convergence.
- `Plotting/` — figure-generation scripts that read the `Results/` spreadsheets below:
  `Pareto_plot.py`, `GA_progress_decluttered.py`, `Compare_policies_Cost_MapEnhanced.py`.
- `Network_info/` the per-edge lookup tables
  `Edge_Info_Miles_DMS.xlsx` / `Edge_Info_Miles_DMS_Updated.xlsx` (road width/cost inputs).
- `Results/<Scenario>/<Budget>/` — the final Pareto-front spreadsheets
  (`ego_moga_results_*.xlsx`, plus `3-policies.xlsx`) for all 12 scenario x budget
  combinations (SanAndres+4in/+5in/+6in, RodgersCreek+6in x Result_5%/10%/15%). These are
  the source of the Table 22 numbers.

`GA_progress_decluttered.py` needs a per-generation `ego_moga_generations_*.xlsx` file,
produced by running `kpls_main.py`.

The sections below give the schema of each file the scripts read or write, so the pipeline
can be reproduced from `Combined_EQ.xlsx` / `Combined_F.xlsx` onward.

## Requirements

`kpls_main.py` needs `Combined_EQ.xlsx` and `Combined_F.xlsx` (the ~1000-LHS-policy training
dataset) at the repository root — see the top-level README for the file schema.

### `Network_info/Edge_Info_Miles_DMS.xlsx`

Per-edge network geometry lookup table, derived from the SUMO road network for
Arden-Arcade, Sacramento. Read by `Pareto_plot.py`.
Single sheet (`Sheet1`), 7,251 rows, columns:

| Edge ID | Start Node Name | End Node Name | Length (miles) | Speed Limit (mph) | Start Node X (DMS) | Start Node Y (DMS) | End Node X (DMS) | End Node Y (DMS) | Has_Bridge | Has_Culvert |
|---|---|---|---|---|---|---|---|---|---|---|
| -1001340249 | 90372688 | 9242631573 | 0.0001 | 49.70 | 121°23'32.02"W | 38°35'45.42"N | 121°23'32.02"W | 38°35'45.08"N | N | N |
| -1007619378#8 | 2957714441 | 90576104 | 0.3669 | 31.07 | 121°24'47.13"W | 38°34'28.33"N | 121°24'22.57"W | 38°34'28.49"N | N | N |

### `Network_info/Edge_Info_Miles_DMS_Updated.xlsx`

Same 7,251 rows as `Edge_Info_Miles_DMS.xlsx` with two added
columns, `Lane No.` and `Total width`, used by the rehabilitation-cost formula in
`kpls_main.py` and by `Compare_policies_Cost_MapEnhanced.py`:

| ... (same 11 columns as above) ... | Lane No. | Total width |
|---|---|---|
| ... | 1 | 3.25 |
| ... | 1 | 3.25 |

### `Results/<Scenario>/<Budget>/ego_moga_results_<timestamp>.xlsx`

Output of `kpls_main.py` (one file per scenario x budget combination, 12 total). Source of the
**Table 22** Pareto-front numbers. Three sheets:

- `Selected Points` (3 rows: `ei_eq_best`, `ei_fl_best`, `midpoint`) — columns `Label`,
  `EI_eq`, `EI_fl`, `Y_eq_original`, `Y_fl_original`, `Decision_Vector` (a 7,251-character
  string of `0`/`1`, one digit per network edge, `1` = rehabilitated under that policy).
  Example row: `Label=ei_eq_best, EI_eq=0.297358, EI_fl=0.068472, Y_eq_original=2720.252881,
  Y_fl_original=7589.143474`.
- `Front_1` — the final non-dominated Pareto front, same `EI_eq`/`EI_fl`/`Y_eq_original`/
  `Y_fl_original`/`Decision_Vector` columns, one row per Pareto-optimal policy (row count
  varies by scenario, e.g. 14 rows for RodgersCreek+6in/10%, 6 rows for SanAndres+4in/5%).
- `All_Fronts` — every non-dominated-sorted generation-200 individual (200 rows), with an
  added `Front_Index` column (1 = Pareto-optimal, 2+ = dominated fronts).

### `Results/<Scenario>/<Budget>/3-policies.xlsx`

Output of `kpls_main.py`, the three representative policies (Earthquake-Centric,
Flood-Centric, Balanced) selected from `ego_moga_results_*.xlsx`'s `Selected Points` sheet.
Source of the per-scenario/budget rows in **Table 5 / Table 22**. Single sheet, 3 rows,
columns `Rehab_Scenario` (`Earthquake-Centric` / `Flood-Centric` / `Balanced`) and
`Decision_Vector` (same 7,251-digit binary string format as above).
