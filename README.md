# CRN_MultiHazard_Kriging_MOEGO

Code and supporting data for:

> **Proactive Rehabilitation Planning for Concrete Road Networks Considering Multiple
> Hazards and Multiple Assets: A Kriging-based Multi-Objective Efficient Global Optimization
> Approach**
> *Resilient Cities and Structures*

A Kriging-surrogate based multi-objective optimization framework for prioritizing rehabilitation
of concrete road network assets (pavement, bridges, culverts) under earthquake and flash-flood
hazards. Rehabilitation policies are evaluated with high-fidelity earthquake/flood damage
models and SUMO traffic simulation; a Kriging surrogate coupled with a Multi-Objective
Efficient Global Optimization (MO-EGO) loop searches for Pareto-optimal policies that jointly
minimize earthquake- and flood-induced Average Disruption Cost (ADC), under a rehabilitation
budget constraint. The framework is demonstrated on the road network of Arden-Arcade,
Sacramento, California.

This repository is a reduced, reproducible case study of the framework: the optimization
scripts, the surrogate model, the hazard-model settings and network-processing routines, and
the SUMO-facing simulation code, together with the environment file needed to run them.

## Repository structure

| Folder | Contents |
|---|---|
| `01_MainFramework_MOEGO/` | The core Kriging surrogate model and MO-EGO optimizer, cost-based rehabilitation budget and network-processing routines, and result sets across earthquake scenarios, flood thresholds, and budget levels |
| `02_HighFidelity_Simulation_Engine/` | Earthquake and flood damage modeling (hazard-model settings, fragility parameters), and SUMO traffic simulation |

Each folder has its own `README.md` describing its scripts, inputs, and outputs in detail.

## Setup

```
pip install -r requirements.txt
```

**Required data**:
- `Combined_EQ.xlsx` and `Combined_F.xlsx` — the 1,000-policy Latin Hypercube training
  dataset (one file per hazard) — must be placed at the repository root. Each file has 1,000
  rows and 7,253 columns: `File Name`, `Mean Serviceability Loss ($)` (the Kriging surrogate
  training target), and `Selected_1` … `Selected_7251` (one binary column per rehabilitable
  network asset — 1 if that asset is rehabilitated under the policy, 0 otherwise). Example
  (first columns only):

  | File Name | Mean Serviceability Loss ($) | Selected_1 | Selected_2 | Selected_3 | Selected_4 |
  |---|---|---|---|---|---|
  | Policy_001 | 35072.53 (EQ) / 7781.09 (FL) | 0 | 0 | 0 | 0 |
  | Policy_002 | 4417.60 (EQ) / 3673.63 (FL) | 1 | 1 | 1 | 1 |

- SUMO network files (`arden_arcade.net.xml`, `arden_arcade_withtype_6to9am.rou.xml`) must be
  placed in the `MapData/` subfolder expected by each script under
  `02_HighFidelity_Simulation_Engine/`.

## Simulation configuration

The high-fidelity earthquake damage model (`02_HighFidelity_Simulation_Engine/`) generates
20 stochastic PGV field realizations per rehabilitation policy (spatially correlated
inter-event and intra-event ground-motion variability), each evaluated with 130 Monte Carlo
damage-state draws (2,600 total damage scenarios per policy). The flood damage model runs
130 Monte Carlo draws per policy. Both counts were selected via a convergence study of mean
Average Disruption Cost.

The MO-EGO optimizer uses a population of 200, 100 generations per infill iteration, uniform
crossover, bit-flip mutation, binary-tournament NSGA-II-elitist selection, a greedy
budget-repair constraint-handling procedure, 3 infill points per iteration, and a stopping
rule of 2 consecutive iterations without a new non-dominated policy (maximum 5 iterations).

## External tools (not part of this Python codebase)

- **OpenSHA ShakeMap Calculator** — generates the PGV field realizations, consumed here as
  `Edge_with_Avg_PGV*.xlsx` input files.
- **ArcGIS Pro** — generates the flood inundation depths, consumed here as the
  `Flood Depth in inch with traffic stop mark - Cleaned.xlsx` input file.

## License

Released under the [MIT License](LICENSE).

## How to cite

See [`CITATION.cff`](CITATION.cff) for structured citation metadata (also used by GitHub's
"Cite this repository" button).
