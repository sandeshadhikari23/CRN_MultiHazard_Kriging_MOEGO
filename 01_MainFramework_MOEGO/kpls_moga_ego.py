"""
moga_ego.py
Multi‑objective GA (NSGA‑II‑like) using the paper’s KPLS model (Bouhlel et al. 2016)
for binary candidate evaluation. Uses log(Y+1) transformation and one PLS component.

Modified for high‑dimensional problems (e.g., 7521 bits):
- Adaptive mutation probability = 1 / dimension
- Larger population size (500+)
- More generations (500+)
- Uniform crossover (default) to improve mixing
"""

import numpy as np
import datetime
import pandas as pd
import os
from pathlib import Path
from kpls_kriging import (
    fit_kpls_model,
    kpls_predict,
    kpls_expected_improvement,
)

OUTPUT_DIR = Path(__file__).resolve().parent  # ".../KPLS/cost_based optimization" -- all exports go here

# ----------------------------------------------------------------------
# Core GA operators (unchanged except added uniform crossover)
# ----------------------------------------------------------------------

def nondominated_sort(pop_objs):
    """
    pop_objs: numpy array shape (N, M), larger-is-better for every objective.
    Returns list of fronts (each front is list of indices).
    """
    pop_objs = np.asarray(pop_objs, dtype=np.float64)
    N = pop_objs.shape[0]
    S = [set() for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if (pop_objs[p] >= pop_objs[q]).all() and (pop_objs[p] > pop_objs[q]).any():
                S[p].add(q)
            elif (pop_objs[q] >= pop_objs[p]).all() and (pop_objs[q] > pop_objs[p]).any():
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(pop_objs, front):
    front = list(front)
    if len(front) == 0:
        return np.array([], dtype=np.float64)
    objs = np.asarray(pop_objs)[front]
    num_obj = objs.shape[1]
    dist = np.zeros(len(front), dtype=np.float64)
    for m in range(num_obj):
        vals = objs[:, m]
        sorted_idx = np.argsort(vals)
        vmin = vals[sorted_idx[0]]
        vmax = vals[sorted_idx[-1]]
        dist[sorted_idx[0]] = np.inf
        dist[sorted_idx[-1]] = np.inf
        if vmax - vmin == 0:
            continue
        for k in range(1, len(front) - 1):
            dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (vmax - vmin)
    return dist

def tournament_select(pop, pop_objs):
    """Return the better of two randomly-chosen individuals (by sum of objectives)."""
    N = len(pop)
    i, j = np.random.randint(0, N), np.random.randint(0, N)
    ai = pop_objs[i].sum()
    aj = pop_objs[j].sum()
    return pop[i] if ai >= aj else pop[j]

def one_point_crossover(a, b):
    d = a.size
    if d <= 1:
        return a.copy(), b.copy()
    pt = np.random.randint(1, d)
    c1 = np.concatenate([a[:pt], b[pt:]])
    c2 = np.concatenate([b[:pt], a[pt:]])
    return c1.astype(np.int64), c2.astype(np.int64)

def uniform_crossover(a, b, p=0.5):
    """
    Uniform crossover: each bit is chosen from parent A with probability p,
    from parent B with probability 1-p.
    """
    mask = np.random.rand(a.size) < p
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1.astype(np.int64), c2.astype(np.int64)

def bitflip_mutation(x, prob):
    x = x.copy()
    mask = (np.random.rand(x.size) < prob)
    x[mask] = 1 - x[mask]
    return x.astype(np.int64)

def project_to_budget(x_bin, costs, budget):
    """
    If x_bin violates budget, greedily remove ones with largest cost until feasible.
    """
    x = x_bin.astype(np.int64).copy()
    total = float(np.dot(x.astype(np.float64), costs.astype(np.float64)))
    if total <= float(budget):
        return x
    ones_idx = np.where(x == 1)[0]
    order = np.argsort(-costs[ones_idx])
    for idx in ones_idx[order]:
        x[idx] = 0
        total = float(np.dot(x.astype(np.float64), costs.astype(np.float64)))
        if total <= float(budget):
            break
    return x.astype(np.int64)

def neighbors_of(x0, n_single=50, n_double=50, d=None, costs=None, budget=None):
    x0 = np.asarray(x0, dtype=np.int64).flatten()
    d = int(x0.size) if d is None else int(d)
    neighs = set()
    neighs.add(tuple(x0.tolist()))
    # single flips
    n_single = min(n_single, d)
    idxs = np.random.choice(d, size=n_single, replace=False)
    for i in idxs:
        x1 = x0.copy()
        x1[i] = 1 - x1[i]
        if costs is None or budget is None or float(np.dot(x1.astype(np.float64), costs.astype(np.float64))) <= float(budget):
            neighs.add(tuple(x1.tolist()))
    # double flips
    for _ in range(int(n_double)):
        i, j = np.random.choice(d, size=2, replace=False)
        x2 = x0.copy()
        x2[i] = 1 - x2[i]
        x2[j] = 1 - x2[j]
        if costs is None or budget is None or float(np.dot(x2.astype(np.float64), costs.astype(np.float64))) <= float(budget):
            neighs.add(tuple(x2.tolist()))
    return [np.array(t, dtype=np.int64) for t in neighs]

# ----------------------------------------------------------------------
# MOGA using KPLS (modified for high dimension)
# ----------------------------------------------------------------------

def run_moga(
    candidates,
    # EQ KPLS parameters
    X_obs_eq, Z_train_eq, Y_obs_eq, mu_eq, sigma_eq, Rinv_eq, eta_eq, x_mean_eq, x_std_eq,
    # FL KPLS parameters
    X_obs_fl, Z_train_fl, Y_obs_fl, mu_fl, sigma_fl, Rinv_fl, eta_fl, x_mean_fl, x_std_fl,
    costs, budget,
    pop_size=500, gens=500, cx_prob=0.9, mut_prob=0.000138, crossover_type='uniform',
):
    """
    candidates: list of binary arrays to seed population.
    Returns final pop (list of binary arrays) and pop_objs (N,2) for (EI_eq, EI_fl).

    New parameters for high dimension:
        pop_size   : population size (default 500)
        gens       : number of generations (default 500)
        mut_prob   : mutation probability per bit. If None, set to 1.0/dimension.
        crossover_type: 'uniform' (default) or 'one_point'.
    """
    generation_history = []
    cand_set = {tuple(c.astype(np.int64).tolist()) for c in candidates}
    pop = [np.array(t, dtype=np.int64) for t in cand_set]
    if not pop:
        raise RuntimeError("No initial candidates provided to MOGA.")

    d = int(pop[0].size)

    # Set mutation probability if not provided
    if mut_prob is None:
        mut_prob = 1.0 / d   # one flipped bit on average per individual
    print(f"MOGA: dimension={d}, pop_size={pop_size}, gens={gens}, mut_prob={mut_prob:.6f}, crossover={crossover_type}")

    while len(pop) < pop_size:
        xr = np.random.randint(0, 2, size=d, dtype=np.int64)
        if float(np.dot(xr.astype(np.float64), costs.astype(np.float64))) <= float(budget):
            pop.append(xr)
    pop = pop[:pop_size]

    # Evaluation function: compute EI using KPLS for both hazards (on log scale)
    def eval_pop(pop_list):
        objs = []
        # y_min from feasible observations (both EQ and FL) – these are already log-transformed
        rehab_costs = (X_obs_eq.astype(np.int64) @ costs.astype(np.float64)).astype(np.float64)
        feasible = rehab_costs <= float(budget)
        if np.any(feasible):
            y_min_eq = float(np.min(Y_obs_eq[feasible]))
            y_min_fl = float(np.min(Y_obs_fl[feasible]))
        else:
            y_min_eq = float(np.min(Y_obs_eq))
            y_min_fl = float(np.min(Y_obs_fl))

        for x in pop_list:
            ei_eq = kpls_expected_improvement(
                x, Z_train_eq, Y_obs_eq, mu_eq, sigma_eq, Rinv_eq,
                eta_eq, x_mean_eq, x_std_eq, y_min_eq,
            )
            ei_fl = kpls_expected_improvement(
                x, Z_train_fl, Y_obs_fl, mu_fl, sigma_fl, Rinv_fl,
                eta_fl, x_mean_fl, x_std_fl, y_min_fl,
            )
            objs.append([float(ei_eq), float(ei_fl)])
        return np.asarray(objs, dtype=np.float64)

    pop_objs = eval_pop(pop)

    generation_history.append({
        'generation': 0,
        'population': [x.copy() for x in pop],
        'objectives': pop_objs.copy(),
    })

    # Choose crossover function
    if crossover_type == 'uniform':
        crossover_func = uniform_crossover
    else:
        crossover_func = one_point_crossover

    for gen in range(int(gens)):
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, pop_objs)
            p2 = tournament_select(pop, pop_objs)
            if np.random.rand() < cx_prob:
                c1, c2 = crossover_func(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = bitflip_mutation(c1, mut_prob)
            c2 = bitflip_mutation(c2, mut_prob)
            c1 = project_to_budget(c1, costs, budget)
            c2 = project_to_budget(c2, costs, budget)
            offspring.append(c1)
            offspring.append(c2)
        offspring = offspring[:pop_size]
        off_objs = eval_pop(offspring)

        combined = pop + offspring
        combined_objs = np.vstack([pop_objs, off_objs])
        fronts = nondominated_sort(combined_objs)
        new_pop, new_objs = [], []
        for f in fronts:
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend([combined[i] for i in f])
                new_objs.extend([combined_objs[i] for i in f])
            else:
                dist = crowding_distance(combined_objs, f)
                order = np.argsort(-dist)
                needed = pop_size - len(new_pop)
                chosen = [f[idx] for idx in order[:needed]]
                new_pop.extend([combined[i] for i in chosen])
                new_objs.extend([combined_objs[i] for i in chosen])
                break
        pop = [np.array(x, dtype=np.int64) for x in new_pop]
        pop_objs = np.asarray(new_objs, dtype=np.float64)

        generation_history.append({
            'generation': gen + 1,
            'population': [x.copy() for x in pop],
            'objectives': pop_objs.copy(),
        })

    return pop, pop_objs, generation_history

# ----------------------------------------------------------------------
# EGO iteration (calls the improved MOGA)
# ----------------------------------------------------------------------

def ego_moga_iteration(
    X_obs_eq, Y_obs_eq, X_obs_fl, Y_obs_fl,
    costs, rehab_budget,
    simulation_oracle_func=None,
    n_neighbors_single=50, n_neighbors_double=50,
    moga_pop=500, moga_gens=500,   # increased for high dimension
    n_pls_components=1,   # this parameter will be overridden to 1 inside (kept for compatibility)
    crossover_type='uniform',
):
    """
    Fit KPLS models for earthquake and flood, generate union neighbors,
    run MOGA on (EI_eq, EI_fl), and return three infill points.
    The Y_* arrays are assumed to be log(Y+1) transformed already.

    Modified: now accepts moga_pop, moga_gens, and crossover_type.
    """
    # ---- 1. Fit KPLS models with one PLS component for both ----
    (mu_eq, sigma_eq, Rinv_eq, theta_eq, W_eq, eta_eq, x_mean_eq, x_std_eq) = fit_kpls_model(
        X_obs_eq, Y_obs_eq, n_components=1
    )
    Z_train_eq = (X_obs_eq.astype(np.float64) - x_mean_eq) / x_std_eq

    (mu_fl, sigma_fl, Rinv_fl, theta_fl, W_fl, eta_fl, x_mean_fl, x_std_fl) = fit_kpls_model(
        X_obs_fl, Y_obs_fl, n_components=1
    )
    Z_train_fl = (X_obs_fl.astype(np.float64) - x_mean_fl) / x_std_fl

    d = int(X_obs_eq.shape[1])

    # ---- 2. Best feasible observed points (for neighbor generation) ----
    rehab_costs_eq = (X_obs_eq.astype(np.int64) @ costs.astype(np.float64)).astype(np.float64)
    feasible_eq = rehab_costs_eq <= float(rehab_budget)
    if np.any(feasible_eq):
        idx_best_eq = int(np.nonzero(feasible_eq)[0][np.argmin(Y_obs_eq[feasible_eq])])
        best_x_eq = X_obs_eq[idx_best_eq].astype(np.int64)
    else:
        best_x_eq = X_obs_eq[int(np.argmin(Y_obs_eq))].astype(np.int64)

    rehab_costs_fl = (X_obs_fl.astype(np.int64) @ costs.astype(np.float64)).astype(np.float64)
    feasible_fl = rehab_costs_fl <= float(rehab_budget)
    if np.any(feasible_fl):
        idx_best_fl = int(np.nonzero(feasible_fl)[0][np.argmin(Y_obs_fl[feasible_fl])])
        best_x_fl = X_obs_fl[idx_best_fl].astype(np.int64)
    else:
        best_x_fl = X_obs_fl[int(np.argmin(Y_obs_fl))].astype(np.int64)

    # ---- 3. Neighbors + random candidates ----
    neighs_eq = neighbors_of(best_x_eq, n_single=n_neighbors_single,
                             n_double=n_neighbors_double, d=d,
                             costs=costs, budget=rehab_budget)
    neighs_fl = neighbors_of(best_x_fl, n_single=n_neighbors_single,
                             n_double=n_neighbors_double, d=d,
                             costs=costs, budget=rehab_budget)
    combined_candidates = neighs_eq + neighs_fl

    attempts = 0
    while len(combined_candidates) < 300 and attempts < 2000:
        xr = np.random.randint(0, 2, size=d, dtype=np.int64)
        if float(np.dot(xr.astype(np.float64), costs.astype(np.float64))) <= float(rehab_budget):
            combined_candidates.append(xr)
        attempts += 1

    # ---- 4. Run MOGA with improved parameters ----
    pop, pop_objs, generation_history = run_moga(
        combined_candidates,
        X_obs_eq, Z_train_eq, Y_obs_eq, mu_eq, sigma_eq, Rinv_eq, eta_eq, x_mean_eq, x_std_eq,
        X_obs_fl, Z_train_fl, Y_obs_fl, mu_fl, sigma_fl, Rinv_fl, eta_fl, x_mean_fl, x_std_fl,
        costs, rehab_budget,
        pop_size=moga_pop, gens=moga_gens, crossover_type=crossover_type,
    )

    # ---- 5. Extract Pareto front and select three points ----
    fronts = nondominated_sort(pop_objs)
    pareto_indices = fronts[0] if len(fronts) > 0 else list(range(len(pop)))
    pareto_solutions = [pop[i] for i in pareto_indices]
    pareto_objs = pop_objs[pareto_indices]

    max_ei_eq_idx = int(np.argmax(pareto_objs[:, 0]))
    max_ei_fl_idx = int(np.argmax(pareto_objs[:, 1]))
    x_ei_eq = pareto_solutions[max_ei_eq_idx]
    x_ei_fl = pareto_solutions[max_ei_fl_idx]

    mean_vec = np.mean(np.vstack([x.astype(np.float64) for x in pareto_solutions]), axis=0)
    mid_bin = (mean_vec >= 0.5).astype(np.int64)
    mid_bin = project_to_budget(mid_bin, costs, rehab_budget)

    # ---- 6. Helper to back‑transform predictions to original scale ----
    # Bias-corrected lognormal back-transform (per the 92.5th-truncation+log
    # diagnostic): the log-scale GP posterior at x is ~Normal(y_log, s_log^2),
    # so E[exp(Z)] = exp(mu + s^2/2) for Z~Normal(mu, s^2) -- the naive
    # exp(y_log)-1 back-transform instead recovers the conditional MEDIAN,
    # which underestimates the mean for right-skewed loss data.
    def predict_original(x):
        y_eq_log, s_eq_log = kpls_predict(x, Z_train_eq, Y_obs_eq, mu_eq, sigma_eq, Rinv_eq,
                                          eta_eq, x_mean_eq, x_std_eq)
        y_fl_log, s_fl_log = kpls_predict(x, Z_train_fl, Y_obs_fl, mu_fl, sigma_fl, Rinv_fl,
                                          eta_fl, x_mean_fl, x_std_fl)
        y_eq_orig = np.exp(y_eq_log + 0.5 * s_eq_log ** 2) - 1.0
        y_fl_orig = np.exp(y_fl_log + 0.5 * s_fl_log ** 2) - 1.0
        return y_eq_orig, y_fl_orig

    # ---- 7. Compute EIs and predicted losses (original scale for reporting) ----
    rehab_costs_all = (X_obs_eq.astype(np.int64) @ costs.astype(np.float64)).astype(np.float64)
    feasible_all = rehab_costs_all <= float(rehab_budget)
    if np.any(feasible_all):
        y_min_eq = float(np.min(Y_obs_eq[feasible_all]))
        y_min_fl = float(np.min(Y_obs_fl[feasible_all]))
    else:
        y_min_eq = float(np.min(Y_obs_eq))
        y_min_fl = float(np.min(Y_obs_fl))

    def kpls_ei_for_point(x):
        ei_eq = kpls_expected_improvement(
            x, Z_train_eq, Y_obs_eq, mu_eq, sigma_eq, Rinv_eq,
            eta_eq, x_mean_eq, x_std_eq, y_min_eq,
        )
        ei_fl = kpls_expected_improvement(
            x, Z_train_fl, Y_obs_fl, mu_fl, sigma_fl, Rinv_fl,
            eta_fl, x_mean_fl, x_std_fl, y_min_fl,
        )
        return float(ei_eq), float(ei_fl)

    ei_e_eq, ei_e_fl = kpls_ei_for_point(x_ei_eq)
    ei_f_eq, ei_f_fl = kpls_ei_for_point(x_ei_fl)
    ei_m_eq, ei_m_fl = kpls_ei_for_point(mid_bin)

    # Original scale predictions for reporting
    y_e_eq_orig, y_e_fl_orig = predict_original(x_ei_eq)
    y_f_eq_orig, y_f_fl_orig = predict_original(x_ei_fl)
    y_m_eq_orig, y_m_fl_orig = predict_original(mid_bin)

    # ---- 8. Export to Excel ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"ego_moga_results_{timestamp}.xlsx"

    df_selected = pd.DataFrame({
        'Label': ['ei_eq_best', 'ei_fl_best', 'midpoint'],
        'EI_eq': [ei_e_eq, ei_f_eq, ei_m_eq],
        'EI_fl': [ei_e_fl, ei_f_fl, ei_m_fl],
        'Y_eq_original': [y_e_eq_orig, y_f_eq_orig, y_m_eq_orig],
        'Y_fl_original': [y_e_fl_orig, y_f_fl_orig, y_m_fl_orig],
        'Decision_Vector': [
            ''.join(map(str, x_ei_eq.tolist())),
            ''.join(map(str, x_ei_fl.tolist())),
            ''.join(map(str, mid_bin.tolist())),
        ]
    })

    # Pareto front original predictions
    pareto_Y_eq_orig = [predict_original(x)[0] for x in pareto_solutions]
    pareto_Y_fl_orig = [predict_original(x)[1] for x in pareto_solutions]

    df_pareto = pd.DataFrame({
        'EI_eq': pareto_objs[:, 0],
        'EI_fl': pareto_objs[:, 1],
        'Y_eq_original': pareto_Y_eq_orig,
        'Y_fl_original': pareto_Y_fl_orig,
        'Decision_Vector': [''.join(map(str, x.tolist())) for x in pareto_solutions]
    })

    # All fronts combined
    all_fronts_data = []
    for f_idx, front in enumerate(fronts, start=1):
        front_sols = [pop[i] for i in front]
        front_objs = pop_objs[front]
        f_Y_eq_orig = [predict_original(x)[0] for x in front_sols]
        f_Y_fl_orig = [predict_original(x)[1] for x in front_sols]
        df_front = pd.DataFrame({
            'Front_Index': f_idx,
            'EI_eq': front_objs[:, 0],
            'EI_fl': front_objs[:, 1],
            'Y_eq_original': f_Y_eq_orig,
            'Y_fl_original': f_Y_fl_orig,
            'Decision_Vector': [''.join(map(str, x.tolist())) for x in front_sols]
        })
        all_fronts_data.append(df_front)
    df_all_fronts = pd.concat(all_fronts_data, ignore_index=True)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_selected.to_excel(writer, index=False, sheet_name='Selected Points')
        df_pareto.to_excel(writer, index=False, sheet_name='Front_1')
        df_all_fronts.to_excel(writer, index=False, sheet_name='All_Fronts')

    print(f"✅ Results exported to {os.path.abspath(filename)}")

    # ---- Export all generations to a single Excel sheet with original losses ----
    gen_filename = OUTPUT_DIR / f"ego_moga_generations_{timestamp}.xlsx"
    all_generations = []

    for gen_data in generation_history:
        gen_idx = gen_data['generation']
        gen_pop = gen_data['population']
        gen_objs = gen_data['objectives']

        pred_eq_orig = []
        pred_fl_orig = []
        budgets_used = []

        for x in gen_pop:
            y_eq_orig, y_fl_orig = predict_original(x)
            pred_eq_orig.append(y_eq_orig)
            pred_fl_orig.append(y_fl_orig)
            budgets_used.append(float(np.dot(x.astype(np.float64), costs.astype(np.float64))))

        df_gen = pd.DataFrame({
            'Generation': gen_idx,
            'EI_eq': gen_objs[:, 0],
            'EI_fl': gen_objs[:, 1],
            'Y_eq_original': pred_eq_orig,
            'Y_fl_original': pred_fl_orig,
            'Budget_Used': budgets_used,
            'Decision_Vector': [''.join(map(str, x.tolist())) for x in gen_pop]
        })
        all_generations.append(df_gen)

    df_all_generations = pd.concat(all_generations, ignore_index=True)

    with pd.ExcelWriter(gen_filename, engine='openpyxl') as writer:
        df_all_generations.to_excel(writer, index=False, sheet_name='All_Generations')

    print(f"✅ Generation history exported to {os.path.abspath(gen_filename)}")

    selected = {
        'ei_eq_best': (x_ei_eq.astype(np.int64), ei_e_eq, ei_e_fl),
        'ei_fl_best': (x_ei_fl.astype(np.int64), ei_f_eq, ei_f_fl),
        'midpoint':   (mid_bin.astype(np.int64), ei_m_eq, ei_m_fl),
        'pareto_solutions': pareto_solutions,
        'pareto_objs': pareto_objs,
    }
    return selected