import os
from pathlib import Path
import pandas as pd
import networkx as nx
import sumolib
import csv

# Repo-relative paths (standalone-repo convention). MapData/ (SUMO network, ~20-30MB) holds
# the network files this script reads; PolicyScenarios/ holds the per-policy Monte Carlo
# damage output produced by the damage-creation script, which this script consumes as input.
_HERE = Path(__file__).resolve().parent
network_file = str(_HERE / "MapData" / "arden_arcade.net.xml")
route_file = str(_HERE / "MapData" / "arden_arcade_withtype_6to9am.rou.xml")
base_folder = str(_HERE / "PolicyScenarios")

# --- Build SUMO network as a NetworkX graph ---
net = sumolib.net.readNet(network_file)
G_base = nx.DiGraph()
for edge in net.getEdges():
    if edge.getFunction() == "internal":
        continue
    edge_id = str(edge.getID())
    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()
    length = edge.getLength()
    speed = edge.getSpeed()
    travel_time = length / speed if speed > 0 else 1e6
    G_base.add_edge(from_node, to_node, edge_id=edge_id, length=length, speed=speed, travel_time=travel_time)

# --- Extract OD pairs from route file (with depart time) ---
vehicles = []
for veh in sumolib.xml.parse(route_file, 'vehicle'):
    veh_id = veh.id
    depart = float(veh.depart) if hasattr(veh, 'depart') else 0.0
    route_edges = veh.route[0].edges.split()
    start_edge = net.getEdge(route_edges[0])
    end_edge = net.getEdge(route_edges[-1])
    start_node = start_edge.getFromNode().getID()
    end_node = end_edge.getToNode().getID()
    vehicles.append({
        'veh_id': veh_id,
        'from': start_node,
        'to': end_node,
        'depart': depart
    })

total_vehicles = len(vehicles)

# --- For each policy folder, combine all scenario files from all realizations ---
for policy_folder in [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]:
    # Find all xlsx files recursively under this policy folder
    scenario_files = []
    for root, dirs, files in os.walk(policy_folder):
        for fname in files:
            if fname.endswith('.xlsx') and not fname.startswith('~$'):
                scenario_files.append(os.path.join(root, fname))
    if not scenario_files:
        continue

    summary_file = os.path.join(policy_folder, "scenario_summary.csv")
    write_header = not os.path.exists(summary_file)

    with open(summary_file, 'a', newline='') as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow([
                'Scenario', 'Vehicles Served', 'Unserved Vehicles', 'Percent Unserved (%)',
                'Total Travel Time (hr, served)', 'Baseline Travel Time (hr, served)',
                'Increase in Travel Time (hr)', 'Total Travel Distance (mi, served)',
                'Baseline Travel Distance (mi, served)', 'Increase in Travel Distance (mi)',
                'Serviceability Loss ($)',
                'Unserved Baseline Distance (mi)', 'Diff from 3:00:00 (hr)', 'Additional Loss ($)'
            ])

        total_files = len(scenario_files)
        for idx, fpath in enumerate(sorted(scenario_files)):
            df = pd.read_excel(fpath)
            # Build edge status dict for this scenario
            edge_status = {str(row['Edge ID']): float(row['Damage State']) for _, row in df.iterrows()}

            served_vehicle_results = []
            unserved_vehicles = []   # store (veh_id, from, to, depart)

            for veh in vehicles:
                from_n = veh['from']
                to_n = veh['to']
                depart_time = float(veh.get('depart', 0))

                # --- Damaged network ---
                G_mod = G_base.copy()
                for u, v, data in list(G_mod.edges(data=True)):
                    eid = data['edge_id']
                    stat = float(edge_status.get(eid, 0))
                    if stat == 1.0:
                        G_mod.remove_edge(u, v)
                    elif stat == 0.5:
                        G_mod[u][v]['travel_time'] *= 2
                    elif stat == 0.25:
                        G_mod[u][v]['travel_time'] *= 1.33
                    # stat == 0: do nothing

                try:
                    path = nx.shortest_path(G_mod, from_n, to_n, weight='travel_time')
                    time = 0.0
                    dist = 0.0
                    for i in range(len(path) - 1):
                        edge_data = G_mod[path[i]][path[i+1]]
                        time += edge_data['travel_time']
                        dist += edge_data['length']
                    served_vehicle_results.append((from_n, to_n, time, dist))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    unserved_vehicles.append((veh['veh_id'], from_n, to_n, depart_time))
                    continue

            # Baseline time/distance for served vehicles
            baseline_time = 0.0
            baseline_dist = 0.0
            for from_n, to_n, _, _ in served_vehicle_results:
                try:
                    path = nx.shortest_path(G_base, from_n, to_n, weight='travel_time')
                    time = 0.0
                    dist = 0.0
                    for i in range(len(path) - 1):
                        edge_data = G_base[path[i]][path[i+1]]
                        time += edge_data['travel_time']
                        dist += edge_data['length']
                    baseline_time += time
                    baseline_dist += dist
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            # NEW: Baseline distance for unserved vehicles
            unserved_baseline_dist = 0.0
            for _, from_n, to_n, _ in unserved_vehicles:
                try:
                    path = nx.shortest_path(G_base, from_n, to_n, weight='travel_time')
                    dist = 0.0
                    for i in range(len(path) - 1):
                        edge_data = G_base[path[i]][path[i+1]]
                        dist += edge_data['length']
                    unserved_baseline_dist += dist
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            unserved_baseline_dist_mi = unserved_baseline_dist / 1609.344

            # NEW: Time difference wrt 3:00:00 for unserved
            diff_hours = 0.0
            for _, _, _, depart_time in unserved_vehicles:
                diff_hours += (10800 - depart_time) / 3600.0  # 3 hrs in sec = 10800

            # --- Calculate Totals ---
            total_time = sum(x[2] for x in served_vehicle_results)
            total_dist = sum(x[3] for x in served_vehicle_results)
            total_time_hr = total_time / 3600
            total_dist_mi = total_dist / 1609.344
            baseline_time_hr = baseline_time / 3600
            baseline_dist_mi = baseline_dist / 1609.344

            increase_time = total_time_hr - baseline_time_hr
            increase_dist = total_dist_mi - baseline_dist_mi
            service_loss = increase_time * 30.12 + increase_dist * 0.67

            vehicles_served = len(served_vehicle_results)
            unserved = total_vehicles - vehicles_served
            percent_unserved = (unserved / total_vehicles) * 100 if total_vehicles else 0

            # NEW: Additional loss
            additional_loss = unserved_baseline_dist_mi * 0.67 + diff_hours * 30.12

            writer.writerow([
                os.path.relpath(fpath, policy_folder), vehicles_served, unserved, percent_unserved,
                total_time_hr, baseline_time_hr, increase_time,
                total_dist_mi, baseline_dist_mi, increase_dist,
                service_loss,
                unserved_baseline_dist_mi, diff_hours, additional_loss
            ])

            print(f"{os.path.relpath(fpath, base_folder)} -- Served: {vehicles_served}/{total_vehicles} | "
                  f"Unserved: {unserved} | Serviceability loss: ${service_loss:.2f} | "
                  f"Additional loss: ${additional_loss:.2f}")

print("\nAll policy-level summaries created!")
