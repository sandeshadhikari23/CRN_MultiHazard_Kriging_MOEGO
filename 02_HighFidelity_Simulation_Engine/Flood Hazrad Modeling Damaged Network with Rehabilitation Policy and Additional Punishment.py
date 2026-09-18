import os
from pathlib import Path
import pandas as pd
import networkx as nx
import sumolib
import csv

# Repo-relative paths (standalone-repo convention). MapData/ (SUMO network, ~20-30MB) holds
# the network files this script reads; Flood_MonteCarlo_withPolicy/ holds the per-policy
# Monte Carlo damage output produced by the damage-scenario script, which this script
# consumes as input.
_HERE = Path(__file__).resolve().parent
network_file = str(_HERE / "MapData" / "arden_arcade.net.xml")
route_file = str(_HERE / "MapData" / "arden_arcade_withtype_6to9am.rou.xml")
main_folder = str(_HERE / "Flood_MonteCarlo_withPolicy")

# --- SUMO network as NetworkX graph (build only once) ---
net = sumolib.net.readNet(network_file)
G_base = nx.DiGraph()
for edge in net.getEdges():
    if edge.getFunction() == "internal":
        continue
    edge_id = edge.getID()
    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()
    length = edge.getLength()  # meters
    speed = edge.getSpeed()    # m/s
    travel_time = length / speed if speed > 0 else 1e6
    G_base.add_edge(from_node, to_node, edge_id=edge_id, length=length, speed=speed, travel_time=travel_time)

# --- Extract OD pairs and hours from route file ---
vehicles = []
for veh in sumolib.xml.parse(route_file, 'vehicle'):
    veh_id = veh.id
    route_edges = veh.route[0].edges.split()
    start_edge = net.getEdge(route_edges[0])
    end_edge = net.getEdge(route_edges[-1])
    start_node = start_edge.getFromNode().getID()
    end_node = end_edge.getToNode().getID()
    depart_time = float(veh.depart)

    if 0 <= depart_time < 3600:
        hour = 1
    elif 3600 <= depart_time < 7200:
        hour = 2
    else:
        hour = 3

    vehicles.append({
        'veh_id': veh_id,
        'from': start_node,
        'to': end_node,
        'depart': depart_time,
        'hour': hour
    })
total_vehicles = len(vehicles)

# --- Recursively process each subfolder ---
for root, dirs, files in os.walk(main_folder):
    excel_files = [f for f in files if f.endswith('.xlsx')]
    if not excel_files:
        continue
    summary_file = os.path.join(root, "scenario_summary.csv")
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

        total_files = len(excel_files)
        for idx, fname in enumerate(sorted(excel_files)):
            fpath = os.path.join(root, fname)
            df = pd.read_excel(fpath)

            # Build edge status dictionaries for each hour
            edge_status = {1: {}, 2: {}, 3: {}}
            for _, row in df.iterrows():
                eid = str(row['Edge ID'])
                edge_status[1][eid] = row.get('Hour 1', 1)
                edge_status[2][eid] = row.get('Hour 2', 1)
                edge_status[3][eid] = row.get('Hour 3', 1)

            served_vehicle_results = []
            unserved_vehicles = []

            for veh in vehicles:
                hour = veh['hour']
                from_n = veh['from']
                to_n = veh['to']
                veh_id = veh['veh_id']
                depart_time = veh['depart']

                # --- Damaged network ---
                G_mod = G_base.copy()
                for u, v, data in list(G_mod.edges(data=True)):
                    eid = data['edge_id']
                    stat = edge_status[hour].get(eid, 1)
                    if stat == 0:
                        G_mod.remove_edge(u, v)
                    elif stat == 0.5:
                        G_mod[u][v]['travel_time'] = G_mod[u][v]['travel_time'] * 2

                try:
                    path = nx.shortest_path(G_mod, from_n, to_n, weight='travel_time')
                    time = 0.0
                    dist = 0.0
                    for i in range(len(path) - 1):
                        edge_data = G_mod[path[i]][path[i+1]]
                        time += edge_data['travel_time']
                        dist += edge_data['length']
                    served_vehicle_results.append((veh_id, from_n, to_n, hour, time, dist))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    unserved_vehicles.append((veh_id, from_n, to_n, depart_time))
                    continue

            # Baseline time/distance for served vehicles
            baseline_time = 0.0
            baseline_dist = 0.0
            for veh_id, from_n, to_n, hour, _, _ in served_vehicle_results:
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

            # Baseline distance for unserved vehicles
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

            # Diff from 3:00:00 for unserved vehicles (use depart time)
            diff_vals = []
            for _, _, _, depart_time in unserved_vehicles:
                diff_hr = (10800.0 - float(depart_time)) / 3600.0
                if diff_hr < 0:
                    diff_hr = 0
                if diff_hr > 3:
                    diff_hr = 3
                diff_vals.append(diff_hr)
            diff_hours_avg = (sum(diff_vals) / len(diff_vals)) if diff_vals else 0.0

            # Damaged network totals for served
            total_time = sum(x[4] for x in served_vehicle_results)
            total_dist = sum(x[5] for x in served_vehicle_results)
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

            # Additional loss
            additional_loss = unserved_baseline_dist_mi * 0.67 + diff_hours_avg * 30.12

            writer.writerow([
                fname, vehicles_served, unserved, percent_unserved,
                total_time_hr, baseline_time_hr, increase_time,
                total_dist_mi, baseline_dist_mi, increase_dist,
                service_loss,
                unserved_baseline_dist_mi, diff_hours_avg, additional_loss
            ])

            print(f"[{idx+1}/{total_files}] {os.path.relpath(fpath, main_folder)} -- "
                  f"Served: {vehicles_served}/{total_vehicles} | Unserved: {unserved} | "
                  f"Loss: ${service_loss:.2f} | Additional: ${additional_loss:.2f}")

print("\nAll subfolder summaries created.")
