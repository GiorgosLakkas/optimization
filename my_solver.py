import math
import sys
import time
import random
import copy
import os

INPUT_FILE = 'instance.txt'
OUTPUT_FILE = 'solution.txt'
TIME_LIMIT = 230  # Τρέχει για περίπου 4 λεπτά

# --- HELPER FUNCTIONS ---

def calculate_route_cost(route, cost_matrix):
    cost = 0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i+1]
        cost += cost_matrix[u][v]
    return cost

def calculate_total_cost(routes, cost_matrix):
    return sum(calculate_route_cost(r, cost_matrix) for r in routes)

def get_route_load(route, node_demands):
    return sum(node_demands[n] for n in route if n != 0)

def save_solution(routes, cost_matrix, filename):
    """Saves the solution to disk immediately."""
    total_cost = 0
    output_lines = []
    for r in routes:
        c = calculate_route_cost(r, cost_matrix)
        total_cost += c
        route_str = "->".join(map(str, r))
        output_lines.append(f"Vehicle cost: {c:.2f} Route: {route_str}")
    output_lines.append(f"Total Cost: {total_cost:.2f}")

    with open(filename, 'w') as f:
        f.write("\n".join(output_lines) + "\n")
        f.flush()
        os.fsync(f.fileno()) # Force write to disk

# --- LOCAL SEARCH OPERATORS ---

def two_opt_optimization(route, cost_matrix):
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1: continue
                A, B = best_route[i-1], best_route[i]
                C, D = best_route[j], best_route[j+1]
                
                old_cost = cost_matrix[A][B] + cost_matrix[C][D]
                new_cost = cost_matrix[A][C] + cost_matrix[B][D]
                
                if new_cost < old_cost - 0.001:
                    best_route[i:j+1] = best_route[i:j+1][::-1]
                    improved = True
    return best_route

def attempt_relocate(routes, cost_matrix, node_demands, capacity):
    for r_src_idx in range(len(routes)):
        for i in range(1, len(routes[r_src_idx]) - 1):
            node = routes[r_src_idx][i]
            demand = node_demands[node]
            
            prev_src = routes[r_src_idx][i-1]
            next_src = routes[r_src_idx][i+1]
            cost_removed = cost_matrix[prev_src][node] + cost_matrix[node][next_src] - cost_matrix[prev_src][next_src]
            
            best_target_route = -1
            best_target_pos = -1
            best_gain = 0
            
            for r_dst_idx in range(len(routes)):
                if r_src_idx == r_dst_idx: continue
                target_route = routes[r_dst_idx]
                if get_route_load(target_route, node_demands) + demand > capacity: continue
                
                for j in range(len(target_route) - 1):
                    A = target_route[j]
                    B = target_route[j+1]
                    cost_added = cost_matrix[A][node] + cost_matrix[node][B] - cost_matrix[A][B]
                    gain = cost_removed - cost_added
                    if gain > 0.001 and gain > best_gain:
                        best_gain = gain
                        best_target_route = r_dst_idx
                        best_target_pos = j + 1
            
            if best_target_route != -1:
                routes[r_src_idx].pop(i)
                routes[best_target_route].insert(best_target_pos, node)
                return True
    return False

def attempt_swap(routes, cost_matrix, node_demands, capacity):
    for r1_idx in range(len(routes)):
        for i in range(1, len(routes[r1_idx]) - 1):
            u = routes[r1_idx][i]
            u_demand = node_demands[u]
            u_prev, u_next = routes[r1_idx][i-1], routes[r1_idx][i+1]
            cost_u_rem = cost_matrix[u_prev][u] + cost_matrix[u][u_next] - cost_matrix[u_prev][u_next]
            
            for r2_idx in range(len(routes)):
                if r1_idx == r2_idx: continue
                load1 = get_route_load(routes[r1_idx], node_demands)
                load2 = get_route_load(routes[r2_idx], node_demands)
                
                for j in range(1, len(routes[r2_idx]) - 1):
                    v = routes[r2_idx][j]
                    v_demand = node_demands[v]
                    
                    if load1 - u_demand + v_demand > capacity: continue
                    if load2 - v_demand + u_demand > capacity: continue
                    
                    v_prev, v_next = routes[r2_idx][j-1], routes[r2_idx][j+1]
                    cost_v_rem = cost_matrix[v_prev][v] + cost_matrix[v][v_next] - cost_matrix[v_prev][v_next]
                    
                    cost_v_add = cost_matrix[u_prev][v] + cost_matrix[v][u_next] - cost_matrix[u_prev][u_next]
                    cost_u_add = cost_matrix[v_prev][u] + cost_matrix[u][v_next] - cost_matrix[v_prev][v_next]
                    
                    gain = (cost_u_rem + cost_v_rem) - (cost_u_add + cost_v_add)
                    if gain > 0.001:
                        routes[r1_idx][i] = v
                        routes[r2_idx][j] = u
                        return True
    return False

def try_item_swaps(routes, families, cost_matrix, node_to_family):
    used_nodes = set()
    for r in routes: used_nodes.update(r)

    for r in routes:
        for i in range(1, len(r) - 1):
            curr = r[i]
            prev = r[i-1]
            nxt = r[i+1]
            fam_id = node_to_family.get(curr)
            if fam_id is None: continue

            curr_cost = cost_matrix[prev][curr] + cost_matrix[curr][nxt]
            best_cand = -1
            best_diff = 0
            
            for cand in families[fam_id]['items']:
                if cand in used_nodes: continue
                new_cost = cost_matrix[prev][cand] + cost_matrix[cand][nxt]
                diff = new_cost - curr_cost
                if diff < -0.001 and diff < best_diff:
                    best_diff = diff
                    best_cand = cand
            
            if best_cand != -1:
                r[i] = best_cand
                used_nodes.remove(curr)
                used_nodes.add(best_cand)
                return True
    return False

def perturb_solution(routes, families, cost_matrix, node_to_family):
    """ Kicks the solution to escape local optima. """
    if len(routes) > 1:
        for _ in range(2): 
            r1_idx = random.randint(0, len(routes)-1)
            r2_idx = random.randint(0, len(routes)-1)
            if r1_idx == r2_idx or len(routes[r1_idx])<3 or len(routes[r2_idx])<3: continue
            idx1 = random.randint(1, len(routes[r1_idx])-2)
            idx2 = random.randint(1, len(routes[r2_idx])-2)
            u, v = routes[r1_idx][idx1], routes[r2_idx][idx2]
            # Force swap without check
            routes[r1_idx][idx1], routes[r2_idx][idx2] = v, u

    for _ in range(3):
        r_idx = random.randint(0, len(routes)-1)
        if len(routes[r_idx]) < 3: continue
        node_idx = random.randint(1, len(routes[r_idx])-2)
        node = routes[r_idx][node_idx]
        fam_id = node_to_family.get(node)
        if fam_id is not None:
            candidates = families[fam_id]['items']
            random_cand = random.choice(candidates)
            used = False
            for r in routes:
                if random_cand in r: used = True
            if not used:
                routes[r_idx][node_idx] = random_cand

# --- MAIN SOLVER ---

def solve():
    print(f"--- Reading {INPUT_FILE} ---")
    start_time = time.time()
    
    with open(INPUT_FILE, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Parsing
    meta = list(map(int, lines[0].split()))
    n_items, n_families, _, capacity, n_vehicles = meta
    items_per_family = list(map(int, lines[1].split()))
    reqs_per_family = list(map(int, lines[2].split()))
    demands_per_family = list(map(int, lines[3].split()))

    cost_matrix = []
    for i in range(4, len(lines)):
        row = list(map(float, lines[i].split()))
        cost_matrix.append(row)

    families = [] 
    node_to_family = {}
    node_demands = {}
    current_id = 1 
    for f_idx in range(n_families):
        count = items_per_family[f_idx]
        ids = []
        d = demands_per_family[f_idx]
        for _ in range(count):
            ids.append(current_id)
            node_to_family[current_id] = f_idx
            node_demands[current_id] = d
            current_id += 1
        families.append({'items': ids, 'req': reqs_per_family[f_idx]})

    # --- INITIAL CONSTRUCTION ---
    print("Constructing Initial Greedy Solution...")
    final_nodes_pool = set()
    for f_idx, fam in enumerate(families):
        sorted_by_depot = sorted(fam['items'], key=lambda n: cost_matrix[0][n])
        selected = sorted_by_depot[:fam['req']]
        final_nodes_pool.update(selected)

    routes = []
    unvisited = list(final_nodes_pool)
    unvisited_set = set(unvisited)
    
    for _ in range(n_vehicles):
        if not unvisited_set: break
        r = [0]
        load = 0
        curr = 0
        while True:
            best = -1
            min_d = float('inf')
            for cand in unvisited_set:
                if load + node_demands[cand] <= capacity:
                    dist = cost_matrix[curr][cand]
                    if dist < min_d:
                        min_d = dist
                        best = cand
            if best != -1:
                r.append(best)
                load += node_demands[best]
                curr = best
                unvisited_set.remove(best)
            else: break
        r.append(0)
        if len(r) > 2: routes.append(r)

    global_best_routes = copy.deepcopy(routes)
    global_best_cost = calculate_total_cost(routes, cost_matrix)
    print(f"Initial Cost: {global_best_cost:.2f}")
    
    # Save initial just in case
    save_solution(global_best_routes, cost_matrix, OUTPUT_FILE)

    # --- ITERATED LOCAL SEARCH LOOP ---
    print(f"Starting ILS for {TIME_LIMIT} seconds...")
    
    while time.time() - start_time < TIME_LIMIT:
        
        improvement = True
        while improvement:
            improvement = False
            if attempt_relocate(routes, cost_matrix, node_demands, capacity): improvement = True
            elif attempt_swap(routes, cost_matrix, node_demands, capacity): improvement = True
            elif try_item_swaps(routes, families, cost_matrix, node_to_family): improvement = True
            
            for i in range(len(routes)):
                old_c = calculate_route_cost(routes[i], cost_matrix)
                routes[i] = two_opt_optimization(routes[i], cost_matrix)
                if calculate_route_cost(routes[i], cost_matrix) < old_c - 0.001:
                    improvement = True
            
            if time.time() - start_time > TIME_LIMIT: break

        current_cost = calculate_total_cost(routes, cost_matrix)
        if current_cost < global_best_cost - 0.001:
            global_best_cost = current_cost
            global_best_routes = copy.deepcopy(routes)
            print(f"New Best Found: {global_best_cost:.2f} (Saved to disk)")
            # SAVE IMMEDIATELY
            save_solution(global_best_routes, cost_matrix, OUTPUT_FILE)

        if time.time() - start_time < TIME_LIMIT:
            perturb_solution(routes, families, cost_matrix, node_to_family)
    
    print(f"Time limit reached. Final Best Cost: {global_best_cost:.2f}")

if __name__ == "__main__":
    solve()