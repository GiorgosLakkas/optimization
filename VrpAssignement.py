def read_input_file(path):
    with open(path, "r") as f:
        
        line1 = f.readline().strip().split()
        N, F, V, capacity, trucks = map(int, line1)

        
        family_sizes = list(map(int, f.readline().strip().split()))
        # You expect 90 integers here, or exactly F integers depending on the spec
        assert len(family_sizes) == F, "Family size count mismatch"

       
        visit_demands = list(map(int, f.readline().strip().split()))
        assert len(visit_demands) == F, "Visit Demand count mismatch"
        
        family_volume = list(map(int, f.readline().strip().split()))
        assert len(family_volume) == F, "Family cost count mismatch"
        
       
        cost_matrix = []
        for _ in range(N + 1):
            row = list(map(int, f.readline().strip().split()))
            assert len(row) == (N + 1), "Cost matrix row length mismatch"
            cost_matrix.append(row)

    
    return {
        "N": N,
        "F": F,
        "visits": V,  #Total visits required
        "capacity": capacity,
        "trucks": trucks,
        "family_sizes": family_sizes, #The line that reoresents how many nodes each family holds
        "visit_demands": visit_demands, #The line that represents how many visits should take place for each family
        "family_volume": family_volume, #The line that represents how much capacity does a visit take from the truck
        "cost_matrix": cost_matrix #The costs from a point to another
    }

data = read_input_file("instance.txt")

    
def build_model (data):
    
    N = data["N"]
    F = data["F"]
    family_sizes = data["family_sizes"]
    visit_demands = data["visit_demands"]
    family_volume = data["family_volume"]
    cost_matrix = data["cost_matrix"]
    t = data["trucks"]
    capacity = data["capacity"]
    
    allNodes = []
    allFamilies = []
    allTrucks= []
    
    
    for i in range(F):
        fam = Family(i+1, family_volume[i], visit_demands[i])
        allFamilies.append(fam)
    
    
    
    allNodes.append(Node(0, None, cost_matrix[0], None))
    
    fam_id = 1
    node_id = 1
    
    for family_size in family_sizes:
        for _ in range(family_size):
            node = Node(node_id, fam_id, cost_matrix[node_id], allFamilies[fam_id - 1].demand)

            allNodes.append(node)
            node_id += 1
        fam_id += 1
        
    for i in range(t):
        truck = Truck(i+1, capacity, [])
        allTrucks.append(truck)
        
        
    return allFamilies, allNodes, allTrucks

    

class Node:
    def __init__(self, node_id, family, cost, demand):
        self.id = node_id
        self.family = family
        self.cost = cost
        self.demand = demand
        self.is_depot = False
        self.is_routed = False
        self.route = None


class Family:
    def __init__(self, family_id, demand, required):
        self.id = family_id
        self.demand = demand
        self.required = required


class Truck:
    def __init__(self, truck_id, remaining, nodes_visited):
        self.id = truck_id
        self.remaining = remaining
        self.nodes = nodes_visited



m, v, n = build_model(data)