import random
import time


class Graph:
    def __init__(self, num_nodes):
        self.graph = dict()
        self.edges = dict()
        self.generate_graph(num_nodes)

    def add_edge(self, node1, node2, cost):
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []

        self.graph[node1].append((node2, int(cost)))

        if node1 not in self.edges:
            self.edges[node1] = dict()
        if node2 not in self.edges:
            self.edges[node2] = dict()

        self.edges[node1][node2] = int(cost)
        self.edges[node2][node1] = int(cost)

    def print_graph(self):
        for source, destination in self.graph.items():
            destination_nodes = [(node, cost) for node, cost in destination]
            print(f"{source} --> {destination_nodes}")

    def get_node_info_by_id(self, node_id):
        if node_id in self.edges:
            edges = self.edges[node_id]
            return f"Node ID: {node_id}\nEdges: {edges}"
        return "Node not found."

    # Fonction qui renvoie un boolean et la valeur des edges ou non
    def are_cities_neighbors(self, city1, city2):
        if city1 in self.graph and city2 in self.graph:
            neighbors = [neighbor for neighbor, _ in self.graph[city1]]
            if city2 in neighbors:
                return True, self.edges[city1][city2]
        return False, None

    def generate_graph(self, node_count):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {node_count} nodes")
        tic = time.perf_counter()
        num_edges = node_count * (node_count - 1) // 2  # Nombre d'arêtes pour un graphe connexe

        for _ in range(num_edges):
            node1 = random.randint(0, node_count - 1)  # Générer un identifiant de nœud aléatoire
            node2 = random.randint(0, node_count - 1)
            cost = random.randint(1, 100)  # Coût aléatoire entre 1 et 100 (minutes)
            self.add_edge(node1, node2, cost)
        toc = time.perf_counter()
        print(f"Generation done in {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")


"""
    g = Graph(3000)

    # Voir si deux nodes sont voisins
    city1 = 1000  # ID ou nom de la première ville
    city2 = 1001  # ID ou nom de la deuxième ville

    are_neighbors, edges = g.are_cities_neighbors(city1, city2)
    if are_neighbors:
        print(f"{city1} and {city2} are neighbors.")
        print(f"Edges between {city1} and {city2}: {edges}")
    else:
        print(f"{city1} and {city2} are not neighbors.")

"""
