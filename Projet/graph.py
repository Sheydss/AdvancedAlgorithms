import random
import time


class Graph:
    def __init__(self, num_nodes):
        self.graph = dict()
        self.generate_graph(num_nodes)

    def add_edge(self, node1, node2, cost):
        if node1 not in self.graph:
            self.graph[node1] = dict()
        if node2 not in self.graph:
            self.graph[node2] = dict()

        if node1 != node2:
            self.graph[node1][node2] = int(cost)
            self.graph[node2][node1] = int(cost)

    def are_cities_neighbors(self, city1, city2):
        if city1 in self.graph and city2 in self.graph:
            neighbors = list(self.graph[city1].keys())
            if city2 in neighbors:
                return True, self.graph[city1][city2]
        return False, None

    def generate_graph(self, node_count):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {node_count} nodes")

        num_edges = node_count * (node_count - 1) // 2  # Nombre d'arêtes pour un graphe connexe

        tic = time.perf_counter()
        for _ in range(num_edges):
            node1 = random.randint(0, node_count - 1)  # Générer un identifiant de nœud aléatoire
            node2 = random.randint(0, node_count - 1)
            cost = random.randint(10, 100)  # Coût aléatoire entre 10 et 100 (minutes)
            self.add_edge(node1, node2, cost)
        toc = time.perf_counter()
        print(f"Generation done in {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

    def print_graph(self):
        for key in self.graph:
            print(f"{key} --> {self.graph[key]}")


"""

    print(g.graph, len(g.graph))
    count = 0
    for key in g.graph:
        print(f" Clé : {key}")
        for ed in g.graph[key]:
            count += 1
            print(f" Valeur : {ed}")

    print(count)
    
    
 if node1 not in self.edges:
            self.edges[node1] = dict()
        if node2 not in self.edges:
            self.edges[node2] = dict()

        self.edges[node1][node2] = int(cost)
        self.edges[node2][node1] = int(cost)

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
