import random
import time
import heapq


class Graph:
    def __init__(self, num_nodes, max_edge):
        self.graph = dict()
        self.generate_graph(num_nodes, max_edge)

    def add_edge(self, node1, node2, cost, max_edge):
        if node1 not in self.graph:
            self.graph[node1] = dict()
        if node2 not in self.graph:
            self.graph[node2] = dict()

        if node1 != node2 and len(self.graph[node1]) < max_edge:
            self.graph[node1][node2] = int(cost)
            self.graph[node2][node1] = int(cost)

    def are_cities_neighbors(self, city1, city2):
        if city1 in self.graph and city2 in self.graph:
            neighbors = list(self.graph[city1].keys())
            if city2 in neighbors:
                return True, self.graph[city1][city2]
        return False, None

    def generate_graph(self, node_count, max_edge):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {node_count} nodes")

        tic = time.perf_counter()
        num_edges = node_count * (node_count - 1) // 2  # Nombre d'arêtes restantes pour un graphe connexe

        for _ in range(num_edges):
            node1 = random.randint(0, node_count - 1)
            node2 = random.randint(0, node_count - 1)
            cost = random.randint(10, 100)  # Coût aléatoire entre 10 et 100 (minutes)
            self.add_edge(node1, node2, cost, max_edge)

        toc = time.perf_counter()
        print(f"Generation done in {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

        def generate_graph(self, node_count, max_edges_per_node):
            print("---------------------------------------------------------")
            print(f"Generating Graph with {node_count} nodes")

            tic = time.perf_counter()

            available_nodes = set(range(node_count))
            selected_nodes = set()

            start_node = random.choice(tuple(available_nodes))
            selected_nodes.add(start_node)
            available_nodes.remove(start_node)

            while available_nodes:
                node = random.choice(tuple(selected_nodes))
                num_edges = min(max_edges_per_node, len(available_nodes))
                if num_edges == 0:
                    break

                for _ in range(num_edges):
                    adjacent_node = random.choice(tuple(available_nodes))
                    cost = random.randint(10, 100)  # Coût aléatoire entre 10 et 100 (minutes)
                    self.add_edge(node, adjacent_node, cost)
                    selected_nodes.add(adjacent_node)
                    available_nodes.remove(adjacent_node)

            toc = time.perf_counter()
            print(f"Generation done in {toc - tic:0.4f} seconds")
            print("---------------------------------------------------------")

    def print_graph(self):
        for key in self.graph:
            print(f"{key} --> {self.graph[key]}")

    def dijkstra(self, start, end):
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0

        heap = [(0, start)]
        while heap:
            current_cost, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            if current_cost > distances[current_node]:
                continue

            for neighbor, cost in self.graph[current_node].items():
                new_cost = current_cost + cost
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor))

        if distances[end] == float('inf'):
            return None, None

        path = self.reconstruct_path(start, end, distances)
        return path, distances[end]

    def reconstruct_path(self, start, end, distances):
        current_node = end
        path = [current_node]
        while current_node != start:
            neighbors = self.graph[current_node]
            min_neighbor = min(neighbors, key=lambda node: distances[node] + neighbors[node])
            path.append(min_neighbor)
            current_node = min_neighbor
        path.reverse()
        return path


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
