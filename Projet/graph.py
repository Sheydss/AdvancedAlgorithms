import heapq
import random
import time

import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, num_nodes, max_edge):
        self.graph = nx.Graph()
        self.generate_random_graph(num_nodes, max_edge, 10, 100)

    def generate_random_graph(self, num_nodes, max_edges_per_node, min_weight, max_weight):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {num_nodes} nodes")
        tic = time.perf_counter()

        # Ajouter les nœuds au graphe
        self.graph.add_nodes_from(range(num_nodes))

        # Générer les arêtes avec des poids aléatoires
        for node in range(num_nodes):
            num_edges = random.randint(1, max_edges_per_node)
            dest_nodes = random.sample(range(num_nodes), num_edges)

            for dest_node in dest_nodes:
                weight = random.randint(min_weight, max_weight)
                if node != dest_node:
                    self.graph.add_edge(node, dest_node, weight=weight)
        toc = time.perf_counter()
        print(f"Generation done in {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

    def dijkstra(self, start, end):
        print("---------------------------------------------------------")
        print(f"Dijkstra : From {start} to {end}")
        tic = time.perf_counter()

        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0

        heap = [(0, start)]
        while heap:
            current_cost, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            if current_cost > distances[current_node]:
                continue

            for neighbor in self.graph.neighbors(current_node):
                cost = self.graph[current_node][neighbor]['weight']
                new_cost = current_cost + cost
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor))

        if distances[end] == float('inf'):
            return None, None

        path = self.reconstruct_path(start, end, distances)
        toc = time.perf_counter()

        print(f"Path : {path} Cost : {distances[end]}")
        print(f"Duration : {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

        return path, distances[end]

    def reconstruct_path(self, start, end, distances):
        current_node = end
        path = [current_node]
        while current_node != start:
            neighbors = self.graph.neighbors(current_node)
            min_neighbor = None
            min_cost = float('inf')
            for neighbor in neighbors:
                cost = self.graph[current_node][neighbor]['weight']
                if distances[neighbor] + cost < min_cost:
                    min_neighbor = neighbor
                    min_cost = distances[neighbor] + cost
            path.append(min_neighbor)
            current_node = min_neighbor
        path.reverse()
        return path

    def print_dijkstra(self, path):
        for i in path:
            list_values = []
            print(f"Node : {i}")
            for edge in self.graph.neighbors(i):
                values = dict()
                values[edge] = self.graph[i][edge]['weight']
                list_values.append(values)
            print(list_values)

    def data_graph(self):
        for node in self.graph.nodes:
            list_values = []
            print(f"Node : {node}")
            for edge in self.graph.neighbors(node):
                values = dict()
                values[edge] = self.graph[node][edge]['weight']
                list_values.append(values)
            print(list_values)

    def plot_graph(self):
        plt.show()

    def color_path(self, path, color):
        # Obtenir les arêtes du chemin
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        # Obtenir les positions des nœuds pour le tracé
        pos = nx.spring_layout(self.graph)

        # Dessiner le graphe avec les arêtes du chemin coloriées
        nx.draw(self.graph, pos=pos, width=0.5, with_labels=True)
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edges, edge_color=color)

        plt.show()






