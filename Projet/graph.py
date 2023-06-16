import heapq
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx


class Graph2:
    def __init__(self, num_nodes, max_edge):
        self.graph = nx.Graph()
        self.generate_random_graph(num_nodes, max_edge, 10, 100)

    def generate_random_graph(self, num_nodes, max_edges_per_node, min_weight, max_weight):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {num_nodes} nodes")
        tic = time.perf_counter()

        # Ajouter les nœuds au graphe avec des coordonnées aléatoires
        for node in range(num_nodes):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.graph.add_node(node, pos=(x, y))

        # Générer les arêtes avec des poids correspondant à la distance euclidienne
        for node in range(num_nodes):
            num_edges = random.randint(1, max_edges_per_node)
            dest_nodes = random.sample(range(num_nodes), num_edges)

            for dest_node in dest_nodes:
                if node != dest_node:
                    pos1 = self.graph.nodes[node]['pos']
                    pos2 = self.graph.nodes[dest_node]['pos']
                    distance = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                    weight = int(distance)
                    self.graph.add_edge(node, dest_node, weight=weight)

        toc = time.perf_counter()
        print(f"Generation done in {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

    def dijkstra(self, start, end):
        print("---------------------------------------------------------")
        print(f"Dijkstra Search: From {start} to {end}")
        tic = time.perf_counter()

        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0

        heap = [(0, start)]
        while heap:
            current_cost, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            for neighbor in self.graph.neighbors(current_node):
                cost = self.graph[current_node][neighbor]['weight']
                new_cost = current_cost + cost
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor))

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

    def plot_graph(self, path=None, color='black'):
        # Obtenir les arêtes du chemin
        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        # Obtenir les positions des nœuds pour le tracé
        pos = nx.get_node_attributes(self.graph, 'pos')

        # Dessiner le graphe avec les arêtes du chemin coloriées
        nx.draw(self.graph, pos=pos, width=0.8, with_labels=True)
        if path is None:
            nx.draw_networkx_edges(self.graph, pos=pos, edge_color=color)
        else:
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edges, edge_color=color)

        plt.show()

    def a_star(self, start, end):
        print("---------------------------------------------------------")
        print(f"A* Search: From {start} to {end}")
        tic = time.perf_counter()

        # Distances du point de départ à chaque nœud (g-cost)
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0

        # Estimations des coûts du point de départ à chaque nœud (f-cost)
        estimations = {node: self.heuristic(node, end) for node in self.graph.nodes}

        # File de priorité pour l'exploration des nœuds
        heap = [(estimations[start], 0, start)]  # (f-cost, g-cost, node)

        # Dictionnaire pour stocker les nœuds précédents sur le chemin optimal
        came_from = {}

        while heap:
            _, current_cost, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            for neighbor in self.graph.neighbors(current_node):
                cost = self.graph[current_node][neighbor]['weight']
                new_cost = current_cost + cost
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, end)
                    heapq.heappush(heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current_node

        path = self.reverse_path(start, end, came_from)
        toc = time.perf_counter()

        print(f"Path: {path} Cost: {distances[end]}")
        print(f"Duration: {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

        return path, distances[end]

    def heuristic(self, node, goal):
        # Heuristique utilisée pour estimer le coût restant (ici, distance euclidienne)
        pos1 = self.graph.nodes[node]['pos']
        pos2 = self.graph.nodes[goal]['pos']
        distance = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
        return distance

    def reverse_path(self, start, end, came_from):
        current_node = end
        path = [current_node]
        while current_node != start:
            current_node = came_from[current_node]
            path.append(current_node)
        path.reverse()
        return path
