import heapq
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self, num_nodes, max_edge):
        self.graph = nx.Graph()
        self.generate_random_graph(num_nodes, max_edge, 10, 100)

    def generate_random_graph(self, num_nodes, max_edges_per_node, min_weight, max_weight):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {num_nodes} nodes")
        tic = time.perf_counter()

        for node in range(num_nodes):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.graph.add_node(node, pos=(x, y))

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

    def get_edge_weight(self, source, destination):
        try:
            return self.graph[source][destination]['weight']
        except KeyError:
            return float('inf')

    def calculate_path_distance(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            source = path[i]
            destination = path[i + 1]
            weight = self.get_edge_weight(source, destination)
            total_distance += weight
        return total_distance

    def simulated_annealing(self, start, goals):
        initial_temperature = 1000.0
        final_temperature = 0.1
        cooling_factor = 0.995
        max_iterations = 1000

        current_path = [start] + random.sample(goals, len(goals))
        current_distance = self.calculate_path_distance(current_path)
        best_path = current_path.copy()
        best_distance = current_distance
        temperature = initial_temperature
        iteration = 0

        while temperature > final_temperature and iteration < max_iterations:
            neighbor = self.get_random_neighbor(current_path)
            neighbor_distance = self.calculate_path_distance(neighbor)
            distance_diff = neighbor_distance - current_distance

            if distance_diff < 0 or random.random() < math.exp(-distance_diff / temperature):
                current_path = neighbor
                current_distance = neighbor_distance

                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance

            temperature *= cooling_factor
            iteration += 1

        return best_path, best_distance

    def get_random_neighbor(self, path):
        random_index = random.randint(1, len(path) - 2)
        neighbor = path.copy()
        random_neighbor_index = random.randint(1, len(path) - 2)
        neighbor[random_index], neighbor[random_neighbor_index] = neighbor[random_neighbor_index], neighbor[random_index]
        return neighbor


    def plot_graph(self, path=None, color='black'):
        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        pos = nx.get_node_attributes(self.graph, 'pos')

        nx.draw(self.graph, pos=pos, width=0.8, with_labels=True)
        if path is None:
            nx.draw_networkx_edges(self.graph, pos=pos, edge_color=color)
        else:
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edges, edge_color=color)

        plt.show()
