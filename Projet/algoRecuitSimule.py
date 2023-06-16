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

    def simulated_annealing(graph, start, goal):
        # Paramètres de l'algorithme
        initial_temperature = 100.0
        final_temperature = 0.1
        cooling_factor = 0.95
        max_iterations = 1000

        # Initialisation
        current_path = random.sample(list(graph.nodes()), len(graph.nodes()))
        current_path.remove(start)
        current_path.remove(goal)
        current_path.insert(0, start)
        current_path.append(goal)
        current_distance = calculate_total_distance(graph, current_path)
        best_path = current_path.copy()
        best_distance = current_distance
        temperature = initial_temperature
        iteration = 0

        # Boucle principale
        while temperature > final_temperature and iteration < max_iterations:
            neighbor = random_neighbor(current_path)
            neighbor_distance = calculate_total_distance(graph, neighbor)
            distance_diff = neighbor_distance - current_distance

            if distance_diff < 0 or random.random() < acceptance_probability(distance_diff, temperature):
                current_path = neighbor
                current_distance = neighbor_distance

                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance

            temperature = cooling_schedule(temperature, iteration)
            iteration += 1

        return best_path, best_distance


    """"
    def calculate_total_distance(path, cities):
        # Calcule la distance totale d'un chemin donné
        total_distance = 0
        for i in range(len(path)):
            city1 = cities[path[i]]
            city2 = cities[path[(i + 1) % len(path)]]
            total_distance += calculate_distance(city1, city2)
        return total_distance

    def random_neighbor(path):
        # Génère un voisin aléatoire à partir du chemin actuel
        neighbor = path.copy()
        index1 = random.randint(0, len(path) - 1)
        index2 = random.randint(0, len(path) - 1)
        neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]
        return neighbor

    def acceptance_probability(distance_diff, temperature):
        # Calcule la probabilité d'accepter un chemin moins optimal
        return math.exp(-distance_diff / temperature)

    def cooling_schedule(initial_temperature, iteration):
        # Schéma de refroidissement pour réduire la température au fil des itérations
        return initial_temperature * (0.95 ** iteration)

    def simulated_annealing(cities):
        # Paramètres de l'algorithme
        initial_temperature = 100.0
        final_temperature = 0.1
        max_iterations = 1000

        # Création d'un chemin initial aléatoire
        path = list(range(len(cities)))
        random.shuffle(path)

        # Initialisation
        best_path = path.copy()
        current_distance = calculate_total_distance(path, cities)
        best_distance = current_distance
        temperature = initial_temperature
        iteration = 0

        # Boucle principale
        while temperature > final_temperature and iteration < max_iterations:
            neighbor = random_neighbor(path)
            neighbor_distance = calculate_total_distance(neighbor, cities)
            distance_diff = neighbor_distance - current_distance

            if distance_diff < 0 or random.random() < acceptance_probability(distance_diff, temperature):
                path = neighbor
                current_distance = neighbor_distance

                if current_distance < best_distance:
                    best_path = path.copy()
                    best_distance = current_distance

            temperature = cooling_schedule(initial_temperature, iteration)
            iteration += 1

        return best_path, best_distance
    """

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
