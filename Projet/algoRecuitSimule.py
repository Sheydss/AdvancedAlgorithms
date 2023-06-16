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


    def initial_state(vertices):

        # Retourne une permutation aléatoire des sommets
        return random.sample(vertices, len(vertices))


    def calculate_distance(vertex1, vertex2):
        # Calcule la distance entre deux sommets donnés
        # vertex1, vertex2 : sommets
        # Retourne la distance entre les deux sommets
        pass


    def calculate_total_distance(path):
        # Calcule la distance totale d'un chemin donné
        # path : chemin (liste de sommets)
        # Retourne la distance totale du chemin
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += calculate_distance(path[i], path[i + 1])
        return total_distance


    def random_neighbor(path):
        # Génère un voisin aléatoire à partir du chemin actuel
        # path : chemin actuel (liste de sommets)
        # Retourne le voisin généré
        neighbor = path.copy()
        index1 = random.randint(0, len(path) - 1)
        index2 = random.randint(0, len(path) - 1)
        neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]
        return neighbor


    def acceptance_probability(distance_diff, temperature):
        # Calcule la probabilité d'accepter un chemin moins optimal
        # distance_diff : différence de distance entre le chemin actuel et le voisin
        # temperature : température actuelle
        # Retourne la probabilité d'acceptation
        return math.exp(-distance_diff / temperature)


    def cooling_schedule(initial_temperature, iteration):
        # Schéma de refroidissement pour réduire la température au fil des itérations
        # initial_temperature : température initiale
        # iteration : itération actuelle
        # Retourne la température mise à jour
        return initial_temperature * (0.95 ** iteration)


    def simulated_annealing(vertices):
        # Paramètres de l'algorithme
        initial_temperature = 100.0
        final_temperature = 0.1
        cooling_factor = 0.95
        max_iterations = 1000

        # Initialisation
        current_path = initial_state(vertices)
        best_path = current_path
        current_distance = calculate_total_distance(current_path)
        best_distance = current_distance
        temperature = initial_temperature
        iteration = 0

        # Boucle principale
        while temperature > final_temperature and iteration < max_iterations:
            neighbor = random_neighbor(current_path)
            neighbor_distance = calculate_total_distance(neighbor)
            distance_diff = neighbor_distance - current_distance

            if distance_diff < 0 or random.random() < acceptance_probability(distance_diff, temperature):
                current_path = neighbor
                current_distance = neighbor_distance

                if current_distance < best_distance:
                    best_path = current_path
                    best_distance = current_distance

            temperature = cooling_schedule(temperature, iteration)
            iteration += 1

        return best_path, best_distance
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
