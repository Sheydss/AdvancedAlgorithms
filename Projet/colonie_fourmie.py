#ChatGPT
import numpy as np
import time
import random
import math
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

        for node in range(num_nodes):
            x = random.uniform(min_weight, max_weight)
            y = random.uniform(min_weight, max_weight)
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

    def data_graph(self):
        for node in self.graph.nodes:
            list_values = []
            print(f"Node : {node}")
            for edge in self.graph.neighbors(node):
                values = dict()
                values[edge] = self.graph[node][edge]['weight']
                list_values.append(values)
            print(list_values)

    def plot_graph(self, paths=None, colors=None):
        pos = nx.get_node_attributes(self.graph, 'pos')

        nx.draw(self.graph, pos=pos, width=0.8, with_labels=True)

        if paths is not None:
            if colors is None:
                colors = ['black'] * len(paths)
            elif len(colors) < len(paths):
                raise ValueError("the number of color is inferior to the number of path")

            for path, color in zip(paths, colors):
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edges, edge_color=color, width=2)

        plt.show()


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def initialiser_pheromones(n):
    return np.ones((n, n))


def mettre_a_jour_pheromones(pheromones, chemin, cout_total, alpha, beta):
    n = pheromones.shape[0]
    if len(chemin) == n:
        for i in range(n):
            ville_courante = chemin[i]
            ville_suivante = chemin[(i + 1) % n]
            pheromones[ville_courante, ville_suivante] = (1 - alpha) * pheromones[ville_courante, ville_suivante] + alpha / cout_total
            pheromones[ville_suivante, ville_courante] = pheromones[ville_courante, ville_suivante]




def choisir_prochaine_ville(pheromones, visites, ville_courante, alpha, beta):
    n = pheromones.shape[0]
    max_phéromone = -1
    ville_suivante = -1
    for ville in range(n):
        if not visites[ville]:
            pheromone = pheromones[ville_courante, ville] * (1.0 / distance(ville_courante, ville) ** beta)
            if pheromone > max_phéromone:
                max_phéromone = pheromone
                ville_suivante = ville
    return ville_suivante


def colonie_de_fourmis(graph, points, nombre_fourmis, alpha, beta, evaporation, iterations, point_a, point_b):
    # ...
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = distance(points[i], points[j])

    indice_a = points.index(point_a)
    indice_b = points.index(point_b)

    pheromones = initialiser_pheromones(n)
    meilleur_chemin = None
    meilleur_cout = float('inf')

    start_time = time.time()

    for _ in range(iterations):
        for _ in range(nombre_fourmis):
            visites = [False] * n
            chemin = []
            cout_total = 0
            ville_courante = indice_a
            visites[ville_courante] = True
            chemin.append(ville_courante)

            while ville_courante != indice_b:
                ville_suivante = choisir_prochaine_ville(pheromones, visites, ville_courante, alpha, beta)
                visites[ville_suivante] = True
                chemin.append(ville_suivante)
                cout_total += distances[ville_courante, ville_suivante]
                ville_courante = ville_suivante

            cout_total += distances[ville_courante, chemin[0]]

            if cout_total < meilleur_cout:
                meilleur_chemin = chemin
                meilleur_cout = cout_total

            mettre_a_jour_pheromones(pheromones, chemin, cout_total, evaporation, 1)

    execution_time = time.time() - start_time

    return meilleur_chemin, int(meilleur_cout), execution_time
