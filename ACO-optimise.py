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

        # Ajouter les nœuds au graphe avec des coordonnées aléatoires
        for node in range(num_nodes):
            x = random.uniform(min_weight, max_weight)
            y = random.uniform(min_weight, max_weight)
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
        # Obtenir les positions des nœuds pour le tracé
        pos = nx.get_node_attributes(self.graph, 'pos')

        # Dessiner le graphe avec les arêtes des chemins coloriées
        nx.draw(self.graph, pos=pos, width=0.8, with_labels=True)

        if paths is not None:
            if colors is None:
                colors = ['black'] * len(paths)  # Par défaut, utiliser la couleur noire pour tous les chemins
            elif len(colors) < len(paths):
                raise ValueError("Le nombre de couleurs fourni est inférieur au nombre de chemins.")

            for path, color in zip(paths, colors):
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edges, edge_color=color, width=2)

        plt.show()


def distance(point1, point2):
    pos1 = graph.graph.nodes[point1]['pos']
    pos2 = graph.graph.nodes[point2]['pos']
    return int(math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2))


def initialiser_pheromones(n):
    return np.ones((n, n))


def mettre_a_jour_pheromones(pheromones, chemin, cout_total, alpha, beta):
    n = pheromones.shape[0]
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


def colonie_de_fourmis(points, nombre_fourmis, alpha, beta, evaporation, iterations):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = distance(points[i], points[j])

    pheromones = initialiser_pheromones(n)
    meilleur_chemin = None
    meilleur_cout = float('inf')

    start_time = time.time()

    for _ in range(iterations):
        for _ in range(nombre_fourmis):
            visites = [False] * n
            chemin = []
            cout_total = 0
            ville_courante = np.random.randint(n)
            visites[ville_courante] = True
            chemin.append(ville_courante)

            for _ in range(n - 1):
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


# Exemple d'utilisation
num_nodes = 5
max_edge = 3

graph = Graph(num_nodes, max_edge)

points = list(range(num_nodes))
nombre_fourmis = 10
alpha = 1.0
beta = 2.0
evaporation = 0.5
iterations = 100

meilleur_chemin, meilleur_cout, temps_execution = colonie_de_fourmis(points, nombre_fourmis, alpha, beta, evaporation, iterations)

print("Meilleur chemin:", meilleur_chemin)
print("Meilleur coût:", meilleur_cout)
print("Temps d'exécution:", temps_execution, "secondes")

# Afficher le graphe avec le meilleur chemin
graph.plot_graph([meilleur_chemin])
