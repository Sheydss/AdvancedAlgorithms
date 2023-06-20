"""
import algoRecuitSimule as algo
import random


def create_objects(max_entiers, max_valeur, start, end):
    if max_entiers > max_valeur - 2:
        raise ValueError("Le maximum d'entiers ne peut pas être supérieur à la différence entre max_valeur et 2")

    entiers = list(set(range(max_valeur)) - {start, end})
    entiers = random.sample(entiers, max_entiers)
    return entiers


if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 10
    max_edges_per_node = 5

    graph = algo.Graph(num_nodes, max_edges_per_node)

    start = 1
    end = 5

    # Générer la liste d'objets (sommets à visiter)
    obj = create_objects(2, num_nodes, start, end)

    # Détermine le meilleur chemin qui passe par les points générés
    path, cost, chemin_g = graph.best_itinerary(obj, start, end)
    graph.plot_graph([chemin_g], ['red'])

"""
from reportlab.lib.colors import red

import algoRecuitSimule as algo

if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 10000
    max_edges_per_node = 1

    graph = algo.Graph(num_nodes, max_edges_per_node)

    # Exemple d'utilisation
    cities = [
        1,3,5,9,45,59,500,30,70,12,500,674
    ]

    best_path, best_distance = algo.Graph.simulated_annealing(graph, 1000, cities)
    print("Meilleur chemin trouvé:", best_path)
    print("Distance totale:", best_distance)
    algo.Graph.plot_graph([best_path], ['red'])
