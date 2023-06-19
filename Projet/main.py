import graph as g
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

    graph = g.Graph(num_nodes, max_edges_per_node)

    start = 1
    end = 5

    # Générer la liste d'objets (sommets à visiter)
    obj = create_objects(2, num_nodes, start, end)

    # Détermine le meilleur chemin qui passe par les points générés
    path, cost, chemin_g = graph.best_itinerary(obj, start, end)
    graph.plot_graph([chemin_g], ['red'])
