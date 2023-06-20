#import graph as g
#
#if __name__ == '__main__':
#    # Exemple d'utilisation
#    num_nodes = 10
#    max_edges_per_node = 5
#
#    graph = g.Graph(num_nodes, max_edges_per_node)
#
#    start = 1
#    end = 5
#
#    graph.a_star(start, end)
#    path, distance = graph.dijkstra(start, end)
#    path2, distance2 = graph.recherche_tabou(start, end, 20, 200)
#    graph.plot_graph(paths=[path, path2], colors=['red', 'green'])

#start colonie fourmie
#import colonie_fourmie as aco
#import random

# Nombre de nœuds et nombre maximum d'arêtes par nœud
#num_nodes = 10
#max_edge = 5

# Générer un graphe aléatoire
#graph = aco.Graph(num_nodes, max_edge)

#start and end points
#start = 0
#end = 9

# Extraire les points du graphe
#points = list(range(num_nodes))

# Paramètres de l'algorithme de la colonie de fourmis
#nombre_fourmis = 10
#alpha = 1.0
#beta = 2.0
#evaporation = 0.5
#iterations = 100

# Exécuter l'algorithme de la colonie de fourmis
#meilleur_chemin, meilleur_cout, temps_execution = aco.colonie_de_fourmis(graph, points, nombre_fourmis, alpha, beta, evaporation, iterations, start, end)


# Afficher les résultats
#print("Meilleur chemin:", meilleur_chemin)
#print("Meilleur coût:", meilleur_cout)
#print("Temps d'exécution:", temps_execution, "secondes")

# Afficher le graphe avec le meilleur chemin
#graph.plot_graph([meilleur_chemin])
#fin


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

    graph = Graph2(num_nodes, max_edges_per_node)

    start = 1
    end = 9

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




