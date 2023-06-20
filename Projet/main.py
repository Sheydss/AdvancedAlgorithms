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


import colonie_fourmie as aco
import random

# Nombre de nœuds et nombre maximum d'arêtes par nœud
num_nodes = 10
max_edge = 5

# Générer un graphe aléatoire
graph = aco.Graph(num_nodes, max_edge)

#start and end points
start = 0
end = 9

# Extraire les points du graphe
points = list(range(num_nodes))

# Paramètres de l'algorithme de la colonie de fourmis
nombre_fourmis = 10
alpha = 1.0
beta = 2.0
evaporation = 0.5
iterations = 100

# Exécuter l'algorithme de la colonie de fourmis
meilleur_chemin, meilleur_cout, temps_execution = aco.colonie_de_fourmis(graph, points, nombre_fourmis, alpha, beta, evaporation, iterations, start, end)


# Afficher les résultats
print("Meilleur chemin:", meilleur_chemin)
print("Meilleur coût:", meilleur_cout)
print("Temps d'exécution:", temps_execution, "secondes")

# Afficher le graphe avec le meilleur chemin
graph.plot_graph([meilleur_chemin])
