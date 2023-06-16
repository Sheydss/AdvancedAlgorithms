from reportlab.lib.colors import red

import algoRecuitSimule as algo

if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 20
    max_edges_per_node = 5

    graph = algo.Graph(num_nodes, max_edges_per_node)

    # Exemple d'utilisation
    cities = [
        1,3,5
    ]

    best_path, best_distance = algo.Graph.simulated_annealing(cities)
    print("Meilleur chemin trouvé:", best_path)
    print("Distance totale:", best_distance)

