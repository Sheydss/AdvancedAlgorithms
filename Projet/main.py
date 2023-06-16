from reportlab.lib.colors import red

import graph as g

if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 20
    max_edges_per_node = 5

    graph = g.Graph(num_nodes, max_edges_per_node)

    start = 1
    end = 5

    graph.a_star(start, end)
    path, distance = graph.dijkstra(start, end)
    graph.plot_graph(path, 'red')


