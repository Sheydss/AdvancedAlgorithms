from reportlab.lib.colors import red

import graph as g

if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 10
    max_edges_per_node = 5

    graph = g.Graph(num_nodes, max_edges_per_node)

    start = 1
    end = 5

    path, distance = graph.dijkstra(start, end)
    graph.data_graph()
    graph.color_path(path, 'red')


