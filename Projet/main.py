from reportlab.lib.colors import red

import graph as g

if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 15
    max_edges_per_node = 5

    graph = g.Graph(num_nodes, max_edges_per_node)

    start = 1
    end = 8
    initial_solution = list(range(num_nodes))

    path, distance = graph.dijkstra(start, end)
    graph.plot_graph(path, 'red')
    graph.data_graph()
    graph.dijkstra(start, end)


