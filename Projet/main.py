import graph as g

if __name__ == '__main__':
    # Exemple d'utilisation
    num_nodes = 10
    max_edges_per_node = 5

    graph = g.Graph(num_nodes, max_edges_per_node)

    start = 1
    end = 5

    graph.a_star(start, end)
    path, distance = graph.dijkstra(start, end)
    path2, distance2 = graph.recherche_tabou(start, end, 20, 200)
    graph.plot_graph(paths=[path, path2], colors=['red', 'green'])
