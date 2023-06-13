import graph as g


if __name__ == '__main__':
    # Exemple d'utilisation
    g = g.Graph(2000, 5)
    #g.print_graph()

    start_node = 1
    end_node = 1578

    path, cost = g.dijkstra(start_node, end_node)
    if path is None:
        print("No path found.")
    else:
        print(f"Shortest Path from {start_node} to {end_node} :", path)
        print("Cost:", cost)




