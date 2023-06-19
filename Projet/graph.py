import heapq
import itertools
import math
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx


class Graph2:
    def __init__(self, num_nodes, max_edge):
        self.graph = nx.Graph()
        self.generate_random_graph(num_nodes, max_edge, 10, 100)

    def generate_random_graph(self, num_nodes, max_edges_per_node, min_weight, max_weight):
        print("---------------------------------------------------------")
        print(f"Generating Graph with {num_nodes} nodes")
        tic = time.perf_counter()

        # Ajouter les nœuds au graphe avec des coordonnées aléatoires
        for node in range(num_nodes):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.graph.add_node(node, pos=(x, y))

        # Générer les arêtes avec des poids correspondant à la distance euclidienne
        for node in range(num_nodes):
            num_edges = random.randint(1, max_edges_per_node)
            dest_nodes = random.sample(range(num_nodes), num_edges)

            for dest_node in dest_nodes:
                if node != dest_node:
                    pos1 = self.graph.nodes[node]['pos']
                    pos2 = self.graph.nodes[dest_node]['pos']
                    distance = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                    weight = int(distance)
                    self.graph.add_edge(node, dest_node, weight=weight)

        toc = time.perf_counter()
        print(f"Generation done in {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

    def dijkstra(self, start, end):

        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0

        heap = [(0, start)]
        while heap:
            current_cost, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            for neighbor in self.graph.neighbors(current_node):
                cost = self.graph[current_node][neighbor]['weight']
                new_cost = current_cost + cost
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor))

        path = self.reconstruct_path(start, end, distances)

        return path, distances[end]

    def reconstruct_path(self, start, end, distances):
        current_node = end
        path = [current_node]
        while current_node != start:
            neighbors = self.graph.neighbors(current_node)
            min_neighbor = None
            min_cost = float('inf')
            for neighbor in neighbors:
                cost = self.graph[current_node][neighbor]['weight']
                if distances[neighbor] + cost < min_cost:
                    min_neighbor = neighbor
                    min_cost = distances[neighbor] + cost
            path.append(min_neighbor)
            current_node = min_neighbor
        path.reverse()
        return path

    def print_dijkstra(self, path):
        for i in path:
            list_values = []
            print(f"Node : {i}")
            for edge in self.graph.neighbors(i):
                values = dict()
                values[edge] = self.graph[i][edge]['weight']
                list_values.append(values)
            print(list_values)

    def data_graph(self):
        for node in self.graph.nodes:
            list_values = []
            print(f"Node : {node}")
            for edge in self.graph.neighbors(node):
                values = dict()
                values[edge] = self.graph[node][edge]['weight']
                list_values.append(values)
            print(list_values)

    def plot_graph(self, paths=None, colors=None):
        # Obtenir les positions des nœuds pour le tracé
        pos = nx.get_node_attributes(self.graph, 'pos')

        # Dessiner le graphe avec les arêtes des chemins coloriées
        nx.draw(self.graph, pos=pos, width=0.8, with_labels=True)

        if paths is not None:
            if colors is None:
                colors = ['black'] * len(paths)  # Par défaut, utiliser la couleur noire pour tous les chemins
            elif len(colors) < len(paths):
                raise ValueError("Le nombre de couleurs fourni est inférieur au nombre de chemins.")

            for path, color in zip(paths, colors):
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edges, edge_color=color, width=2)

        plt.show()

    def a_star(self, start, end):

        # Distances du point de départ à chaque nœud (g-cost)
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0

        # Estimations des coûts du point de départ à chaque nœud (f-cost)
        estimations = {node: self.heuristic(node, end) for node in self.graph.nodes}

        # File de priorité pour l'exploration des nœuds
        heap = [(estimations[start], 0, start)]  # (f-cost, g-cost, node)

        # Dictionnaire pour stocker les nœuds précédents sur le chemin optimal
        came_from = {}

        while heap:
            _, current_cost, current_node = heapq.heappop(heap)

            if current_node == end:
                break

            for neighbor in self.graph.neighbors(current_node):
                cost = self.graph[current_node][neighbor]['weight']
                new_cost = current_cost + cost
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, end)
                    heapq.heappush(heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current_node

        path = self.reverse_path(start, end, came_from)

        return path, distances[end]

    def heuristic(self, node, goal):
        # Heuristique utilisée pour estimer le coût restant (ici, distance euclidienne)
        pos1 = self.graph.nodes[node]['pos']
        pos2 = self.graph.nodes[goal]['pos']
        distance = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
        return distance

    def reverse_path(self, start, end, came_from):
        current_node = end
        path = [current_node]
        while current_node != start:
            current_node = came_from[current_node]
            path.append(current_node)
        path.reverse()
        return path

    def voisinage(self, node, visited, destination_finale, solution_courante):
        voisins = []
        for v in self.graph.neighbors(node):
            if v not in visited or v == destination_finale:
                if v not in solution_courante:
                    voisins.append(v)
        return voisins

    def valeur_contenu(self, source, destination):
        if source != destination:
            return self.graph[source][destination]['weight']
        return float('inf')

    def calculer_distance(self, chemin):
        distance_totale = 0

        for i in range(len(chemin) - 1):
            source = chemin[i]
            destination = chemin[i + 1]
            distance = self.valeur_contenu(source, destination)
            distance_totale += distance

        return distance_totale

    def tabou_search(self, solution_initiale, destination_finale, taille_tabou, iter_max):

        nb_iter = 0
        liste_tabou = deque(maxlen=taille_tabou)

        solution_courante = [solution_initiale]
        meilleurs_voisin = solution_courante
        meilleure_globale = solution_courante

        # Avoir une valeur élevée juste pour pouvoir comparer :
        valeur_meilleure_globale = float('inf')

        liste_tabou.append(solution_courante)

        while nb_iter < iter_max:
            valeur_meilleure = float('inf')

            # Parcours des voisins
            for voisin in self.voisinage(solution_courante[-1], liste_tabou, destination_finale, solution_courante):
                valeur_to_voisin = self.valeur_contenu(solution_courante[-1], voisin)
                # Selection du meilleur voisin
                if valeur_to_voisin < valeur_meilleure:
                    meilleurs_voisin = voisin
                # Changement de la meilleure valeur pour continuer de comparer
                valeur_meilleure = self.valeur_contenu(solution_courante[-1], voisin)

            # Permet de garder la valeur du chemin avec le dernier meilleur voisin
            if self.valeur_contenu(solution_courante[-1], meilleurs_voisin) < valeur_meilleure_globale:
                meilleure_globale.append(meilleurs_voisin)
                if solution_courante[-1] != meilleurs_voisin:
                    valeur_meilleure_globale = self.valeur_contenu(solution_courante[-1], meilleurs_voisin)
                nb_iter = 0
            else:
                nb_iter += 1

            # Progression dans le graphe en prenant le meilleur voisin
            solution_courante = meilleure_globale
            liste_tabou.append(meilleurs_voisin)

            if solution_courante[-1] == destination_finale:
                break
        distance_meilleure_globale = self.calculer_distance(meilleure_globale)
        return meilleure_globale, distance_meilleure_globale

    def genetic_search(self, source, target, population_size=50, generations=100, mutation_rate=0.2):
        print("---------------------------------------------------------")
        print(f"Genetic Search: From {source} to {target}")
        tic = time.perf_counter()
        # Créer une population initiale aléatoire
        population = []
        for _ in range(population_size):
            path = self.random_shortest_path(source, target)
            population.append(path)
        
        for gen in range(generations):
            #print(gen, ' génération')
            # Évaluer la fitness de chaque individu de la population
            fitness_scores = []
            for individual in population:
                individual = individual[:individual.index(target)]
                #print(individual)
                fitness = self.calculer_distance(individual)  # Longueur du chemin
                fitness_scores.append(fitness)
            
            # si la sommes des fitness est de 0 alors tous les individus on trouvé un chemin direct
            # pour éviter une division par 0 et ne pas faire des tours inutile je sors du for 
            if sum(fitness_scores) != 0:
                break
            # Sélectionner les meilleurs individus pour la reproduction
            selected_parents = []
            for _ in range(int(population_size / 2)):
                parent1 = self.random_selection(population, fitness_scores)
                parent2 = self.random_selection(population, fitness_scores)
                selected_parents.append((parent1, parent2))

            # Reproduction (croisement)
            offspring_population = []
            for parents in selected_parents:
                parent1, parent2 = parents
                temp = self.crossover(parent1, parent2)
                offspring_population.append(temp)

            # Mutation
            for i in range(len(offspring_population)):
                if random.random() < mutation_rate:
                    mp = len(offspring_population[i]) - 2
                    if mp >= 1:
                        mutation_point = random.randint(1, mp)
                        p1 = self.graph.neighbors(offspring_population[i][mutation_point-1])
                        p2 = self.graph.neighbors(offspring_population[i][mutation_point+1])
                        p = list(set(p1).intersection(p2))
                        new_node = random.choice(p)
                        offspring_population[i][mutation_point] = new_node
            
            # Remplacement de la population
            population = offspring_population
        
        # Retourner le chemin le plus court trouvé
        best_path = min(population, key=lambda x: self.calculer_distance(x))
        distance_meilleure_globale = self.calculer_distance(best_path)

        toc = time.perf_counter()

        print(f"Path: {best_path} Cost: {distance_meilleure_globale}")
        print(f"Duration: {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")
        return best_path, distance_meilleure_globale


    def random_shortest_path(self, source, target):
        # Générer un chemin aléatoire entre la source et la cible
        path = []
        path.append(source)
        current_node = source
        while current_node != target:
            neighbors = list(self.graph.neighbors(current_node))
            next_node = random.choice(neighbors)
            path.append(next_node)
            current_node = next_node
        return path


    def random_selection(self, population, fitness_scores):
        # Sélectionner un individu de la population de manière aléatoire pondérée par les scores de fitness
        total_fitness = sum(fitness_scores)
        probabilities = [(total_fitness - score) / total_fitness for score in fitness_scores]
        return random.choices(population, probabilities)[0]

    def crossover(self, parent1, parent2):
        for i in range(1, len(parent1)):
            for j in range(1, len(parent2)):
                if parent2[j] == parent1[i]:
                    if(i < j):
                        #print(i, j, parent1, parent2, parent1[:i] + parent2[j:], sep=' - ')
                        return parent1[:i] + parent2[j:]
                    else:
                        #print(j, i, parent2, parent1, parent2[:j] + parent1[i:], sep=' - ')
                        return parent2[:j] + parent1[i:]
        return parent1

    def best_itinerary(self, points, start, end):
        print("---------------------------------------------------------")
        print(f"Looking for Best Path:")
        tic = time.perf_counter()

        best_objects_path = [start]
        chemin_global = []
        cout_meilleur_chemin = float('inf')

        for permutation in itertools.permutations(points):
            _, cout_start = self.dijkstra(start, permutation[0])
            cout_total = cout_start

            paths = []

            for i in range(len(permutation) - 1):
                point_actuel = permutation[i]
                point_suivant = permutation[i + 1]
                path, cout = self.dijkstra(point_actuel, point_suivant)
                cout_total += cout
                paths.extend(path)

            if cout_total < cout_meilleur_chemin:
                cout_meilleur_chemin = cout_total
                best_objects_path = list(permutation)
                chemin_global = paths[:]

        chemin_global = self.remove_consecutive_duplicates(chemin_global)
        chemin_global.insert(0, start)
        chemin_global.append(end)
        toc = time.perf_counter()

        print(f"Objects Path: {best_objects_path}")
        print(f"Path: {chemin_global}")
        print(f"Cost: {cout_meilleur_chemin}")
        print(f"Duration: {toc - tic:0.4f} seconds")
        print("---------------------------------------------------------")

        return best_objects_path, cout_meilleur_chemin, chemin_global

    def remove_consecutive_duplicates(self, lst):
        result = [lst[0]]  # Ajouter le premier élément à la liste résultat

        for i in range(1, len(lst)):
            if lst[i] != lst[i - 1]:
                result.append(lst[i])  # Ajouter l'élément à la liste résultat s'il est différent du précédent

        return result

    def base_itinerary_cost(self, path):
        distance_totale = 0

        for i in range(len(path) - 1):
            source = path[i]
            destination = path[i + 1]
            _, distance = self.dijkstra(source, destination)
            distance_totale += distance

        return distance_totale
