import random
import networkx
class algoGenetic:

    def __init__(self, graph, generations = 20, population_size = 20):
            self.graph = graph
            self.generations = generations
            self.population_size = population_size

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
    
    def create_random_population(self, start_node, end_node, points):
        population = []
        for _ in range(self.population_size):
            path = [start_node]
            current_node = start_node
            while not(all(item in path for item in points) and current_node == end_node): 
                neighbors = list(self.graph.neighbors(current_node))
                next_node = random.choice(neighbors)
                path.append(next_node)
                current_node = next_node
            population.append(path)
        return population

    def calculate_fitness(self, paths):
        fitness_scores = []
        for path in paths:
            fitness_scores.append(self.calculer_distance(path))
        return fitness_scores

    def random_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness > 0:
            probabilities = [(total_fitness - score) / total_fitness for score in fitness_scores]
            return random.choices(population, probabilities)[0]
        return 0

    def crossover(self, parents, predefined_points, start_node, end_node):
        childs = []
        for parent1, parent2 in parents:
            child = []
            #### doen't work ####
            childs.append(child)
        return childs

    def mutate(self, paths, mutation_rate, points):
        for path in paths:
            if random.random() < mutation_rate:
                mp = len(path) - 2
                if mp >= 1:
                    mutation_point = random.randint(1, mp)
                    d = 0
                    while path[mutation_point] in points and d < len(path):
                        mutation_point = random.randint(1, mp)
                        d+=1
                    p1 = self.graph.neighbors(path[mutation_point-1])
                    p2 = self.graph.neighbors(path[mutation_point+1])
                    p = list(set(p1).intersection(p2))
                    new_node = random.choice(p)
                    path[mutation_point] = new_node
        return paths

    def genetic_algorithm(self, start_node, end_node, points, mutation_rate = 0.2):
        population = self.create_random_population(start_node, end_node, points)
        print(population)
        for gen in range(self.generations):
            print('generation', gen)
            fitness_scores = self.calculate_fitness(population)
            
            parents = []
            for _ in range(int(self.population_size / 2)):
                parent1 = self.random_selection(population, fitness_scores)
                parent2 = self.random_selection(population, fitness_scores)
                parents.append((parent1, parent2))
                
            offspring = self.crossover(parents, points, start_node, end_node)
            offspring = self.mutate(offspring, mutation_rate, points)
            population = parents + offspring
        
        best_path = min(population, key=lambda path: self.calculate_fitness([path])[0])
        return best_path