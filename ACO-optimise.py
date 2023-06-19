import numpy as np
import time

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def initialiser_pheromones(n):
    return np.ones((n, n))

def mettre_a_jour_pheromones(pheromones, chemin, cout_total, alpha, beta):
    n = pheromones.shape[0]
    for i in range(n):
        ville_courante = chemin[i]
        ville_suivante = chemin[(i + 1) % n]
        pheromones[ville_courante, ville_suivante] = (1 - alpha) * pheromones[ville_courante, ville_suivante] + alpha / cout_total
        pheromones[ville_suivante, ville_courante] = pheromones[ville_courante, ville_suivante]

def choisir_prochaine_ville(pheromones, visites, ville_courante, alpha, beta):
    n = pheromones.shape[0]
    max_phéromone = -1
    ville_suivante = -1
    for ville in range(n):
        if not visites[ville]:
            pheromone = pheromones[ville_courante, ville] * (1.0 / distance(ville_courante, ville) ** beta)
            if pheromone > max_phéromone:
                max_phéromone = pheromone
                ville_suivante = ville
    return ville_suivante

def colonie_de_fourmis(points):
    nombre_fourmis = 10
    alpha = 1.0
    beta = 2.0
    evaporation = 0.5
    iterations = 100
    n = len(points)
    #faire disparaitre cette partie
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = distance(points[i], points[j])

    pheromones = initialiser_pheromones(n)
    meilleur_chemin = None
    meilleur_cout = float('inf')

    start_time = time.time()

    for _ in range(iterations):
        for _ in range(nombre_fourmis):
            visites = [False] * n
            chemin = []
            cout_total = 0
            ville_courante = np.random.randint(n)
            visites[ville_courante] = True
            chemin.append(ville_courante)

            for _ in range(n - 1):
                ville_suivante = choisir_prochaine_ville(pheromones, visites, ville_courante, alpha, beta)
                visites[ville_suivante] = True
                chemin.append(ville_suivante)
                cout_total += distances[ville_courante, ville_suivante]
                ville_courante = ville_suivante

            cout_total += distances[ville_courante, chemin[0]]

            if cout_total < meilleur_cout:
                meilleur_chemin = chemin
                meilleur_cout = cout_total

            mettre_a_jour_pheromones(pheromones, chemin, cout_total, evaporation, 1)

    execution_time = time.time() - start_time

    return meilleur_chemin, meilleur_cout, execution_time

# Exemple d'utilisation
points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

meilleur_chemin, meilleur_cout, temps_execution = colonie_de_fourmis(points)

print("Meilleur chemin:", meilleur_chemin)
print("Meilleur coût:", meilleur_cout)
print("Temps d'exécution:", temps_execution, "secondes")
