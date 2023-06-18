#code de chatGPT

import numpy as np


# Fonction pour initialiser les phéromones
def initialiser_pheromones(n):
    return np.ones((n, n))

# Fonction pour mettre à jour les phéromones
def mettre_a_jour_pheromones(pheromones, chemin, cout_total, alpha, beta):
    n = pheromones.shape[0]
    for i in range(n):
        ville_courante = chemin[i]
        ville_suivante = chemin[(i + 1) % n]
        pheromones[ville_courante, ville_suivante] = (1 - alpha) * pheromones[ville_courante, ville_suivante] + alpha / cout_total
        pheromones[ville_suivante, ville_courante] = pheromones[ville_courante, ville_suivante]

# Fonction pour choisir la prochaine ville à visiter
def choisir_prochaine_ville(pheromones, visites, ville_courante, alpha, beta):
    n = pheromones.shape[0]
    max_phéromone = -1
    ville_suivante = -1
    for ville in range(n):
        if not visites[ville]:
            pheromone = pheromones[ville_courante, ville]
            if pheromone > max_phéromone:
                max_phéromone = pheromone
                ville_suivante = ville
    return ville_suivante

# Fonction pour résoudre le TSP avec l'algorithme de la colonie de fourmis
def colonie_de_fourmis(distances, nombre_fourmis, alpha, beta, evaporation, iterations):
    n = distances.shape[0] # Nombre de villes
    pheromones = initialiser_pheromones(n)
    meilleur_chemin = None
    meilleur_cout = float('inf')

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

            cout_total += distances[ville_courante, chemin[0]] # Ajouter le coût du retour à la première ville

            if cout_total < meilleur_cout:
                meilleur_chemin = chemin
                meilleur_cout = cout_total

            mettre_a_jour_pheromones(pheromones, chemin, cout_total, evaporation, 1)

    return meilleur_chemin, meilleur_cout

# Exemple d'utilisation
distances = np.array([[0, 2, 9, 10],
                      [1, 0, 6, 4],
                      [15, 7, 0, 8],
                      [6, 3, 12, 0]])

nombre_villes = distances.shape[0]
nombre_fourmis = 10
alpha = 1.0
beta = 2.0
evaporation = 0.5
iterations = 100

meilleur_chemin, meilleur_cout = colonie_de_fourmis(distances, nombre_fourmis, alpha, beta, evaporation, iterations)

print("Meilleur chemin:", meilleur_chemin)
print("Meilleur coût:", meilleur_cout)
