#generation of a random graph
# problem : since I work in jupyter-lab, sometime when I run the code, well it take time and don't execute my code at all
# since it's currently 12:59 pm I will stop this activity for now. Maybe I will resume it in the morning.

#imports
import random
import networkx as nx
import matplotlib.pyplot as plt

#we must generate the cities and relations between cities


#first we generate a random size for the list of points
taille = random.randint(1,10) # I don't want to make a large graph so we will stick to this

# then we generate a liste of points of this size
list_city = []
for i in range (taille) :
    if (i !=0):
        list_city.append(i) # so we will have stuff like 1, 2, 3 etc ...)

# Now we generate couple of points (x,y) as x !=y, x and y are city from list_city
vectors = [(random.randint(1,taille),random.randint(1,taille)) for city in list_city]
#now we want to make sure that every cities are related so we will se if there is two cities not related we will add them in the list
for i in list_city :
    for j in list_city :
        for (a,b) in vectors:
            if (i!=j and (i,j) != (a,b) ) :
                vectors.append((i,j))

#the distance between cities
distance = {i: random.randint(1,40) for i in range(taille)}
print(distance)
#Now we generate the graph

#we create the graph
G = nx.Graph()

#add the cities
G.add_nodes_from(list_city)

#add the relations
G.add_edges_from(vectors)

#print the graph now
#nx.draw(G, with_labels=True)
#plt.show()
# ta_da !

# Now I just need to create a graph that's like hamiltonien (done, need more tests)
# and make a list of the kilometers between each city( the cost) and print it on the graph
# need to return some informations in a way that my ACO algorithm can understand
