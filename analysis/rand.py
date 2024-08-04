from random import randint, choice
import numpy as np
import matplotlib.pyplot as plt
edges = []
degrees = np.zeros(100)
N = 100
for i in range(5000):
    source = randint(1,100)
    degrees[source-1] +=1 
    dest = randint(1,100)
 
    while source == dest:
        dest = randint(1,100)
    degrees[dest-1] +=1   
    edges.append(source)
    edges.append(dest)    
print(edges)

probabilities = np.zeros(100)

for i in range(int(1e7)):
    chosen_node = choice(edges)
    probabilities[chosen_node-1] +=1
probabilities /= np.sum(probabilities)
plt.scatter(degrees,probabilities)
plt.xlabel("node degree")
plt.ylabel("selection probabilitiy")
plt.show()


