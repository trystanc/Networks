import numpy as np
import matplotlib.pyplot as plt
from logbin import logbin
data = np.loadtxt("PA_data_1.csv", delimiter = ",")

#from the datasets, logbin those with the same m
xs = []
ys = []
bin_edges = []
for m in range(7):
    m_data = data[20*m:20*(m+1)].flatten()
    x,y,bin_edge = logbin(m_data.astype(int), scale = 1.3)
    xs.append(np.array(x))
    ys.append(np.array(y))

    bin_edges.append(bin_edge)
print(bin_edges[1])
#get the standard error on the height of each bin from each iteration
all_errors = []
for m in range(6):
    y_values = []
    for iter in range(20):
        dataset = data[m*20+iter]
        hist, bin_edges = np.histogram(dataset, bins = bin_edges[m].reshape(bin_edges[m].shape[0]), density = True)
        y_values.append(hist)
  
    y_values = np.array(y_values)

    all_errors.append(np.std(y_values, axis = 1)/3)

        


    


#plot p(k) with associated uncertainties.
for i in range(len(xs)):
    plt.scatter(xs[i], ys[i], marker = 'x', label = f"m = {(5)*(i+1)}")
    plt.plot(xs[i], (2*(5*(i+1))*(5*(i+1)+1))/((xs[i])*(xs[i]+1)*(xs[i]+2)), ls = '--')
plt.xscale("log")
plt.legend()
plt.yscale("log")
plt.ylabel("P(k)")
plt.xlabel("k")
plt.show()


