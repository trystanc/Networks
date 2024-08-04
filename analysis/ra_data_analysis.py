import numpy as np
import matplotlib.pyplot as plt
from logbin import logbin
import scipy.stats as sp
import matplotlib

data = np.loadtxt("RA_data_1.csv", delimiter = ",")
ms = [2,3,4,5,10,15,20]
def kol_smir_test(data,dist,m):
    vals = sp.ks_1samp(data, dist, args = [m])
    return vals

def ba_theory(k,m):
    '''
    returns the value of p(k) for a given given m, where k is an array or an integer.
    '''
    return (2*(m))*(m+1)/((k)*(k+1)*(k+2))

def ba_theory_cum(k, *args):
    z = args[0]
    k *= k>=z
    output = 1/z - 1/(z+1)
    output += 1/(k+2) - 1/(k+1)
    output *= z*(z+1)

    return output

    



def chi_sq(observed_distribution, expected_distribution):

    chi2 = np.sum(((observed_distribution - expected_distribution)**2/expected_distribution))
    p_value = 1 - sp.chi2.cdf(chi2,len(observed_distribution))
    return (chi2, p_value)
    


#from the datasets, logbin those with the same m
xs = []
ys = []
y_errs =[]
probs = []
bin_edges = []
ks_p_values = []
k1_values = []
for m in range(7):
    m_data = data[20*m:20*(m+1)].flatten()
    k1_values.append(np.max(m_data))

    x,y= logbin(m_data.astype(int), scale=1.2)
    xs.append(np.array(x))
    ks_p_values.append(sp.ks_1samp(m_data, ba_theory_cum, args = [ms[m]-1]))
    ys.append(np.array(y))
print(ks_p_values)

#get the standard error on the height of each bin from each iteration
all_errors = []

for m in range(7):
    y_values = []


    for iter in range(20):
        dataset = data[m*20+iter]
        if k1_values[m] not in dataset:
            dataset = np.append(dataset, k1_values[m])

            x,y = logbin(dataset.astype(int), scale = 1.2, modify = True)
        else:
            x,y = logbin(dataset.astype(int), scale = 1.2)
        y_values.append(y)
    y_values = np.array(y_values)
    err = np.std(y_values, axis = 0)/np.sqrt(20)
    all_errors.append(np.std(y_values, axis = 0)/np.sqrt(20))  

#plot p(k) with associated uncertainties.
for i in range(7):
    plt.scatter(xs[i], ys[i],  marker = 'x', label = f"m = {ms[i]}")
    plt.plot(xs[i], (2*(ms[i]))*(ms[i]+1)/((xs[i])*(xs[i]+1)*(xs[i]+2)), ls = "dashed")

plt.legend()
plt.yscale("log")
plt.ylabel("P(k)")
plt.xlabel("k")
plt.show()
