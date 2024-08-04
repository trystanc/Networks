# %%
import numpy as np
import matplotlib.pyplot as plt
from logbin import logbin
import scipy.stats as sp
import matplotlib
import scipy.stats
import numpy as np
data = np.loadtxt("RA_data_1.csv", delimiter = ",")
print(data.shape)
ms = [2,4,8,16,32]



def rand_dist(k,m):
    output = np.power(m/(m+1), k-m)
    output /= m+1
    return output
def kol_smir_test(data, m):
    N = len(data)
    
    data_sorted = np.sort(data)

    data_cdf = 1. * np.arange(len(data)) / (len(data) - 1)
    dist_cdf = ba_theory_cum(data_sorted, m)
    D = np.max(np.abs(data_cdf - dist_cdf))
    stat = np.sqrt(N)*D
    p_value = scipy.special.kolmogorov(stat)
    return p_value
def chi_sq(observed_distribution, expected_distribution):

    chi2 = np.sum((observed_distribution - expected_distribution)**2/expected_distribution)
    dof = len(observed_distribution) - 1
    return (chi2, sp.chi2.sf(chi2,dof))

# %%
xs = []
ys = []
y_errs =[]
probs = []
bin_edges = []
ks_p_values = []
k1_values = []
for m in range(5):
    m_data = data[10*m:10*(m+1)].flatten()
    k1_values.append(np.max(m_data))
    if m>1:
        x,y= logbin(m_data.astype(int), scale=1.05)
    else:
         x,y= logbin(m_data.astype(int), scale=1.05)
    indicies = np.array(y)>0
    xs.append(np.array(x))
    expected = rand_dist(x[indicies],ms[m])
    ks_p_values.append(chi_sq(y[indicies], expected[:]))
    ys.append(np.array(y))
print(ks_p_values)
from logbin import logbin
final_errors = []
for m in range(5):
    y_vals = []
    for n in range(10):
        realisation = data[m*10+n,:]
        x_new,y_new = logbin(realisation.astype(int), scale = 1.05, modify = k1_values[m])
        print(len(x_new))
        y_vals.append(y_new)
    y_vals = np.array(y_vals)
    final_errors.append(1.96*np.std(y_vals, axis = 0)/np.sqrt(10))


# %%
from scipy.stats import chi2
def reduced_chi_sq(data, expected, errors):
    chi = np.sum(((data-expected)/errors)**2)
    p_val = 1 - chi2.cdf(chi,len(data)-1)
    return chi/(len(data)-1), p_val
results = []
for i in range(5):
    indicies = np.logical_and(np.array(ys[i])>0, np.array(xs[i])< 3.2*max(xs[i]/4))
    x_vals = xs[i][indicies]
    y_vals = ys[i][indicies]
    yerrs = final_errors[i][indicies]
    expected = rand_dist(x_vals, ms[i])
    results.append(reduced_chi_sq(y_vals[1:], expected[1:], np.sqrt(10)*yerrs[1:]/1.96))
print(results)


# %%
marker = ["D", "o", "^", "v", "s", "h", "x"]
colors = ["crimson", "rebeccapurple", "royalblue", "mediumseagreen", "lightcoral","darkolivegreen", "orange"]
plt.rcParams["figure.figsize"] = (5,2.5)
plt.rcParams["font.family"] = "Sans Serif"
plt.rcParams['xtick.labelsize'] = "10"
plt.rcParams['ytick.labelsize'] = "10"

#plot p(k) with associated uncertainties.
for i in range(5):
    indicies = np.array(ys[i])>0
    x_vals = xs[i][indicies]
    y_vals = ys[i][indicies]
    yerrs = final_errors[i][indicies]
    plt.errorbar(x_vals, y_vals, yerr = yerrs, marker = 'x', ls = "none" , capsize = 1.5, fillstyle ="none",
    markersize = 3, color=colors[i], label = f"m = {ms[i]}", elinewidth = 0.6)
    plt.plot(x_vals, rand_dist(x_vals, ms[i]), linewidth = 0.6, color = colors[i], ls = "dashed")
plt.xscale("linear")
plt.legend(fontsize = 8)
plt.yscale("log")
plt.ylabel("P(k)")
plt.xlabel("k")
plt.savefig('ra.png', dpi=300, bbox_inches = "tight")
plt.show()


