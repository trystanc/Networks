# %%
import numpy as np
from scipy.special import gammaln
from logbin import logbin
import scipy.stats as sp
import matplotlib.pyplot as plt
data = np.loadtxt("mixed_data_1.csv", delimiter = ',')
print(data.shape)
ms = [3,9,27,81]
def dist(k,m):
    r = m/3
    output = np.log(3)
    output -= np.log(3*(1+r)+2*r)
    output += gammaln(5*r/2+5/2)
    output -= gammaln(5*r/2)
    output += gammaln(k+3*r/2)
    output -= gammaln(k+3*r/2+5/2)
    return np.exp(output)

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
scales = [1.4,1.3,1.25,1.2]
for m in range(4):
    m_data = data[10*m:10*(m+1)].flatten()
    k1_values.append(np.max(m_data))
    if m>1:
        x,y= logbin(m_data.astype(int), scale=scales[m])
    else:
         x,y= logbin(m_data.astype(int), scale=scales[m])
    indicies = np.array(y)>0
    xs.append(np.array(x))
    expected = dist(x[indicies],ms[m])
    ks_p_values.append(chi_sq(y[indicies], expected[:]))
    ys.append(np.array(y))
print(ks_p_values)

from logbin import logbin
final_errors = []
for m in range(4):
    y_vals = []
    for n in range(10):
        realisation = data[m*10+n,:]
        x_new,y_new = logbin(realisation.astype(int), scale = scales[m], modify = k1_values[m])
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
for i in range(4):
    indicies = np.logical_and(np.array(ys[i])>0, np.array(xs[i])< max(xs[i]*0.01))
    x_vals = xs[i][indicies]
    y_vals = ys[i][indicies]
    yerrs = final_errors[i][indicies]
    expected = dist(x_vals, ms[i])
    results.append(reduced_chi_sq(y_vals[1:], expected[1:], np.sqrt(10)*yerrs[1:]/1.96))
print(results)


# %%


# %%
marker = ["D", "o", "^", "v", "s", "h", "x"]
colors = ["crimson", "rebeccapurple", "royalblue", "mediumseagreen", "lightcoral","darkolivegreen", "orange"]
plt.rcParams["figure.figsize"] = (5,2.5)
plt.rcParams["font.family"] = "Sans Serif"
plt.rcParams['xtick.labelsize'] = "10"
plt.rcParams['ytick.labelsize'] = "10"

#plot p(k) with associated uncertainties.
for i in range(4):
    indicies = np.array(ys[i])>0
    x_vals = xs[i][indicies]
    y_vals = ys[i][indicies]
    yerrs = final_errors[i][indicies]
    plt.errorbar(x_vals, y_vals, yerr = yerrs, marker = 'x', ls = "none" , capsize = 1.5, fillstyle ="none",
    markersize = 3, color=colors[i], label = f"m = {ms[i]}", elinewidth = 0.6, alpha = 0.7)
    plt.plot(x_vals, dist(x_vals, ms[i]), linewidth = 0.6, color = colors[i], alpha = 0.5, ls = "dashed")
plt.xscale("log")
plt.legend(fontsize = 8)
plt.yscale("log")
plt.ylabel("P(k)")
plt.xlabel("k")
plt.savefig('mixed.png', dpi=300, bbox_inches = "tight")
plt.show()


