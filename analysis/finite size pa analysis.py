# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.stats import linregress
import tikzplotlib
from logbin import logbin
marker = ["D", "o", "^", "v", "s", "p", "p"]
colors = ["crimson", "rebeccapurple", "royalblue", "mediumseagreen", "lightcoral","darkolivegreen"]
def ba_theory(k,m):
    '''
    returns the value of p(k) for a given given m, where k is an array or an integer.
    '''
    return (2*(m))*(m+1)/((k)*(k+1)*(k+2))

data = np.loadtxt("PA_data_fs.csv", delimiter = ',')
sizes = np.array([100,1000,10000,100000,500000,1000000])
xs = []
ys = []
max_k = []
bin_edges = []
for m in range(6):
    m_data = data[60*m:60*(m+1)].flatten()
    m_data = m_data[m_data>0]
    x,y = logbin(m_data.astype(int), scale = 1.2)
    xs.append(np.array(x))
    ys.append(np.array(y))
    max_k.append(np.max(m_data))


# %%
final_errors = []
for m in range(6):
    y_vals = []
    for n in range(60):
        realisation = data[m*60+n,:]
        x_new,y_new = logbin(realisation.astype(int), scale = 1.2, modify = max_k[m])
        print(len(x_new))
        y_vals.append(y_new)
    y_vals = np.array(y_vals)
    final_errors.append(1.96*np.std(y_vals, axis = 0)/np.sqrt(60))

# %%
plt.rcParams["figure.figsize"] = (5,2.5)
plt.rcParams["font.family"] = "Arial"
plt.rcParams['xtick.labelsize'] = "10"
plt.rcParams['ytick.labelsize'] = "10"
for i in range(6):
    string = f"N={sizes[i]}"
    plt.errorbar(xs[i],ys[i], yerr = final_errors[i], elinewidth =0.6, ls = "none", capsize = 1.8, marker = "x", markersize= 2.5, color = colors[i], label = string)
t = np.linspace(start = 5, stop = 5000)
plt.plot(t,2*5*6/((t)*(t+1)*(t+2)), ls = "--", linewidth = 1,label = "infinite limit")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$k$", fontsize = 10)
plt.ylabel("$P(k)$", fontsize = 10)
plt.legend(fontsize = 8)
plt.savefig('fixed_m1.png', dpi=300, bbox_inches = "tight")
plt.show()

for i in range(6):
    plt.errorbar(xs[i]/sizes[i]**0.5, ys[i]*(xs[i])*(xs[i]+1)*(xs[i]+2), yerr =final_errors[i]*(xs[i])*(xs[i]+1)*(xs[i]+2) ,elinewidth = 1,
    capsize = 1.5, ls = "none",color = colors[i], marker = "x", markersize = 4,
         label = f"$N={sizes[i]}$")


t = np.linspace(start = 0.005, stop = 7, num = 100)
plt.plot(t,[2*5*(5+1) for i in range(100)], ls = "--", label = "$2m(m+1)$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$k/N^{0.5}$", fontsize = 10)
plt.ylabel("$P(k)k(k+1)(k+2)$", fontsize = 10)
plt.legend(fontsize = 8)
plt.savefig('data_collapse1.png', dpi=300, bbox_inches = "tight")
plt.show()
mean_k1s = []
err_k1s = []



for m in range(6):
    k1s = []
    for iter in range(60):
        m_data = data[m*60+iter]
        m_data = m_data[m_data>0]
        k1s.append(np.max(m_data))
    k1s = np.array(k1s)
    mean_k1s.append(np.mean(k1s))
    err_k1s.append(1.96*np.std(k1s)/np.sqrt(60))

t = np.linspace(start = 1e2, stop = 1e6)
plt.plot(t, -0.5 + np.sqrt(1+4*5*(5+1)*t)/2,  color = "red", label = "predicted fit")


def large_deg_dist(N):
    return -0.5 + np.sqrt(1+4*5*(5+1)*N)/2

def chi_sq(observed_distribution, expected_distribution):

    chi2 = np.sum((observed_distribution - expected_distribution)**2/expected_distribution)
    dof = len(observed_distribution) - 2
    return (chi2, sp.chi2.sf(chi2,dof))
print(chi_sq(mean_k1s[:],large_deg_dist(sizes[:])))
slope, intercept, r_value, p_value, std_err = linregress(sizes[2:]**0.5,mean_k1s[2:])
print(slope, intercept,r_value**2, p_value, std_err)
plt.plot(t, slope*np.sqrt(t), label = "line of best fit")
plt.errorbar(sizes,mean_k1s, marker = 'x', yerr = err_k1s, markersize = 1.5, color = "navy" , ls = "none", capsize = 2)     
plt.ylabel("$k_1$")
plt.xlabel("$N$")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig('k1_scaling1.png', dpi=300, bbox_inches = "tight")
plt.show()


