# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.stats import linregress
import tikzplotlib
from logbin import logbin
marker = ["D", "o", "^", "v", "s", "p", "p"]
colors = ["crimson", "rebeccapurple", "royalblue", "mediumseagreen", "lightcoral","darkolivegreen"]
data = np.loadtxt("RA_data_fs.csv", delimiter = ',')

# %%
def rand_dist(k,m):
    output = np.power(m/(m+1), k-m)
    output /= m+1
    return output

scales = [1.07 for i in range(5)]
scales.append(1.2)
sizes = np.array([100,1000,10000,100000,500000,1000000])
xs = []
ys = []
max_k = []
bin_edges = []
for m in range(6):
    m_data = data[60*m:60*(m+1)].flatten()
    m_data = m_data[m_data>0]
    x,y = logbin(m_data.astype(int), scale = scales[m])
    xs.append(np.array(x))
    ys.append(np.array(y))
    max_k.append(np.max(m_data))

# %%
final_errors = []
for m in range(6):
    y_vals = []
    for n in range(60):
        realisation = data[m*60+n,:]
        x_new,y_new = logbin(realisation.astype(int), scale = scales[m], modify = max_k[m])
        print(len(x_new))
        y_vals.append(y_new)
    y_vals = np.array(y_vals)
    final_errors.append(1.96*np.std(y_vals, axis = 0)/np.sqrt(60))

# %%



# %%
plt.rcParams["figure.figsize"] = (5,2.5)
plt.rcParams["font.family"] = "Arial"
plt.rcParams['xtick.labelsize'] = "10"
plt.rcParams['ytick.labelsize'] = "10"
for i in range(6):
    string = f"N={sizes[i]}"
    plt.errorbar(xs[i],ys[i], yerr = final_errors[i], elinewidth =0.6, alpha =0.7,ls = "none", capsize = 1.8, marker = "x", markersize= 4, color = colors[i], label = string)
t = np.linspace(start = 5, stop = 105)
plt.plot(t,rand_dist(t,5), ls = "--", linewidth = 1,alpha = 0.5, label = "infinite limit")
plt.xscale("linear")
plt.yscale("log")
plt.xlabel("$k$", fontsize = 10)
plt.ylabel("$P(k)$", fontsize = 10)
plt.legend(fontsize = 8)
plt.savefig('fixed_m2.png', dpi=300, bbox_inches = "tight")
plt.show()

for i in range(6):
    plt.errorbar(xs[i]/np.log(sizes[i]), ys[i]/rand_dist(xs[i],5), yerr = final_errors[i]/rand_dist(xs[i],5), elinewidth = 1,
    capsize = 1.5, ls = "none",color = colors[i], marker = "x", markersize = 4, alpha =0.7,
         label = f"$N={sizes[i]}$")



plt.xscale("log")
plt.yscale("log")
plt.xlabel("$k/log(N)$", fontsize = 10)
plt.ylabel("$P_{measured}(k)/P_{theory}(k)$", fontsize = 10)
plt.legend(fontsize = 8)
plt.savefig('data_collapse2.png', dpi=300, bbox_inches = "tight")
plt.show()
mean_k1s = []
err_k1s = []

def large_deg(m, N):
    output = np.log(N)/np.log((m+1)/m)
    output += m    

    return output

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
plt.plot(t, large_deg(5,t),  color = "red", label = "predicted fit")


slope, intercept, r_value, p_value, std_err = linregress(np.log(sizes[2:]),mean_k1s[2:])
print(slope, intercept,r_value**2, p_value, std_err)
plt.plot(t, intercept+slope*np.log(t), label = "line of best fit")
plt.errorbar(sizes,mean_k1s, marker = 'x', yerr = err_k1s, markersize = 1.5, color = "navy" , ls = "none", capsize = 2)     
plt.ylabel("$k_1$")
plt.xlabel("$N$")
plt.legend()
plt.minorticks_on()
plt.xscale("log")
plt.yscale("linear")
plt.savefig('k1_scaling2.png', dpi=300, bbox_inches = "tight")
plt.show()


