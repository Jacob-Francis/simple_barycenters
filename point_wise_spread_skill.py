import pwbarycentres as pwb
from graph_dp import SinkhornDataProcessor
from utils import *

import pickle
import torch

import os 

with open(snakemake.input[0], 'rb') as f:
    dictionary_in_time = pickle.load(f)

times =dictionary_in_time['times'] 
members=dictionary_in_time['members']
X = dictionary_in_time['grid']

# -------------------------------------------
# Compute the skill scores:
# -------------------------------------------

def ensemble_mean_error(mean, observation):
    M = np.prod(observation.shape)
    return np.sqrt((mean - observation)**2)

def summary_ensemble_mean_error(mean, observation):
    temp = ensemble_mean_error(mean, observation)
    return np.mean(temp), np.std(temp)

def spread(densities, mean):
    batch_densities = np.stack(densities, axis=-1)
    M = np.prod(mean.shape)
    N = len(densities)
    return np.sqrt(((batch_densities.reshape(-1,N) - mean.reshape(-1, 1))**2).sum(axis=-1)/(N-1))

def summary_spread(densities, mean):
    temp = spread(densities, mean)
    return np.mean(temp), np.std(temp)

mu_e_list = []
sd_e_list = []
mu_s_list = []
st_s_list = []

ensemble_mean_list = []
for k,t in enumerate(times):
    size = len(members)
    ensemble_mean_l2 =  np.zeros_like(dictionary_in_time[t]['forecasts'][0][0])
    for mem in members:
        ensemble_mean_l2 += dictionary_in_time[t]['forecasts'][mem][0]
    ensemble_mean_l2 /= size
    ensemble_mean_list.append(ensemble_mean_l2)

for k, t in enumerate(times):
    ensemble_mean_l2 = ensemble_mean_list[k]
    mu_e, sd_e = summary_ensemble_mean_error(ensemble_mean_l2, dictionary_in_time[t]['observation'][0])
    ensemble_members = [dictionary_in_time[t]['forecasts'][mem][0] for mem in members]
    mu_s, st_s = summary_spread(ensemble_members, ensemble_mean_l2)
    mu_e_list.append(mu_e.item())
    sd_e_list.append(sd_e.item())
    mu_s_list.append(mu_s.item())
    st_s_list.append(st_s.item())    


# ----------------------------------------------------
# plot
# ----------------------------------------------------
time_steps = np.arange(len(mu_e_list)) 
mu_e = np.array(mu_e_list)
sd_e = np.array(sd_e_list)

mu_s = np.array(mu_s_list)
sd_s = np.array(st_s_list)

# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(8,5))

# Plot with error bars
axes[0].errorbar(time_steps, mu_e, yerr=sd_e, fmt='o-', label="Error", capsize=5)
axes[0].errorbar(time_steps, mu_s, yerr=sd_s, fmt='s-', label="Spread", capsize=5)
axes[0].axhline(0)

# Labels and legend
axes[0].set_xlabel("Time Steps")
axes[0].set_ylabel("Mean Value")
axes[0].set_title("Mean with Standard Deviation (Error Bars)")
axes[0].legend()
axes[0].grid(True)

# Plot with error bars
axes[1].plot(mu_e, mu_s, '.-')
# linear line
max_val = max(mu_e.max(), mu_s.max())   
axes[1].plot([0, max_val], [0, max_val], 'k--')
axes[1].set_xlabel("Error")
axes[1].set_ylabel("Spread")
axes[1].grid(True)
    
plt.tight_layout()
plt.savefig(snakemake.output[0])