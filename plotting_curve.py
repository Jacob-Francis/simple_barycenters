
# actually maybe it's over a few epsilon? Or a few rho? but unified in type for sure.
# input [0] ROOT_FILE+"pkl/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
# input [1]  ROOT_FILE+"pkl/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",


# load data
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np


# should abstract times and members from global config
times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
debiasing = [True, False]
epsilons = [0.05, 0.005, 0.001]
rhos = [1.0, 0.1, 0.01]
# colour blind friendly colors
colours = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#different marker for aprox
markers = dict(kl='o', balanced='s', tv='^')
aprox_types = ['kl', 'balanced', 'tv']

ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"

debiasing = 'True'
epsilon = 0.005
aprox_type = 'balanced'
count = 0
rho = 1.0

for rho in rhos:
    for epsilon in epsilons:
        for debiasing in ['True', 'False']:
            count = 0
            min_val = float('inf')
            max_val = float('-inf')
            fig, ax = plt.subplots(1, 2, figsize=(8, 5))

            for aprox_type in aprox_types:
                # actually maybe it's over a few epsilon? Or a few rho? but unified in type for sure.
                bary_file = ROOT_FILE+f"pkl/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
                obs_file = ROOT_FILE+f"pkl/observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}.pkl"

                with open(bary_file, 'rb') as f:
                    barycentre_costs = pickle.load(f)

                with open(obs_file, 'rb') as f:
                    observation_costs = pickle.load(f)

                # calculate costs
                spread = np.zeros(len(times))
                error = np.zeros(len(times))
                for k, t in enumerate(times):
                    temp = np.zeros(len(members)+1)

                    for i in range(len(members)+1):
                        temp[i] = observation_costs[t][f'debias_{i}']
                    error[k] = (observation_costs[t]['bias'] - 1/(len(members)+1) * temp.sum()).item()

                    temp = np.zeros(len(members)+1)
                    for i in range(len(members)+1):
                        temp[i] = barycentre_costs[t][f'debias_{i}']
                    spread[k] = (barycentre_costs[t]['bias'] - 1/(len(members)+1) * temp.sum()).item()

                max_val = max(max_val, max(spread.max(), error.max()))
                min_val = min(min_val, min(spread.min(), error.min()))

                # plot costs
                # print(times, spread, error)
                ax[0].plot(times, spread, '--', marker=markers[aprox_type], label='spread', color=colours[count])
                # empty shape
                ax[0].plot(times, error, '--', marker=markers[aprox_type], markerfacecolor='none', label='error', color=colours[count])  
                ax[0].plot([], [], '-', label=f'aprox={aprox_type}', color=colours[count])
            
                ax[1].plot(spread, error, '--', marker=markers[aprox_type], label=f'aprox={aprox_type}', color=colours[count])
                count += 1
                
                # linear trend line
            ax[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='linear trend')

            ax[0].set_title('Barycentre MMUOT Cost over Time')
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('Cost')
            ax[0].legend()
            ax[1].set_title('Barycentre MMUOT Cost vs Observation MMUOT Cost')
            ax[1].set_xlabel('Barycentre MMUOT Cost (spread)')
            ax[1].set_ylabel('Observation MMUOT Cost (error)')

            ax[1].legend()

            plt.savefig('spread_curves/spread_skill_curve_rho_{}_eps_{}_debiasing_{}.png'.format(rho, epsilon, debiasing))