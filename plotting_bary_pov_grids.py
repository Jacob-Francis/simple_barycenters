
# actually maybe it's over a few epsilon? Or a few rho? but unified in type for sure.
# input [0] ROOT_FILE+"pkl/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
# input [1]  ROOT_FILE+"pkl/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",


# load data
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys

# Get arguments after script name
args = sys.argv[1:]

# Convert to list of ints
data_sets = list(map(int, args))

# should abstract times and members from global config
epsilons = [0.001] #, 0.0005] #0.01,
aprox_types = ['kl', 'tv'] # 'balanced',
data_sets = [k for k in range(1, 10)] + [11] + [k for k in range(15,25+1)] # 10 desn't run, 12, 13, 14
ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"

###################################### plotting rho : 1.0 ,0.001, kl and tv, debiased only ######################################
###################################### plotting rho : 10.0 ,0.01, kl and tv, debiased only ######################################

# colour blind friendly colors
colours = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#different marker for aprox
markers = dict(kl='o', balanced='s', tv='^')


for data_set in data_sets:
    data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{data_set}.pkl"

    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)
    times = dataset['times']
    members = dataset['members']
    X = torch.meshgrid(*dataset['grid'], indexing='ij')
    X = torch.stack(X, dim=-1)

    for k,t in enumerate(times):
        print(f"Time: {t}")
        obs_mass = sum(dataset[t]['observation'][0].flatten())/200**2
        # ensemble mean
        ensemble_mean = np.mean([f[0] for f in dataset[t]['forecasts']], axis=0)
        print('Number members', len(dataset[t]['forecasts']))
        for epsilon in epsilons:
            count = 0
            fig, ax = plt.subplots(1, 4, figsize=(8*4, 5))

            plt.rcParams.update({"font.size": 14})

            for l, debiasing in enumerate([True]):
                axes = [ax[0], ax[1], ax[2], ax[3]]
                for j, (rho, aprox_type) in enumerate([(1.0, 'kl'), (0.001, 'kl'), (1.0, 'tv'),  (0.001, 'tv')]):
                # for j, (rho, aprox_type) in enumerate([(10.0, 'kl'), (0.01, 'kl'), (10.0, 'tv'),  (0.01, 'tv')]):

                    # actually maybe it's over a few epsilon? Or a few rho? but unified in type for sure.

                    # bary_file = ROOT_FILE+f"pkl/{data_set}/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
                    # obs_file = ROOT_FILE+f"pkl/{data_set}/observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}.pkl"
                    cost_file = ROOT_FILE+f"pkl/{data_set}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
                    act_bary = ROOT_FILE+f"pkl/{data_set}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
                    obs_se = ROOT_FILE+f"pkl/{data_set}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"

                    with open(act_bary, 'rb') as f:
                        bary_output = pickle.load(f)
                    
                    with open(obs_se, 'rb') as f:
                        obs_se_costs = pickle.load(f)

                    with open(cost_file, 'rb') as f:
                        se_costs = pickle.load(f)

                    bary_se_spread = se_costs[k][0]['total_cost']
                    obs_se_error = obs_se_costs[k][0]['total_cost']

                    barycentre = bary_output[k]

                    axes[j].set_facecolor("#f9f6f1")
                    bary = barycentre.reshape(200,  200).detach().cpu().numpy()
                    img = axes[j].pcolormesh(
                        X[:, :, 0],
                        X[:, :, 1],
                        np.ma.masked_where(bary <= 1e-40, bary),
                        cmap='Blues',
                        shading='auto'
                    )
                    plt.colorbar(img, ax=axes[j], label='Barycentre Density')  # Add colorbar for the barycentre

                    # add support contour
                    mask = ensemble_mean>0
                    axes[j].contour(
                        X[:, :, 0],
                        X[:, :, 1],
                        mask.astype(int),    
                        levels=[0.5],
                        colors='#b98224',
                        linestyles='dashdot',
                        linewidths=2,
                    )
                    axes[j].set_title(f"mass ratio: {barycentre.sum().item()/obs_mass:.4g} obs se error: {obs_se_error:.4g}, se spread: {bary_se_spread:.4g}", fontsize=12)

            ax[0].set_ylabel("Debiased", rotation=90, fontsize=20, fontweight='bold')

            column_headers = ['rho: 1.0, kl', 'rho: 0.001, kl', 'rho: 1.0, tv', 'rho: 0.001, tv']
            # column_headers = ['rho: 10.0, kl', 'rho: 0.01, kl', 'rho: 10.0, tv', 'rho: 0.01, tv']

            for col in range(4):
                # Get the position of the top subplot in that column
                pos = ax[col].get_position()
                
                # Place text slightly above it
                fig.text(
                    pos.x0 + pos.width / 2,
                    pos.y1 + 0.04,
                    column_headers[col],
                    ha='center',
                    va='bottom',
                    fontsize=20,
                    fontweight='bold'
                )

            try:
                plt.savefig(f'barycentre_grid_pov/{data_set}/{data_set}_bary_pov_time{t}_eps{str(epsilon)[2:]}.png', dpi=200)
            # if not folder dataset exists, create it and save
            except FileNotFoundError:
                import os
                os.makedirs(f'barycentre_grid_pov/{data_set}', exist_ok=True)
                plt.savefig(f'barycentre_grid_pov/{data_set}/{data_set}_bary_pov_time{t}_eps{str(epsilon)[2:]}.png')
            plt.close('all')

print("Done plotting!")