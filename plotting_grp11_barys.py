
# load data
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
from matplotlib.lines import Line2D
from utils import calculate_true_spread_skill

# Get arguments after script name
args = sys.argv[1:]

# Convert to list of ints
data_sets = list(map(int, args))

epsilon = 0.001 
aprox_types = ['kl', 'tv'] # 'balanced',
ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"
debiasing=True

colours = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
markers = dict(kl='o', balanced='s', tv='^')

plt.rcParams.update({"font.size": 14})

# ##################################################################
#          PLotting set1 and set 2 examples for 'zeros' robustness
# ########################################################
aprox_type='kl'
rho=1.0

fig_ex, ax_ex = plt.subplots(1, 4, figsize=(9*4, 7))

bary_list = [] # for colourbar
cblabel = [
    'Set23 16 | $ \rho = 1 $',
    'Set23 32 | $ \rho = 1 $',
    'Set23 16 | $ \rho = 0.001 $',
    'Set23 32 | $ \rho = 0.001 $'
]
for j, (d, k, rho) in enumerate([(23, 3, 1.0), (23, 4,1.0), (23, 3,0.001), (23, 4,0.001)]):
    act_bary = ROOT_FILE+f"pkl/{d}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
    ax = ax_ex[j]
    with open(act_bary, 'rb') as f:
        bary_output = pickle.load(f)

    # costs
    cost_file = ROOT_FILE+f"pkl/{d}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
    obs_se = ROOT_FILE+f"pkl/{d}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
    
    with open(obs_se, 'rb') as f:
        obs_se_costs = pickle.load(f)

    with open(cost_file, 'rb') as f:
        se_costs = pickle.load(f)

    print('here', len(se_costs))
    print('here', se_costs.keys())
    bary_se_spread = se_costs[k][0]['total_cost']
    obs_se_error = obs_se_costs[k][0]['total_cost']
    
    data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{d}.pkl"

    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    # grid
    X = torch.meshgrid(*dataset['grid'], indexing='ij')
    X = torch.stack(X, dim=-1)

    # ensemble mean
    t = dataset['times'][k]
    ensemble_mean = np.mean([f[0] for f in dataset[t]['forecasts']], axis=0)
    obs_mass = sum(dataset[t]['observation'][0].flatten())/200**2

    # plot
    barycentre = bary_output[k]

    ax.set_facecolor("#f9f6f1")
    barycentre = barycentre.reshape(200,  200).detach().cpu().numpy()
    img = ax.pcolormesh(
        X[:, :, 0],
        X[:, :, 1],
        np.ma.masked_where(barycentre <= 1e-40, barycentre),
        cmap='Blues',
        shading='auto'
    )
    plt.colorbar(img, ax=ax,     orientation='horizontal',
    fraction=0.05,
    pad=0.08, label=cblabel[j])  # Add colorbar for the barycentre

    # add support contour
    mask = ensemble_mean>0
    ax.contour(
        X[:, :, 0],
        X[:, :, 1],
        mask.astype(int),    
        levels=[0.5],
        colors='#b98224',
        linestyles='dashdot',
        linewidths=2,
    )
    ax.set_title(f"mass ratio: {barycentre.sum().item()/obs_mass:.4g} error: {obs_se_error:.4g}, spread: {bary_se_spread:.4g}")
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

fig_ex.savefig('/home/jjf817/PhD_jobs/simple_barycentres/spread_curves/grp11/grp11_bary_ex_rho1_kl.png', bbox_inches='tight', dpi=200)
plt.close()

fig_ex, ax_ex = plt.subplots(1, 4, figsize=(9*4, 7))
