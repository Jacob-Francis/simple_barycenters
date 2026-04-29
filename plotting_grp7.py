
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

# # ##################################################################
# #          PLotting set1 and set 2 examples for 'zeros' robustness
# # ########################################################
# aprox_type='kl'
# rho=1.0

# fig_ex, ax_ex = plt.subplots(1, 4, figsize=(9*4, 7))

# bary_list = [] # for colourbar
# cblabel = [
#     'Set1 1',
#     'Set1 2',
#     'Set2 1',
#     'Set2 2'
# ]
# for j, (d,k) in enumerate([(1,0), (1,1), (2,0), (2,1)]):
#     act_bary = ROOT_FILE+f"pkl/{d}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
#     ax = ax_ex[j]
#     with open(act_bary, 'rb') as f:
#         bary_output = pickle.load(f)

#     # costs
#     cost_file = ROOT_FILE+f"pkl/{d}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
#     obs_se = ROOT_FILE+f"pkl/{d}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
    
#     with open(obs_se, 'rb') as f:
#         obs_se_costs = pickle.load(f)

#     with open(cost_file, 'rb') as f:
#         se_costs = pickle.load(f)

#     bary_se_spread = se_costs[k][0]['total_cost']
#     obs_se_error = obs_se_costs[k][0]['total_cost']
    
#     data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{d}.pkl"

#     with open(data_file, 'rb') as f:
#         dataset = pickle.load(f)

#     # grid
#     X = torch.meshgrid(*dataset['grid'], indexing='ij')
#     X = torch.stack(X, dim=-1)

#     # ensemble mean
#     t = dataset['times'][k]
#     ensemble_mean = np.mean([f[0] for f in dataset[t]['forecasts']], axis=0)
#     obs_mass = sum(dataset[t]['observation'][0].flatten())/200**2

#     # plot
#     barycentre = bary_output[k]

#     ax.set_facecolor("#f9f6f1")
#     barycentre = barycentre.reshape(200,  200).detach().cpu().numpy()
#     img = ax.pcolormesh(
#         X[:, :, 0],
#         X[:, :, 1],
#         np.ma.masked_where(barycentre <= 1e-40, barycentre),
#         cmap='Blues',
#         shading='auto'
#     )
#     plt.colorbar(img, ax=ax,     orientation='horizontal',
#     fraction=0.05,
#     pad=0.08, label=cblabel[j])  # Add colorbar for the barycentre

#     # add support contour
#     mask = ensemble_mean>0
#     ax.contour(
#         X[:, :, 0],
#         X[:, :, 1],
#         mask.astype(int),    
#         levels=[0.5],
#         colors='#b98224',
#         linestyles='dashdot',
#         linewidths=2,
#     )
#     ax.set_title(f"mass ratio: {barycentre.sum().item()/obs_mass:.4g} error: {obs_se_error:.4g}, spread: {bary_se_spread:.4g}")
#     # ax.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig_ex.savefig('/home/jjf817/PhD_jobs/simple_barycentres/spread_curves/grp7/grp7_bary_ex_rho1_kl.png', bbox_inches='tight', dpi=200)
# plt.close()

# fig_ex, ax_ex = plt.subplots(1, 4, figsize=(9*4, 7))

# bary_list = [] # for colourbar
# cblabel = [
#     'Set3 1',
#     'Set3 4',
#     'Set4 1',
#     'Set4 4'
# ]
# for j, (d,k) in enumerate([(3,1), (3,4), (4,1), (4,4)]):
#     act_bary = ROOT_FILE+f"pkl/{d}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
#     ax = ax_ex[j]
#     with open(act_bary, 'rb') as f:
#         bary_output = pickle.load(f)

#     # costs
#     cost_file = ROOT_FILE+f"pkl/{d}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
#     obs_se = ROOT_FILE+f"pkl/{d}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
    
#     with open(obs_se, 'rb') as f:
#         obs_se_costs = pickle.load(f)

#     with open(cost_file, 'rb') as f:
#         se_costs = pickle.load(f)

#     bary_se_spread = se_costs[k][0]['total_cost']
#     obs_se_error = obs_se_costs[k][0]['total_cost']
    
#     data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{d}.pkl"

#     with open(data_file, 'rb') as f:
#         dataset = pickle.load(f)

#     # grid
#     X = torch.meshgrid(*dataset['grid'], indexing='ij')
#     X = torch.stack(X, dim=-1)

#     # ensemble mean
#     t = dataset['times'][k]
#     ensemble_mean = np.mean([f[0] for f in dataset[t]['forecasts']], axis=0)
#     obs_mass = sum(dataset[t]['observation'][0].flatten())/200**2

#     # plot
#     barycentre = bary_output[k]

#     ax.set_facecolor("#f9f6f1")
#     barycentre = barycentre.reshape(200,  200).detach().cpu().numpy()
#     img = ax.pcolormesh(
#         X[:, :, 0],
#         X[:, :, 1],
#         np.ma.masked_where(barycentre <= 1e-40, barycentre),
#         cmap='Blues',
#         shading='auto'
#     )
#     plt.colorbar(img, ax=ax,     orientation='horizontal',
#     fraction=0.05,
#     pad=0.08, label=cblabel[j])  # Add colorbar for the barycentre

#     # add support contour
#     mask = ensemble_mean>0
#     ax.contour(
#         X[:, :, 0],
#         X[:, :, 1],
#         mask.astype(int),    
#         levels=[0.5],
#         colors='#b98224',
#         linestyles='dashdot',
#         linewidths=2,
#     )
#     ax.set_title(f"mass ratio: {barycentre.sum().item()/obs_mass:.4g} error: {obs_se_error:.4g}, spread: {bary_se_spread:.4g}")
#     # ax.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig_ex.savefig('/home/jjf817/PhD_jobs/simple_barycentres/spread_curves/grp7/grp7_ellipsebary_ex_rho1_kl.png', bbox_inches='tight', dpi=200)

# ##################################################################
# plot spread skill true and Se for set3 and set4 - without decomposition and fixed rho
# ##################################################################

# colour blind friendly colors
colours = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#different marker for aprox
markers = dict(kl='o', balanced='s', tv='^')
aprox_colors = dict(kl='#377eb8', balanced='#4daf4a', tv='#ff7f00')
cost_colours = dict(
    spread='#377eb8', 
    skill='#ff7f00', 
    se_cost='#4daf4a',
    se_cost_bias='#f781bf',
    mass_ratio='#a65628'
    )
line_style={
    11:'--',
    12:':',
    13:'-.',
    14:(0, (3, 5, 1, 5, 1, 5)),
}

fig_ss, ax_ss = plt.subplots(1, 2, figsize=(9*2, 7))

for data_set in [11,12,13,14]:
    data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{data_set}.pkl"

    try:
        with open(data_file, 'rb') as f:
            obvs_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {data_file}")
        list_of_files_not_found.append(data_file)
        continue # try next data set

    times = obvs_dict['times']
    members = obvs_dict['members']
    M = len(members)

    if 'weights' in obvs_dict:
        weights = obvs_dict['weights']
    else:   
        weights = None

    count = 0
    l = 0
    rho = 1.0
    debiasing = True
   
    max_val = {
        'TL': float('-inf'),
        'TR': float('-inf'),
    } 
    min_val = {
        'TL': float('inf'),
        'TR': float('inf'),
    }
    for aprox_type in aprox_types:
        cost_file = ROOT_FILE+f"pkl/{data_set}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
        act_bary = ROOT_FILE+f"pkl/{data_set}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
        obs_cost = ROOT_FILE+f"pkl/{data_set}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"

        with open(obs_cost, 'rb') as f:
            obs_se_costs = pickle.load(f)

        with open(act_bary, 'rb') as f:
            bary_output = pickle.load(f)
        
        with open(cost_file, 'rb') as f:
            se_costs = pickle.load(f)

        # calculate costs
        spread = np.zeros(len(times))
        mass_ratio = np.zeros(len(times))
        se_spread = np.zeros(len(times))
        se_spread_decomp = np.zeros((M, len(times)))
        obs_se = np.zeros(len(times))
        obs_se_decomp = np.zeros((M, len(times)))
        
        # Consistuent_terms
        obs_dict = {
            'transport_cost': np.zeros((M, len(times)), dtype=np.float64),
            'marginal_penalty': np.zeros((M, len(times)), dtype=np.float64),
        }
        
        bary_dict = {
            'transport_cost': np.zeros((M, len(times)), dtype=np.float64),
            'marginal_penalty': np.zeros((M, len(times)), dtype=np.float64),
        }
        
        def gathering_costs(costs, dict_to_fill, t):
            # print(obs_se_costs[k][0]['subbreakdown'][(0, 1)].keys())
            #dict_keys(['dual_term1', 'dual_term2', 'dual_term3', 'dual_term4', 'primal_c_pi', 'primal_divergence_term', 'primal_entropy', 'uot_mu_mu', 'weight'])
            for i, node_keys in enumerate(costs['subbreakdown'].keys()):
                if isinstance(node_keys, tuple):
                    node_dict = costs['subbreakdown'][node_keys]
                    dict_to_fill['transport_cost'][i, t] = node_dict['primal_c_pi']
                    dict_to_fill['marginal_penalty'][i, t] = node_dict['primal_divergence_term']


        for k, t in enumerate(times):
            # barycentre cost - spread
            se_spread[k] = se_costs[k][0]['total_cost']
            # debiasing constants handelling within
            
            if weights is not None:
                print(f"TOCHECK: Using weights for time {t}: {weights[k]}")
                w = np.array([1/we for we in weights[k]]) # to rescale back up
            else:
                w = np.ones(M) * M  # uniform weights if not provided

            if debiasing:
                se_per_data = np.stack(se_costs[k][0]['unbalanced_sinkhorn_terms'])*w + se_costs[k][0]['debiasing_term'] - np.stack(se_costs[k][0]['uot_mu_mu_terms'])*w/2
            else:
                se_per_data = np.stack(se_costs[k][0]['unbalanced_sinkhorn_terms'])*w
            se_spread_decomp[:, k] = se_per_data

            # observation cost - error
            if debiasing:
                se_per_data = np.stack(obs_se_costs[k][0]['unbalanced_sinkhorn_terms'])*w + obs_se_costs[k][0]['debiasing_term'] - np.stack(obs_se_costs[k][0]['uot_mu_mu_terms'])*w/2
            else:
                se_per_data = np.stack(obs_se_costs[k][0]['unbalanced_sinkhorn_terms'])*w

            obs_se[k] = obs_se_costs[k][0]['total_cost']
            obs_se_decomp[:, k] = se_per_data

            # fill dicts
            gathering_costs(obs_se_costs[k][0], obs_dict, k)
            gathering_costs(se_costs[k][0], bary_dict, k)

            # mass ratio
            barycentre = bary_output[k]
            observation = obvs_dict[t]['observation'][0]
            mass_ratio[k] = barycentre.sum().item()/observation.sum().item()
        
        max_val['TR'] = max(max_val['TR'], max(obs_se.max(), se_spread.max()))
        min_val['TR'] = min(min_val['TR'], min(obs_se.min(), se_spread.min()))


        # zero check
        labels = ['spread', 'error', 'spread decomp', 'error decomp']
        for k, values in enumerate([se_spread, obs_se,  se_spread_decomp, obs_se_decomp]):
            if np.any(values <= 0):
                print(f"Warning {data_set}:  -ve in {labels[k]} {aprox_type} at time {t}. average: {np.mean(values[values <= 0])}")

        # plot as simple points per time first
        # top left normal spread skill
        # later
        # top right full se
        ax_ss[1].plot(se_spread, obs_se,  marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[data_set], markersize=12, markerfacecolor='none')

    # Now plot independently
    mu_e, sd_e, mu_s, sd_s = calculate_true_spread_skill(
        ROOT_FILE + f"ensemble_data/ensemble_dataset_{data_set}.pkl"
    )
    min_val['TL'] = min(mu_e.min(), mu_s.min())
    max_val['TL'] = max(mu_e.max(), mu_s.max())
    ax_ss[0].plot(mu_s, mu_e,
                    linestyle=line_style[data_set],
                    marker='x',
                    color='black',
                    label='true spread-skill',
                    markersize=12)

    # plot min/max linear lines
    keys = ['TL', 'TR']
    for i in range(2):
            ax_ss[i].plot(
                [min_val[keys[i]], max_val[keys[i]]],
                [min_val[keys[i]], max_val[keys[i]]],
                linestyle='--',
                color='grey',
                markersize=15,
                alpha=0.5
            )
    
    # save for different rho
    ax_ss[0].set(xlabel='Spread', ylabel='Error')
    ax_ss[1].set(xlabel=r'Sinkhorn Spread', ylabel=r'Sinkhorn Error')

from matplotlib.ticker import MaxNLocator
ax_ss[1].xaxis.set_major_locator(MaxNLocator(5))

legend_elements = [
            # ---- Divergence marker meaning ----
            Line2D([0], [0],
                marker=markers['kl'], color=aprox_colors['kl'],
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='KL'
            ),
            Line2D([0], [0],
                marker=markers['tv'], color=aprox_colors['tv'],
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='TV'
            ),
            Line2D([0], [0],
                marker='x', color='black',
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='True'
            ),

            # ---- spread-skill diagonal ----
            Line2D([0], [0],
                color='black', linestyle='--',
                label='Ideal spread = error'
            ),
            *[Line2D([0], [0],
                color='black', linestyle=line_style[key],
                label=f'Set{key}'
            ) for key in line_style.keys()],
        ]

# add legend to fig_decomp
fig_ss.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
    )

fig_ss.savefig(f'spread_curves/grp7/grp7_ss_1114_rho{rho}.png', dpi=200, bbox_inches='tight')

plt.close('all')

# ########################################################
# plot decompositiont
fig_ss, ax_ss = plt.subplots(1, 2, figsize=(9*2, 7))

for data_set in [11, 12, 13, 14]:
    data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{data_set}.pkl"

    try:
        with open(data_file, 'rb') as f:
            obvs_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {data_file}")
        list_of_files_not_found.append(data_file)
        continue # try next data set

    times = obvs_dict['times']
    members = obvs_dict['members']
    M = len(members)

    if 'weights' in obvs_dict:
        weights = obvs_dict['weights']
    else:   
        weights = None

    count = 0
    l = 0
    rho = 1.0
    debiasing = True
   
    max_val = {
        'TL': float('-inf'),
        'TR': float('-inf'),
    } 
    min_val = {
        'TL': float('inf'),
        'TR': float('inf'),
    }
    for aprox_type in aprox_types:
        cost_file = ROOT_FILE+f"pkl/{data_set}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
        act_bary = ROOT_FILE+f"pkl/{data_set}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
        obs_cost = ROOT_FILE+f"pkl/{data_set}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"

        with open(obs_cost, 'rb') as f:
            obs_se_costs = pickle.load(f)

        with open(act_bary, 'rb') as f:
            bary_output = pickle.load(f)
        
        with open(cost_file, 'rb') as f:
            se_costs = pickle.load(f)

        # calculate costs
        spread = np.zeros(len(times))
        mass_ratio = np.zeros(len(times))
        se_spread = np.zeros(len(times))
        se_spread_decomp = np.zeros((M, len(times)))
        obs_se = np.zeros(len(times))
        obs_se_decomp = np.zeros((M, len(times)))
        
        # Consistuent_terms
        obs_dict = {
            'transport_cost': np.zeros((M, len(times)), dtype=np.float64),
            'marginal_penalty': np.zeros((M, len(times)), dtype=np.float64),
        }
        
        bary_dict = {
            'transport_cost': np.zeros((M, len(times)), dtype=np.float64),
            'marginal_penalty': np.zeros((M, len(times)), dtype=np.float64),
        }
        
        def gathering_costs(costs, dict_to_fill, t):
            # print(obs_se_costs[k][0]['subbreakdown'][(0, 1)].keys())
            #dict_keys(['dual_term1', 'dual_term2', 'dual_term3', 'dual_term4', 'primal_c_pi', 'primal_divergence_term', 'primal_entropy', 'uot_mu_mu', 'weight'])
            for i, node_keys in enumerate(costs['subbreakdown'].keys()):
                if isinstance(node_keys, tuple):
                    node_dict = costs['subbreakdown'][node_keys]
                    dict_to_fill['transport_cost'][i, t] = node_dict['primal_c_pi']
                    dict_to_fill['marginal_penalty'][i, t] = node_dict['primal_divergence_term']


        for k, t in enumerate(times):
            # barycentre cost - spread
            se_spread[k] = se_costs[k][0]['total_cost']
            # debiasing constants handelling within
            
            if weights is not None:
                print(f"TOCHECK: Using weights for time {t}: {weights[k]}")
                w = np.array([1/we for we in weights[k]]) # to rescale back up
            else:
                w = np.ones(M) * M  # uniform weights if not provided

            if debiasing:
                se_per_data = np.stack(se_costs[k][0]['unbalanced_sinkhorn_terms'])*w + se_costs[k][0]['debiasing_term'] - np.stack(se_costs[k][0]['uot_mu_mu_terms'])*w/2
            else:
                se_per_data = np.stack(se_costs[k][0]['unbalanced_sinkhorn_terms'])*w
            se_spread_decomp[:, k] = se_per_data

            # observation cost - error
            if debiasing:
                se_per_data = np.stack(obs_se_costs[k][0]['unbalanced_sinkhorn_terms'])*w + obs_se_costs[k][0]['debiasing_term'] - np.stack(obs_se_costs[k][0]['uot_mu_mu_terms'])*w/2
            else:
                se_per_data = np.stack(obs_se_costs[k][0]['unbalanced_sinkhorn_terms'])*w

            obs_se[k] = obs_se_costs[k][0]['total_cost']
            obs_se_decomp[:, k] = se_per_data

            # fill dicts
            gathering_costs(obs_se_costs[k][0], obs_dict, k)
            gathering_costs(se_costs[k][0], bary_dict, k)

            # mass ratio
            barycentre = bary_output[k]
            observation = obvs_dict[t]['observation'][0]
            mass_ratio[k] = barycentre.sum().item()/observation.sum().item()
        
        # BL: bary_dict['transport_cost'].mean(axis=0), obs_dict['transport_cost'].mean(axis=0)
        max_val['TL'] = max(max_val['TL'], max(obs_dict['transport_cost'].mean(axis=0).max(), bary_dict['transport_cost'].mean(axis=0).max()))
        min_val['TL'] = min(min_val['TL'], min(obs_dict['transport_cost'].mean(axis=0).min(), bary_dict['transport_cost'].mean(axis=0).min()))

        # BR: bary_dict['marginal_penalty'].mean(axis=0), obs_dict['marginal_penalty'].mean(axis=0)
        max_val['TR'] = max(max_val['TR'], max(obs_dict['marginal_penalty'].mean(axis=0).max(), bary_dict['marginal_penalty'].mean(axis=0).max()))
        min_val['TR'] = min(min_val['TR'], min(obs_dict['marginal_penalty'].mean(axis=0).min(), bary_dict['marginal_penalty'].mean(axis=0).min()))

        # zero check
        labels = ['spread', 'error', 'spread decomp', 'error decomp']
        for k, values in enumerate([se_spread, obs_se,  se_spread_decomp, obs_se_decomp]):
            if np.any(values <= 0):
                print(f"Warning {data_set}:  -ve in {labels[k]} {aprox_type} at time {t}. average: {np.mean(values[values <= 0])}")

        # plot as simple points per time first
        # top left normal spread skill
        # later
        # top right full se
        # ax_ss[1].plot(se_spread, obs_se,  marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[data_set], markersize=12, markerfacecolor='none')
        ax_ss[0].plot(bary_dict['transport_cost'].mean(axis=0), obs_dict['transport_cost'].mean(axis=0), marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[data_set], markersize=12, markerfacecolor='none')
        # bottom right marginal penalty decomp
        ax_ss[1].plot(bary_dict['marginal_penalty'].mean(axis=0), obs_dict['marginal_penalty'].mean(axis=0), marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[data_set], markersize=12, markerfacecolor='none')

    from matplotlib.ticker import MaxNLocator
    ax_ss[1].xaxis.set_major_locator(MaxNLocator(5))

    # plot min/max linear lines
    keys = ['TL', 'TR']
    for i in range(2):
            ax_ss[i].plot(
                [min_val[keys[i]], max_val[keys[i]]],
                [min_val[keys[i]], max_val[keys[i]]],
                linestyle='--',
                color='grey',
                markersize=15,
                alpha=0.5
            )
    
    # save for different rho
    ax_ss[0].set(xlabel='Spread', ylabel='Error')
    ax_ss[1].set(xlabel=r'Sinkhorn Spread', ylabel=r'Sinkhorn Error')
from matplotlib.ticker import MaxNLocator
ax_ss[0].xaxis.set_major_locator(MaxNLocator(5))

legend_elements = [
            # ---- Divergence marker meaning ----
            Line2D([0], [0],
                marker=markers['kl'], color=aprox_colors['kl'],
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='KL'
            ),
            Line2D([0], [0],
                marker=markers['tv'], color=aprox_colors['tv'],
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='TV'
            ),
            Line2D([0], [0],
                marker='x', color='black',
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='True'
            ),

            # ---- spread-skill diagonal ----
            Line2D([0], [0],
                color='black', linestyle='--',
                label='Ideal spread = error'
            ),
            *[Line2D([0], [0],
                color='black', linestyle=line_style[key],
                label=f'Set{key}'
            ) for key in line_style.keys()],
        ]

# add legend to fig_decomp
fig_ss.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
    )

fig_ss.savefig(f'spread_curves/grp7/grp7_ss_decomp_1114_rho{rho}.png', dpi=200, bbox_inches='tight')

plt.close('all')