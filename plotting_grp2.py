
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
    10.0:':',
    1.0:'-.',
    0.01:'--',
    0.001:'-'
}

fig_ss, ax_ss = plt.subplots(1, 2, figsize=(9*2, 7))

toggle=True

for data_set in [5]:
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
    for rho in [10.0, 1.0, 0.01, 0.001]:
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
                    #flip sign assuming small
                    assert np.mean(values[values <= 0]) < 1e-5, np.mean(values[values <= 0]) # small enough for set tolerance

                    values[values<= 0] = np.abs(values[values<= 0])
        
            # plot as simple points per time first
            # top left normal spread skill
            # later
            # top right full se
            ax_ss[1].plot(se_spread, obs_se,  marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[rho], markersize=12, markerfacecolor='none')

    # Now plot independently
    if toggle:
        mu_e, sd_e, mu_s, sd_s = calculate_true_spread_skill(
            ROOT_FILE + f"ensemble_data/ensemble_dataset_{data_set}.pkl"
        )
        min_val['TL'] = min(mu_e.min(), mu_s.min())
        max_val['TL'] = max(mu_e.max(), mu_s.max())
        ax_ss[0].plot(mu_s, mu_e,
                        linestyle='-',
                        marker='x',
                        color='black',
                        label='true spread-skill',
                        markersize=12)
        toggle=False # only do this once

    # plot min/max linear lines
    keys = ['TL', 'TR']
    for i in range(1):
            ax_ss[i].plot(
                [min_val[keys[i]], max_val[keys[i]]],
                [min_val[keys[i]], max_val[keys[i]]],
                linestyle='--',
                color='grey',
                markersize=15,
                alpha=0.5
            )
    
    # save for different rho
    ax_ss[0].set(xlabel='Spread', ylabel='Error/Skill')
    ax_ss[1].set(xlabel=r'Spread (Barycenter $S_{\epsilon}$ cost)', ylabel=r'Error/Skill (Observation $S_{\epsilon}$ Cost)')

from matplotlib.ticker import MaxNLocator
ax_ss[1].xaxis.set_major_locator(MaxNLocator(5))
# log scale
# ax_ss[1].set(yscale='log', xscale='log')

legend_elements = [
            # ---- Divergence marker meaning ----
            Line2D([0], [0],
                marker=markers['kl'], color='black',
                linestyle='none', markersize=12,
                markerfacecolor='black',
                label='KL'
            ),
            Line2D([0], [0],
                marker=markers['tv'], color='black',
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
                color='black', linestyle=line_style[rho],
                label=rho,
            )  for rho in [10.0, 1.0, 0.01, 0.001]
],
        ]

# add legend to fig_decomp
fig_ss.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
    )

fig_ss.savefig(f'spread_curves/grp2/grp2_ss_5_allrho.png', dpi=200, bbox_inches='tight')

plt.close('all')


fig_ss, ax_ss = plt.subplots(1, 2, figsize=(9*2, 7))

toggle=True

for data_set in [5]:
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
    for rho in [10.0, 1.0, 0.01, 0.001]:
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
                    #flip sign assuming small
                    assert np.mean(values[values <= 0]) < 1e-5, np.mean(values[values <= 0]) # small enough for set tolerance

                    values[values<= 0] = np.abs(values[values<= 0])
        
            # plot as simple points per time first
            # top left normal spread skill
            # later
            # top right full se
            # ax_ss[1].plot(se_spread, obs_se,  marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[rho], markersize=12, markerfacecolor='none')
            # bottom left transport decomp
            ax_ss[0].plot(bary_dict['transport_cost'].mean(axis=0), obs_dict['transport_cost'].mean(axis=0), marker=markers[aprox_type], color=aprox_colors[aprox_type],  linestyle=line_style[rho],markersize=12, markerfacecolor='none')
            # bottom right marginal penalty decomp
            ax_ss[1].plot(bary_dict['marginal_penalty'].mean(axis=0), obs_dict['marginal_penalty'].mean(axis=0), marker=markers[aprox_type], color=aprox_colors[aprox_type], linestyle=line_style[rho], markersize=12, markerfacecolor='none')

    # plot min/max linear lines
    # keys = ['TL', 'TR']
    # for i in range(2):
    #         ax_ss[i].plot(
    #             [min_val[keys[i]], max_val[keys[i]]],
    #             [min_val[keys[i]], max_val[keys[i]]],
    #             linestyle='--',
    #             color='grey',
    #             markersize=15,
    #             alpha=0.5
    #         )
    
    # save for different rho
    ax_ss[0].set(xlabel='Spread (Transport)', ylabel='Error/Skill (Transport)')
    ax_ss[1].set(xlabel=r'Spread (Marginal Penalty)', ylabel=r'Error/Skill (Marginal Penalty)')

# log scale
# ax_ss[1].set(yscale='log', xscale='log')

legend_elements = [
            # ---- Divergence marker meaning ----
            Line2D([0], [0],
                marker=markers['kl'], color='black',
                linestyle='none', markersize=12,
                markerfacecolor='black',
                label='KL'
            ),
            Line2D([0], [0],
                marker=markers['tv'], color='black',
                linestyle='none', markersize=12,
                markerfacecolor='none',
                label='TV'
            ),
            *[Line2D([0], [0],
                color='black', linestyle=line_style[rho],
                label=rho,
            )  for rho in [10.0, 1.0, 0.01, 0.001]
],
        ]

# add legend to fig_decomp
fig_ss.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
    )

fig_ss.savefig(f'spread_curves/grp2/grp2_ss_5decomp_allrho.png', dpi=200, bbox_inches='tight')

plt.close('all')