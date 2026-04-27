'''
I want this to be a 4 by 4 of spread-skill esque plots. 
top left being the normal spread-skill. top right being full se
the bottom being the deocmposition into transport and marginal penalty for both obs and barycentre.
'''


# load data
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from utils import calculate_true_spread_skill

import sys

# should abstract times and members from global config
debiasing = [True]
epsilons = [0.001] #0.01,
rhos = [1.0, 0.001]
aprox_types = ['kl', 'tv'] # 'balanced',
data_sets = [11] #[12,13,14,21,22] #[k for k in range(1,10)] + [11] + [k for k in range(15, 25+1)]
ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"

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

legend_elements = [
            # ---- Mean curves (large markers) ----
            # Line2D([0], [0],
            #     color=cost_colours['spread'], linestyle='--',
            #     marker='o', markersize=10,
            #     label='Spread (mean)'
            # ),
            # Line2D([0], [0],
            #     color=cost_colours['skill'], linestyle='-.',
            #     marker='o', markersize=10,
            #     label='Error (mean)'
            # ),

            # ---- Individual members (faded small markers) ----
            # Line2D([0], [0],
            #     color=cost_colours['spread'], linestyle='',
            #     marker='o', markersize=6,
            #     alpha=0.4,
            #     label='Spread (per member)'
            # ),
            # Line2D([0], [0],
            #     color=cost_colours['skill'], linestyle='',
            #     marker='o', markersize=6,
            #     alpha=0.4,
            #     label='Error (per member)'
            # ),

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

            # ---- spread-skill diagonal ----
            Line2D([0], [0],
                color='black', linestyle='--',
                label='Ideal spread = error'
            ),
        ]

list_of_files_not_found = []

for data_set in data_sets:
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

    for epsilon in epsilons:
        count = 0

        plt.rcParams.update({"font.size": 14})
        for l, (debiasing, rho) in enumerate([(True, 1.0), (True, 0.01)]):  #(True, 10.), , (True, 0.001)
            # decomposition plot
            fig_decomp, ax_decomp = plt.subplots(2, 2, figsize=(4*3, 5*3))

            max_val = {
                'TL': float('-inf'),
                'TR': float('-inf'),
                'BL': float('-inf'),
                'BR': float('-inf'),
            } 
            min_val = {
                'TL': float('inf'),
                'TR': float('inf'),
                'BL': float('inf'),
                'BR': float('inf'),
            }
            for aprox_type in aprox_types:
                # axes = [ax[0, l], ax[1, l]]

                # actually maybe it's over a few epsilon? Or a few rho? but unified in type for sure.
                # bary_file = ROOT_FILE+f"pkl/{data_set}/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl"
                # obs_file = ROOT_FILE+f"pkl/{data_set}/observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}.pkl"
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

                    # print(obs_se_costs[k][0]['subbreakdown'][(0, 1)].keys())
                    # # dict_keys(['total_cost', 'unbalanced_sinkhorn_terms', 'uot_mu_mu_terms', 'debiasing_term', 'epsilon', 'rho', 'aprox', 'debiasing', 'subbreakdown'])
                    # #dict_keys(['dual_term1', 'dual_term2', 'dual_term3', 'dual_term4', 'primal_c_pi', 'primal_divergence_term', 'primal_entropy', 'uot_mu_mu', 'weight'])
                    # assert 0

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

                # BL: bary_dict['transport_cost'].mean(axis=0), obs_dict['transport_cost'].mean(axis=0)
                max_val['BL'] = max(max_val['BL'], max(obs_dict['transport_cost'].mean(axis=0).max(), bary_dict['transport_cost'].mean(axis=0).max()))
                min_val['BL'] = min(min_val['BL'], min(obs_dict['transport_cost'].mean(axis=0).min(), bary_dict['transport_cost'].mean(axis=0).min()))

                # BR: bary_dict['marginal_penalty'].mean(axis=0), obs_dict['marginal_penalty'].mean(axis=0)
                max_val['BR'] = max(max_val['BR'], max(obs_dict['marginal_penalty'].mean(axis=0).max(), bary_dict['marginal_penalty'].mean(axis=0).max()))
                min_val['BR'] = min(min_val['BR'], min(obs_dict['marginal_penalty'].mean(axis=0).min(), bary_dict['marginal_penalty'].mean(axis=0).min()))

                # zero check
                labels = ['spread', 'error', 'spread decomp', 'error decomp']
                for k, values in enumerate([se_spread, obs_se,  se_spread_decomp, obs_se_decomp]):
                    if np.any(values <= 0):
                        print(f"Warning {data_set}:  -ve in {labels[k]} {aprox_type} at time {t}. average: {np.mean(values[values <= 0])}")

                # plot as simple points per time first
                # top left normal spread skill
                # later
                # top right full se
                ax_decomp[0, 1].plot(se_spread, obs_se,  marker=markers[aprox_type], color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')
                # bottom left transport decomp
                ax_decomp[1, 0].plot(bary_dict['transport_cost'].mean(axis=0), obs_dict['transport_cost'].mean(axis=0), marker=markers[aprox_type], color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')
                # bottom right marginal penalty decomp
                ax_decomp[1, 1].plot(bary_dict['marginal_penalty'].mean(axis=0), obs_dict['marginal_penalty'].mean(axis=0), marker=markers[aprox_type], color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')

            # Now plot independently
            mu_e, sd_e, mu_s, sd_s = calculate_true_spread_skill(
                ROOT_FILE + f"ensemble_data/ensemble_dataset_{data_set}.pkl"
            )
            min_val['TL'] = min(mu_e.min(), mu_s.min())
            max_val['TL'] = max(mu_e.max(), mu_s.max())
            ax_decomp[0, 0].plot(mu_s, mu_e,
                            linestyle=':',
                            marker='x',
                            color='grey',
                            label='true spread-skill')

            # plot min/max linear lines
            keys = ['TL', 'TR', 'BL', 'BR']
            for i in range(2):
                for j in range(2):
                    ax_decomp[i, j].plot(
                        [min_val[keys[i*2 + j]], max_val[keys[i*2 + j]]],
                        [min_val[keys[i*2 + j]], max_val[keys[i*2 + j]]],
                        linestyle='--',
                        color='black',
                        alpha=0.5
                    )
            
            # save for different rho
            for i in range(2):
                for j in range(2):
                    ax_decomp[i, j].set_xlabel('Spread (Barycentre Cost)', fontsize=14)
                    ax_decomp[i, j].set_ylabel('Skill/Error (Observation Cost)')
                    # ax_decomp[i, j].legend()

                    # log scale
                    # ax_decomp[i, j].set_xscale('log')
                    # ax_decomp[i, j].set_yscale('log')
            
            # titles
            ax_decomp[0, 0].set_title(f"True Spread-Skill", fontsize=14)
            ax_decomp[0, 1].set_title(f"SE Spread-Skill", fontsize=14)
            ax_decomp[1, 0].set_title(f"Transport Cost Decomposition", fontsize=14)
            ax_decomp[1, 1].set_title(f"Marginal Penalty Decomposition", fontsize=14)

            # add legend to fig_decomp
            fig_decomp.legend(
                handles=legend_elements,
                loc='lower center',
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.5, 0.00)
                )

            fig_decomp.suptitle(f"Cost Decomposition for dataset {data_set}, epsilon {epsilon}, rho {rho}", fontsize=16, fontweight='bold')
            fig_decomp.tight_layout(rect=[0, 0.03, 1, 0.95])
            # fig_decomp.savefig('testing.png', dpi=200)
            fig_decomp.savefig(f'spread_curves/{data_set}/se_spread_decomp_eps{str(epsilon)[2:]}_rho{rho}.png', dpi=200)

        plt.close('all')

print('Files not found:')
for file in list_of_files_not_found:
    print(f"  {file}")