# load data
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from utils import calculate_true_spread_skill


import sys

# # Get arguments after script name
# args = sys.argv[1:]

# # Convert to list of ints
# data_sets = list(map(int, args))

# print(data_sets)   # for testing

# should abstract times and members from global config
debiasing = [True]
epsilons = [0.001] #0.01,
rhos = [1.0, 0.001]
aprox_types = ['kl', 'tv'] # 'balanced',
data_sets = [23, 24, 25] # 10 desn't run 7?
group = 11
ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"

# colour blind friendly colors
colours = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
linestyles_dict = dict(zip(data_sets, ['-', '--', '-.', ':']))

#different marker for aprox
markers = dict(kl='o', balanced='s', tv='^')
# i want greys for the aprox types, w
aprox_colors = dict(kl='#377eb8', balanced='#4daf4a', tv='#ff7f00')
cost_colours = dict(
    spread='#377eb8', 
    skill='#ff7f00', 
    se_cost='#4daf4a',
    se_cost_bias='#f781bf',
    mass_ratio='#a65628'
    )

# main spread-skill figure
fig_1, ax_1 = plt.subplots(1, 1, figsize=(8,6))
fig_2, ax_2 = plt.subplots(1, 1, figsize=(8,6))
ax = [ax_1, ax_2]

fig_decomp, ax_decomp = plt.subplots(2, 2, figsize=(8*2, 6*2))


for data_set in data_sets:
    data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{data_set}.pkl"

    with open(data_file, 'rb') as f:
        obvs_dict = pickle.load(f)
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
        for l, (debiasing, rho) in enumerate([(True, 1.0)]):  #, (True, 0.001)
            # decomposition plot

            max_val = float('-inf')
            min_val = float('inf')
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
                    mass_ratio[k] = barycentre.sum().item()/(observation.sum().item()/200**2)
                max_val = max(max_val, max(obs_se.max(), se_spread.max()))
                min_val = min(min_val, min(obs_se.min(), se_spread.min()))

                # zero check
                for values in [se_spread, obs_se,  se_spread_decomp, obs_se_decomp]:
                    if np.any(values <= 0):
                        print(f"Warning: Found zero or negative values in {aprox_type} at time {t}. This may affect log-scale plotting. {np.mean(values[values <= 0])}")

                ax[0].semilogy(times, se_spread, linestyles_dict[data_set], marker=markers[aprox_type], markersize=10, markerfacecolor=None, label='spread', color=cost_colours['spread'])
                ax[0].semilogy(times, obs_se, linestyles_dict[data_set], marker=markers[aprox_type], markersize=10, markerfacecolor=None, label='error', color=cost_colours['skill'])  
                ax[0].semilogy(times, se_spread_decomp.T, marker=markers[aprox_type], linestyle='', markersize=5, markerfacecolor=None, label='se spread per data', color=cost_colours['spread'], alpha=0.5)
                ax[0].semilogy(times, obs_se_decomp.T, marker=markers[aprox_type], linestyle='', markersize=5, markerfacecolor=None, label='se spread per data', color=cost_colours['skill'], alpha=0.5)
                ax[0].set_title(f"Av. mass ratio: {mass_ratio.mean():.4g}", fontsize=16)

                # process zeros
                ax[1].loglog(np.where(se_spread < 0, abs(se_spread), se_spread), np.where(obs_se < 0, abs(obs_se), obs_se), linestyles_dict[data_set], marker=markers[aprox_type], label=f'aprox={aprox_type}', markerfacecolor=None, color=aprox_colors[aprox_type])
                # ax[1].set_title(f"rho: {rho}, debiasing: {debiasing}", fontsize=16)

                # plot as simple points per time first
                ax_decomp[0, 0].plot(times, obs_dict['transport_cost'].mean(axis=0), linestyles_dict[data_set], marker=markers[aprox_type], label=f'{aprox_type} obs transport', color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')
                ax_decomp[0, 1].plot(times, obs_dict['marginal_penalty'].mean(axis=0), linestyles_dict[data_set], marker=markers[aprox_type], label=f'{aprox_type} obs marg', color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')
                ax_decomp[1, 0].plot(times, bary_dict['transport_cost'].mean(axis=0), linestyles_dict[data_set], marker=markers[aprox_type], label=f'{aprox_type} bary transport', color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')
                ax_decomp[1, 1].plot(times, bary_dict['marginal_penalty'].mean(axis=0), linestyles_dict[data_set], marker=markers[aprox_type], label=f'{aprox_type} bary marg', color=aprox_colors[aprox_type], markersize=10, markerfacecolor='none')

                print(f'bary {data_set}', bary_dict['transport_cost'].mean(axis=0))

            # linear fit with max min
            ax[1].loglog([max(min_val,0.0), max_val], [max(min_val,0.0), max_val], 'k--', label='linear trend', alpha=0.5)
            ax[1].set_xlabel('Spread (Barycentre MMUOT Cost)')
            ax[1].set_ylabel('Skill/Error (Observation MMUOT Cost)')
            ax[0].set_xlabel('Case "time"')
            ax[0].set_ylabel('Value')

            pos = ax[1].get_position()

            ax_overlay = fig_2.add_axes(pos, frameon=False)  # new axis in same spot
            ax_overlay.patch.set_alpha(0)                  # transparent background

            # Optional: hide ticks so it doesn’t look messy
            ax_overlay.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)

            # Now plot independently
            mu_e, sd_e, mu_s, sd_s = calculate_true_spread_skill(
                ROOT_FILE + f"ensemble_data/ensemble_dataset_{data_set}.pkl"
            )
            min_val = min(mu_e.min(), mu_s.min())
            max_val = max(mu_e.max(), mu_s.max())
            ax_overlay.plot(mu_s, mu_e,
                            linestyle=linestyles_dict[data_set],
                            marker='x',
                            color='grey',
                            label='true spread-skill')
            
            # save for different rho
            labels = ['Observation Transport', 'Observation Marginal Penalty', 'Barycentre Transport', 'Barycentre Marginal Penalty']
            for i in range(2):
                for j in range(2):
                    ax_decomp[i, j].set_xlabel('Time', fontsize=14)
                    ax_decomp[i, j].set_ylabel(labels[i*2 + j], fontsize=14)
                    # ax_decomp[i, j].legend()

            # fig_decomp.suptitle(f"Cost Decomposition for dataset {data_set}, epsilon {epsilon}, rho {rho}", fontsize=16, fontweight='bold')
            # fig_decomp.tight_layout(rect=[0, 0.03, 1, 0.95])

    
        ax[0].set_ylabel("Scores", rotation=90, fontsize=20, fontweight='bold')
        ax[1].set_ylabel("Spread-Skill", rotation=90, fontsize=20, fontweight='bold')

        # column_headers = ['Debiased, rho: 1.0']

        # for col in range(1):
        #     # Get the position of the top subplot in that column
        #     pos = ax[0].get_position()
            
        #     # Place text slightly above it
        #     fig.text(
        #         pos.x0 + pos.width / 2,
        #         pos.y1 + 0.02,
        #         column_headers[col],
        #         ha='center',
        #         va='bottom',
        #         fontsize=20,
        #         fontweight='bold'
        #     )
        
        legend_elements = [
            # ---- Mean curves (large markers) ----
            # Line2D([0], [0],
            #     color=cost_colours['spread'], 
            #     marker='o', markersize=10,
            #     label='Spread (mean)'
            # ),
            # Line2D([0], [0],
            #     color=cost_colours['skill'],
            #     marker='o', markersize=10,
            #     label='Error (mean)'
            # ),

            # # ---- Individual members (faded small markers) ----
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

            # ---- spread-skill diagonal ----
            Line2D([0], [0],
                color='black', marker='x',
                linestyle='none', markersize=12,
                label='True Spread-Skill'
            ),
            *[ 
                Line2D([0], [0],
                    color='black', linestyle=linestyles_dict[data_set],
                    label=f'Dataset {data_set}'
                ) for data_set in data_sets
            ]                
        ]

        fig_1.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.1)
        )
        fig_2.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.1)
        )
        fig_decomp.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=6,
            frameon=False,
            bbox_to_anchor=(0.5, 0.00)
        )

try:
    fig_1.savefig(f'spread_curves/grp{group}/grp{group}_se_vary_eps{str(epsilon)[2:]}.png', dpi=200, bbox_inches="tight")
    
# if not folder dataset exists, create it and save
except FileNotFoundError:
    import os
    os.makedirs(f'spread_curves/grp{group}', exist_ok=True)
    fig_1.savefig(f'spread_curves/grp{group}/grp{group}_se_vary_eps{str(epsilon)[2:]}_rho{rho}.png', dpi=200, bbox_inches="tight")

fig_2.savefig(f'spread_curves/grp{group}/grp{group}_se_sprea_skill_eps{str(epsilon)[2:]}_rho{rho}.png', dpi=200, bbox_inches="tight")

fig_decomp.savefig(f'spread_curves/grp{group}/grp{group}_se_spread_decomp_eps{str(epsilon)[2:]}_rho{rho}.png', dpi=200, bbox_inches="tight")

plt.close('all')