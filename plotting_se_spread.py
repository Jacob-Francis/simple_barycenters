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
data_sets = [k for k in range(1, 10)] + [k for k in range(11,25+1)] # 10 desn't run
# data_sets = [21] # [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 18, 19, 20, 21]  # [ 2, 5, 7, 8,9,  17, 11]
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

        fig, ax = plt.subplots(2, 2, figsize=(4*3, 5*3))

        plt.rcParams.update({"font.size": 14})
        for l, (debiasing, rho) in enumerate([(True, 1.0), (True, 0.001)]):
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

                    print(obs_se_costs[k][0].keys())

                    assert 0

                    obs_se[k] = obs_se_costs[k][0]['total_cost']
                    obs_se_decomp[:, k] = se_per_data

                    # mass ratio
                    barycentre = bary_output[k]
                    observation = obvs_dict[t]['observation'][0]
                    mass_ratio[k] = barycentre.sum().item()/observation.sum().item()
                
                max_val = max(max_val, max(obs_se.max(), se_spread.max()))
                min_val = min(min_val, min(obs_se.min(), se_spread.min()))

                ax[0, l].plot(times, se_spread, '--', marker=markers[aprox_type], markersize=10, label='spread', color=cost_colours['spread'])
                ax[0, l].plot(times, obs_se, '-.', marker=markers[aprox_type], markersize=10, label='error', color=cost_colours['skill'])  
                ax[0, l].plot(times, se_spread_decomp.T, marker=markers[aprox_type], linestyle='', markersize=5, markerfacecolor=None, label='se spread per data', color=cost_colours['spread'], alpha=0.5)
                ax[0, l].plot(times, obs_se_decomp.T, marker=markers[aprox_type], linestyle='', markersize=5, markerfacecolor=None, label='se spread per data', color=cost_colours['skill'], alpha=0.5)
                ax[0,l].set_title(f"Av. mass ratio: {mass_ratio.mean():.4g}", fontsize=16)

                ax[1, l].plot(se_spread, obs_se, 'k--', marker=markers[aprox_type], label=f'aprox={aprox_type}')
                ax[1,l].set_title(f"rho: {rho}, debiasing: {debiasing}", fontsize=16)

            # linear fit with max min
            ax[1, l].plot([min_val, max_val], [min_val, max_val], 'k--', label='linear trend', alpha=0.5)
            ax[1, l].set_xlabel('Spread (Barycentre MMUOT Cost)')
            ax[1, l].set_ylabel('Skill/Error (Observation MMUOT Cost)')
            ax[0, l].set_xlabel('Case "time"')
            ax[0, l].set_ylabel('Value')

            pos = ax[1, l].get_position()

            ax_overlay = fig.add_axes(pos, frameon=False)  # new axis in same spot
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
                            linestyle=':',
                            marker='x',
                            color='grey',
                            label='true spread-skill')
            # ax_overlay.plot([min_val, max_val], [min_val, max_val], '--',
            #                 color='grey',
            #                 )
    
        ax[0, 0].set_ylabel("Scores", rotation=90, fontsize=20, fontweight='bold')
        ax[1, 0].set_ylabel("Spread-Skill", rotation=90, fontsize=20, fontweight='bold')

        column_headers = ['Debiased, rho: 1.0', 'Debiased, rho: 0.001']

        for col in range(2):
            # Get the position of the top subplot in that column
            pos = ax[0, col].get_position()
            
            # Place text slightly above it
            fig.text(
                pos.x0 + pos.width / 2,
                pos.y1 + 0.02,
                column_headers[col],
                ha='center',
                va='bottom',
                fontsize=20,
                fontweight='bold'
            )
        
        legend_elements = [
            # ---- Mean curves (large markers) ----
            Line2D([0], [0],
                color=cost_colours['spread'], linestyle='--',
                marker='o', markersize=10,
                label='Spread (mean)'
            ),
            Line2D([0], [0],
                color=cost_colours['skill'], linestyle='-.',
                marker='o', markersize=10,
                label='Error (mean)'
            ),

            # ---- Individual members (faded small markers) ----
            Line2D([0], [0],
                color=cost_colours['spread'], linestyle='',
                marker='o', markersize=6,
                alpha=0.4,
                label='Spread (per member)'
            ),
            Line2D([0], [0],
                color=cost_colours['skill'], linestyle='',
                marker='o', markersize=6,
                alpha=0.4,
                label='Error (per member)'
            ),

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

        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.00)
        )


        try:
            plt.savefig(f'spread_curves/{data_set}/{data_set}_se_spread_eps{str(epsilon)[2:]}.png', dpi=200)
        # if not folder dataset exists, create it and save
        except FileNotFoundError:
            import os
            os.makedirs(f'spread_curves/{data_set}', exist_ok=True)
            plt.savefig(f'spread_curves/{data_set}/{data_set}_se_spread_eps{str(epsilon)[2:]}.png', dpi=200)
        plt.close('all')
