

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"

def plot_paper_tiles(set_to_plot, group):
    cols=4
    rows=1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 9, rows * 8), dpi=200)
    plt.rcParams.update({"font.size": 14})

    for _, (data_set,t) in enumerate(set_to_plot):
        # load data
        data_file = ROOT_FILE+f"ensemble_data/ensemble_dataset_{data_set}.pkl"

        with open(data_file, 'rb') as f:
            dictionary_in_time = pickle.load(f)

        X = dictionary_in_time['grid']
        obs = dictionary_in_time[t]['observation']
        forecasts_mean = np.stack([k[0] for k in dictionary_in_time[t]['forecasts']], axis=-1).mean(-1)
        ax = axes[_]
        ax.set_title(f'Set{data_set}, t = {t}')
        ax.set_facecolor("#f9f6f1")

        im = ax.pcolormesh(
            X[:, :, 0],
            X[:, :, 1],
            np.ma.masked_where(forecasts_mean <= 0, forecasts_mean),
            cmap='copper_r',
            shading='auto'
        )
        cbar = plt.colorbar(im, ax=ax, label='Forecast Mean')

        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))  # switch to sci notation if very small/large
        cbar.update_ticks()

        ax.contour(
            X[:, :, 0],
            X[:, :, 1],
            obs[0],
            levels=[np.unique(obs[0])[1]-1e-9] if len(np.unique(obs[0])) == 2 else [np.unique(obs[0])[1]-1e-9, np.unique(obs[0])[2]-1e-9],
            # cmap='copper',
            colors= mcolors.to_hex(plt.cm.Blues(0.65)), # #b98224
            linestyles='dashdot',
            linewidths=2,
        )

    plt.savefig(ROOT_FILE+"ensemble_data/paper_tiles_grp_{}.png".format(group), bbox_inches='tight')



# ------------------------------------------------------------------
# plot figures for Boundary, rotation and orientation
# [set2 t=0, set2 t=1, set3 t=4, set4 t=4]
# ------------------------------------------------------------------
set_to_plot = [(2,0), (2,1), (3,4), (4,4)]
group = 1

plot_paper_tiles(set_to_plot, group)

# ------------------------------------------------------------------
# plot figures for Boundary, rotation and orientation
# [set6 t=0,1,2,3]
# ------------------------------------------------------------------
set_to_plot = [(6,0), (6,1), (6,2), (6,3)]
group = 3

plot_paper_tiles(set_to_plot, group)

# ------------------------------------------------------------------
# plot figures for Boundary, rotation and orientation
# [set6 t=0,1,2,3]
# ------------------------------------------------------------------
set_to_plot = [(9,4), (12,30), (13,30), (14,30)]
group = '5_7'

plot_paper_tiles(set_to_plot, group)

# ------------------------------------------------------------------
# plot figures for Boundary, rotation and orientation
# [set6 t=0,1,2,3]
# ------------------------------------------------------------------
set_to_plot = [(22, 6), (23,4), (23,8), (23,32)]
group = '10_11'

plot_paper_tiles(set_to_plot, group)