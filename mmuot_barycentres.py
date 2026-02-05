import torch
from utils import mmuot_general_costings
import pwbarycentres as pwb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import mmuot

# break down snakemake input into details
muot_cost_file = snakemake.output[0].split('/')[-1].replace('.pkl', '')
# example: observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}
epsilon = float(muot_cost_file.split('_')[3])
rho = float(muot_cost_file.split('_')[5])
aprox_type = muot_cost_file.split('_')[7]
debiasing = muot_cost_file.split('_')[9]

print(f" epsilon={epsilon}, rho={rho}, aprox_type={aprox_type}")
import yaml

with open("global_config.yaml") as f:
    global_config = yaml.safe_load(f)

globals().update(global_config)
tol = float(tol)

# pick cuda device
best_gpu = None
max_free_mem = 0

for i in cudas_allowed:
    torch.cuda.set_device(i)
    free_mem, total_mem = torch.cuda.mem_get_info(i)  # bytes

    if free_mem > max_free_mem:
        max_free_mem = free_mem
        best_gpu = i

cuda = best_gpu

# load the data
with open(snakemake.input[0], 'rb') as f:
    dictionary_in_time = pickle.load(f)

times = dictionary_in_time['times']
members = dictionary_in_time['members']
grid = dictionary_in_time['grid']

bary_list = []
cost_list = []
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

for t in times:
    data_bary_list = dictionary_in_time[t]['forecasts']


    data_processor = mmuot.generate_mmuotdataprocessor_star_graph(
        data_bary_list, 
        grid=grid,
        cuda_device=f'cuda:{cuda}',)
    
    data_processor, conlist = mmuot.mmuot_sinkhorn_loop(
        data_processor,
        epsilon,
        rho,
        max_iterations=max_iterations,
        tol=tol,
        aprox=aprox_type,
        prod=False,
        convergence_tracking=True,
        verbose=False,
        barycentre=True
    )
    se, cost_dict = mmuot.mmuot_dual_cost(
        data_processor, epsilon, rho, aprox_type, prod=False, no_kernal_term=False
    )
    barycentre = mmuot.mmuot_marginal_j(data_processor, 0, epsilon, prod=False, update_alpha=False)[0]

    bary_list.append(barycentre)
    cost_list.append(cost_dict)
    # convergence checks

    ax.semilogy(conlist, label=f"{t}")
    ax.set_ylabel("potential update change")
    
plt.savefig(snakemake.output[1])
plt.clf()

# save the data
with open(snakemake.output[0], "wb") as f:
    pickle.dump(bary_list, f)

with open(snakemake.output[3], "wb") as f:
    pickle.dump(cost_list, f)

# save the barycentres
# -------------------------------------------
# PLOTTING:
# -------------------------------------------

# Plot the four density fields
fig, axes = plt.subplots(len(times), 3, figsize=(13, 5*len(times)))

for k, t in enumerate(times):
    ensemble_mean = np.mean(np.stack([f for f,k in dictionary_in_time[t]['forecasts']]), axis=0)

    img = axes[k, 0].imshow(ensemble_mean.reshape(200,  200), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
    fig.colorbar(img, ax=axes[k, 0])

    img = axes[k, 1].imshow(dictionary_in_time[t]['observation'][0].reshape(200,  200), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
    fig.colorbar(img, ax=axes[k, 1])

    img = axes[k, 2].imshow(bary_list[k].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
    fig.colorbar(img, ax=axes[k, 2])

plt.savefig(snakemake.output[2])

