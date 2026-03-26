import torch
import pwbarycentres as pwb
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ROOT = '/home/jjf817/PhD_jobs/simple_barycentres/test/'
# # dummy class to hold run snakmeke input and output
# class dummy_snake:
#     input = ["/home/jjf817/PhD_jobs/simple_barycentres/ensemble_data/ensemble_dataset_1.pkl"]
#     output = [
#         ROOT + "1/atest_out_eps_0.005_rho_0.001_aprox_kl_debiasing_False.pkl",
#         ROOT + "1/atest_conv_eps_0.005_rho_0.001_aprox_kl_debiasing_False.png",
#         ROOT + "1/atest_plot_eps_0.005_rho_0.001_aprox_kl_debiasing_False.png",
#         ROOT + "1/atest_cost_eps_0.005_rho_0.001_aprox_kl_debiasing_False.pkl",
#     ]

# snakemake = dummy_snake()

# break down snakemake input into details
muot_cost_file = snakemake.output[0].split('/')[-1].replace('.pkl', '')
# example: observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}
epsilon = float(muot_cost_file.split('_')[3])
rho = float(muot_cost_file.split('_')[5])
aprox_type = muot_cost_file.split('_')[7]
debiasing = muot_cost_file.split('_')[9] == 'True' # its boolean

print(f" epsilon={epsilon}, rho={rho}, aprox_type={aprox_type}, debiasing={debiasing}, ")
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
# cuda = np.random.choice(cudas_allowed)
print(f"Using cuda device: {cuda} from {cudas_allowed}")

# load the data
with open(snakemake.input[0], 'rb') as f:
    dictionary_in_time = pickle.load(f)

times = dictionary_in_time['times']
members = dictionary_in_time['members']
grid = dictionary_in_time['grid']

if 'weights' in dictionary_in_time:
    weights = dictionary_in_time['weights']
else:
    weights = None

bary_list = []
cost_list = []
fig, ax = plt.subplots(1, 3, figsize=(4*3, 4))

for _, t in enumerate(times):
    data_bary_list = dictionary_in_time[t]['forecasts']
    # Generate data holding class
    data_processor = pwb.generate_barycentredataprocessor(
        data_bary_list, 
        barycentre_grid=grid,
        grid=grid, 
        weights=weights[_] if weights is not None else None,
        cuda_device=f'cuda:{cuda}',
        potentials = 'f'
    )

    try:
        data_processor, barycentre, potential_error_list, barycentre_error_list, energy_list = (
            pwb.asymmetric_sinkhorn_log_algorithm(
                data_processor,
                epsilon=epsilon,
                rho=rho,
                aprox=aprox_type,
                max_iterates=max_iterations,
                tol=tol,
                epsilon_annealing=False,
                debiasing=debiasing,
                verbose=False,
                measure_constraints=False,
                lags={
                    'barycentre': 1,
                    'debiasing': 1,
                },
                energy_tracking=True
            )
        )
    except ValueError:
        print("7.5) Value Error trying annealing and more its ", )
        data_processor, barycentre, potential_error_list, barycentre_error_list, energy_list = (
            pwb.asymmetric_sinkhorn_log_algorithm(
                data_processor,
                epsilon=epsilon,
                rho=rho,
                aprox=aprox_type,
                max_iterates=max_iterations*5,
                tol=tol,
                epsilon_annealing=True,
                debiasing=debiasing,
                verbose=False,
                measure_constraints=False,
                lags={
                    'barycentre': 1,
                    'debiasing': 1,
                },
                energy_tracking=True
            )
        )

    # rescale 
    barycentre = barycentre / np.prod(barycentre.shape)
    bary_list.append(barycentre)

    # cost calcualtion
    cc, _, se_dict = pwb.asymmetric_cost(
            data_processor,
            epsilon,
            rho,
            aprox=aprox_type,
            debiasing=debiasing,
            verbose=False,
            return_breakdown=True,
            primal_costs=True,
        )
    
    print('COST:', cc)
    cost_list.append([se_dict])

    # convergence checks

    ax[0].semilogy(barycentre_error_list, label=f"{t}")
    ax[0].set_xlabel("Outer Iteration")
    ax[0].set_ylabel("Bary update change")

    ax[1].semilogy(potential_error_list, label=f"{t}")
    ax[1].set_xlabel("Outer Iteration")
    ax[1].set_ylabel("potential update change")

    ax[2].semilogy(energy_list['total_cost'], label=f"{t}")
    ax[2].set_xlabel("Outer Iteration")
    ax[2].set_ylabel("Energy")

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
    axes[k, 0].set_title(f"Ensemble mean, sum={ensemble_mean.sum():.2f}")

    img = axes[k, 1].imshow(dictionary_in_time[t]['observation'][0].reshape(200,  200), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
    fig.colorbar(img, ax=axes[k, 1])
    axes[k, 1].set_title(f"Observation, sum={dictionary_in_time[t]['observation'][0].sum()/200**2:.2f}")


    img = axes[k, 2].imshow(bary_list[k].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
    fig.colorbar(img, ax=axes[k, 2])
    axes[k, 2].set_title(f"Barycenter, sum={bary_list[k].sum():.2f}")

plt.savefig(snakemake.output[2])

