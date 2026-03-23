
import torch
from utils import mmuot_general_costings
import numpy as np

# break down snakemake input into details
muot_cost_file = snakemake.output[0].split('/')[-1].replace('.pkl', '')
# example: observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}
epsilon = float(muot_cost_file.split('_')[3])
rho = float(muot_cost_file.split('_')[5])
aprox_type = muot_cost_file.split('_')[7]

print(f" epsilon={epsilon}, rho={rho}, aprox_type={aprox_type}")
import yaml

with open("global_config.yaml") as f:
    global_config = yaml.safe_load(f)

globals().update(global_config)
tol = float(tol)
mmuot_tol = float(mmuot_tol)

# # pick cuda device
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
# print(f"Using cuda device: {cuda} from {cudas_allowed}")

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

cost_dict_over_time = {}

time_series_debiased_costs = []

for _, t in enumerate(times):
    cost, cost_dict = mmuot_general_costings(
    centre_data=dictionary_in_time[t]['observation'],
    leaf_data=dictionary_in_time[t]['forecasts'],
    epsilon=epsilon, 
    rho=rho, 
    aprox_type=aprox_type, 
    max_iterates=mmuot_max_iterations, 
    tol=mmuot_tol, 
    grid=grid, 
    device=f'cuda:{cuda}',
    weights=weights[_] if weights is not None else None,
    )

    # calculate full cost
    time_series_debiased_costs.append(
        cost # plus one for centre
    )
    cost_dict_over_time[t] = cost_dict

    # save dict - per thing incase thigs crash 
    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(cost_dict_over_time, f)

