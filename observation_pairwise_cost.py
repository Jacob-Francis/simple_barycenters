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
mmuot_tol = float(mmuot_tol)
zero_tol = float(zero_tol)

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

bary_list = []
cost_list = []
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
print('NEW CASE')
for _, t in enumerate(times):
    centre_data=dictionary_in_time[t]['observation']
    data_bary_list = dictionary_in_time[t]['forecasts']

    # total mass checking:
    print('TOTAL MASS obs: ', centre_data[0].sum().item())
    for i, d in enumerate(data_bary_list):
        print(f'total mass forecast {i}: ', d[0].sum().item())

    # Generate data holding class
    data_processor = pwb.generate_barycentredataprocessor(
        data_bary_list, 
        barycentre_grid=grid,
        grid=grid, 
        weights=weights[_] if weights is not None else None,
        cuda_device=f'cuda:{cuda}',
        potentials = 'f'
    )

    # checking sums
    # for node in data_processor.graph.nodes:
    #     print(node, data_processor.data_dict[node]['density'].sum().item(), (data_processor.data_dict[node]['density']*data_processor.data_dict[node]['cell_areas']).sum().item())
    # print('obs', centre_data[0].sum().item(), (centre_data[0]*data_processor.data_dict[node]['cell_areas'].cpu().numpy()).sum().item())
    # assert 0
    try:
        data_processor, barycentre, potential_error_list, barycentre_error_list = (
            pwb.asymmetric_sinkhorn_log_algorithm(
                data_processor,
                epsilon=epsilon,
                rho=rho,
                aprox=aprox_type,
                max_iterates=mmuot_max_iterations,
                tol=mmuot_tol,
                epsilon_annealing=False,
                debiasing=debiasing,
                verbose=False,
                measure_constraints=False,
                fixed_barycentre=centre_data[0],
                zero_tol=zero_tol,
            )
        )
    except ValueError:
        data_processor = pwb.generate_barycentredataprocessor(
            data_bary_list, 
            barycentre_grid=grid,
            grid=grid, 
            weights=weights[_] if weights is not None else None,
            cuda_device=f'cuda:{cuda}',
            potentials = 'f'
        )
        print("7.5) Value Error trying annealing and more its ", )
        data_processor, barycentre, potential_error_list, barycentre_error_list = (
            pwb.asymmetric_sinkhorn_log_algorithm(
                data_processor,
                epsilon=epsilon,
                rho=rho,
                aprox=aprox_type,
                max_iterates=mmuot_max_iterations,
                tol=mmuot_tol,
                epsilon_annealing=True,
                debiasing=debiasing,
                verbose=False,
                measure_constraints=False,
                fixed_barycentre=centre_data[0],
                zero_tol=1e-12
            )
        )
    
    # cost calcualtion
    cc, _, se_dict = pwb.asymmetric_cost(
            data_processor,
            epsilon,
            rho,
            aprox=aprox_type,
            debiasing=debiasing,
            verbose=False,
            return_breakdown=True,
            fixed_barycentre=centre_data[0],
            primal_cost=True,
            sym_tol=mmuot_tol,
        )
    
    print('COST:', cc)
    cost_list.append([se_dict])

    print('convergence check: ', potential_error_list[-1], barycentre_error_list[-1])

with open(snakemake.output[0], "wb") as f:
    pickle.dump(cost_list, f)

