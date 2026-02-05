import pwbarycentres as pwb
from graph_dp import SinkhornDataProcessor
from utils import *

import pickle
import torch

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
cuda = 2


for debiasing in [True, False]:
    for epsilon in [0.05, 0.005, 0.001]:
        for rho in [1.0, 0.1, 0.01]:
            for aprox_type in ['kl', 'balanced', 'tv']:
                file_title = f'testing_rho{rho:.3g}_{aprox_type}_debiasing_{debiasing}_eps{epsilon:.3g}'
                times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
                members = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                # Set the path to the data
                path_to_data = '/home/jjf817/macomp001/jjf817/PhD_jobs/ICP_Cases/MescoVict_cases/'

                # Gather time series data
                data_list = []
                dictionary_in_time = {}
                for t in times:
                    data_list = []
                    for mem in members:
                        O, F, X = generate_circle_ensemble(mem, t, path_to_data)

                        data_list.append([ process_array(F), process_array(X)])

                    data_list = [[ process_array(O), process_array(X)]] + data_list
                    len(data_list)
                    dictionary_in_time[t] = data_list

                grid = process_array(X, device=f'cuda:{cuda}')
                print("Data loaded")

                # -------------------------------------------
                # Compute the skill scores:
                # -------------------------------------------

                def ensemble_mean_error(mean, observation):
                    M = np.prod(observation.shape)
                    return torch.sqrt(((mean - observation)**2).sum() / M)

                def summary_ensemble_mean_error(mean, observation):
                    temp = ensemble_mean_error(mean, observation)
                    return torch.mean(temp), torch.std(temp)

                def spread(densities, mean):
                    batch_densities = torch.stack(densities, dim=-1)
                    M = np.prod(mean.shape)
                    N = len(densities)
                    return torch.sqrt(((batch_densities.view(-1,N) - mean.view(-1, 1))**2).sum(axis=-1)/(N-1))

                def summary_spread(densities, mean):
                    temp = spread(densities, mean)
                    return torch.mean(temp), torch.std(temp)

                mu_e_list = []
                sd_e_list = []
                mu_s_list = []
                st_s_list = []

                ensemble_mean_list = []
                for k,t in enumerate(times):
                    size = len(members)
                    ensemble_mean_l2 =  torch.zeros_like(dictionary_in_time[t][0][0])
                    for mem in members:
                        ensemble_mean_l2 += dictionary_in_time[t][mem][0]
                    ensemble_mean_l2 /= size
                    ensemble_mean_list.append(ensemble_mean_l2)

                for k, t in enumerate(times):
                    ensemble_mean_l2 = ensemble_mean_list[k]
                    mu_e, sd_e = summary_ensemble_mean_error(ensemble_mean_l2, dictionary_in_time[t][0][0])
                    ensemble_memebers = [dictionary_in_time[t][mem][0] for mem in members]
                    mu_s, st_s = summary_spread(ensemble_memebers, ensemble_mean_l2)
                    mu_e_list.append(mu_e.item())
                    sd_e_list.append(sd_e.item())
                    mu_s_list.append(mu_s.item())
                    st_s_list.append(st_s.item())    


                # Plot the four density fields
                fig, axes = plt.subplots(len(times), 2, figsize=(7, 5*len(times)))

                for k, t in enumerate(times):

                    img = axes[k, 0].imshow(ensemble_mean_list[k].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
                    fig.colorbar(img, ax=axes[k, 0])

                    img = axes[k, 1].imshow(dictionary_in_time[t][0][0].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
                    fig.colorbar(img, ax=axes[k, 1])


                plt.savefig(f'figs/l2_plot_{file_title}.png')
                # ----------------------------------------------------
                # plot
                # ----------------------------------------------------

                time_steps = np.arange(len(mu_e_list)) 
                mu_e = np.array(mu_e_list)
                sd_e = np.array(sd_e_list)

                mu_s = np.array(mu_s_list)
                sd_s = np.array(st_s_list)

                # Create the figure
                fig, axes = plt.subplots(1, 2, figsize=(8,5))

                # Plot with error bars
                axes[0].errorbar(time_steps, mu_e, yerr=sd_e, fmt='o-', label="Error", capsize=5)
                axes[0].errorbar(time_steps, mu_s, yerr=sd_s, fmt='s-', label="Spread", capsize=5)
                axes[0].axhline(0)

                # Labels and legend
                axes[0].set_xlabel("Time Steps")
                axes[0].set_ylabel("Mean Value")
                axes[0].set_title("Mean with Standard Deviation (Error Bars)")
                axes[0].legend()
                axes[0].grid(True)

                # Plot with error bars
                axes[1].plot(mu_e, mu_s, '.-')
                # linear line
                print('Shapes max, ', mu_e, mu_s)
                max_val = max(mu_e.max(), mu_s.max())   
                axes[1].plot([0, max_val], [0, max_val], 'k--')
                axes[1].set_xlabel("Error")
                axes[1].set_ylabel("Spread")
                axes[1].grid(True)
                    
                plt.tight_layout()
                plt.savefig(f'figs/true_skill_scores_{file_title}.png')
                # assert False

                # -------------------------------------------
                # Calculate Barycentre!
                # -------------------------------------------
                bary_list = []
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))

                for k, t in enumerate(times):
                    data = dictionary_in_time[t]

                    # Generate data holding class
                    data_processor = pwb.generate_barycentredataprocessor(
                        data,
                        barycentre_grid=process_array(X, device=f'cuda:{cuda}'),
                        grid=process_array(X, device=f'cuda:{cuda}'), 
                        weights=None, 
                        cuda_device=f'cuda:{cuda}',
                        potentials = 'a'
                    )

                    data_processor, barycentre, potential_error_list, barycentre_error_list = (
                        pwb.asymmetric_sinkhorn_algorithm(
                            data_processor,
                            epsilon=process_array(epsilon, device=f'cuda:{cuda}'),
                            rho=rho,
                            aprox=aprox_type,
                            max_iterates=10000,
                            tol=1e-9,
                            epsilon_annealing=False,
                            debiasing=debiasing,
                            debiasing_update_freq=3,
                            mass_scaling=False,
                        )
                    )

                    bary_list.append(barycentre)


                    ax[0].semilogy(barycentre_error_list, label=f"{k}")
                    ax[0].set_xlabel("Outer Iteration")
                    ax[0].set_ylabel("Bary update change")
    
                    ax[1].semilogy(potential_error_list, label=f"{k}")
                    ax[1].set_xlabel("Outer Iteration")
                    ax[1].set_ylabel("potential update change")
                
                plt.savefig(f"figs/{file_title}_error_convergence.pdf")
                plt.clf()

                # -------------------------------------------
                # PLOTTING:
                # -------------------------------------------

                # Plot the four density fields
                fig, axes = plt.subplots(len(times), 3, figsize=(13, 5*len(times)))

                for k, t in enumerate(times):

                    img = axes[k, 0].imshow(ensemble_mean_list[k].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
                    fig.colorbar(img, ax=axes[k, 0])

                    img = axes[k, 1].imshow(dictionary_in_time[t][0][0].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
                    fig.colorbar(img, ax=axes[k, 1])

                    img = axes[k, 2].imshow(bary_list[k].reshape(200,  200).detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="Greys", alpha=0.8)#, norm=LogNorm())
                    fig.colorbar(img, ax=axes[k, 2])

                plt.savefig(f'figs/all_plots_{file_title}.png')

                import pickle

                with open(f"pkl/bary_{file_title}.pkl", "wb") as f:
                    pickle.dump(bary_list, f)

                with open(f"pkl/dictionary_in_time_{file_title}.pkl", "wb") as f:
                    pickle.dump(dictionary_in_time, f)

                # with open("data.pkl", "rb") as f:
                #     loaded_data = pickle.load(f)