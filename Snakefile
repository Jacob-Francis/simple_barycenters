debiasing = [True] # False
epsilon = [0.005, 0.001] # 0.005, 
rho = [1.0, 0.001] #[10.0, 1.0, 0.01, 0.001]
aprox_type = ['kl', 'tv'] 
data_sets = [12,13,14,21,22] # [k for k in range(1,10)] + [k for k in range(11,25+1)] # 10 desn't run
ROOT_FILE = "/home/jjf817/PhD_jobs/simple_barycentres/"

rule all:
    input:
        expand(ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl", data_set=data_sets),
        expand(ROOT_FILE+"ensemble_data/true_skill_scores_{data_set}.png", data_set=data_sets),
        # expand(ROOT_FILE+"pkl/{data_set}/observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}.pkl",
        #        epsilon=epsilon,
        #        rho=rho,
        #        aprox_type=aprox_type,
        #        data_set=data_sets
        #        ),
        expand(ROOT_FILE+"pkl/{data_set}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
               debiasing=debiasing,
               epsilon=epsilon,
               rho=rho,
               aprox_type=aprox_type,
               data_set=data_sets),
        expand(ROOT_FILE+"figs/{data_set}/barycentres_plot_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.png",    
               debiasing=debiasing,
               epsilon=epsilon,
               rho=rho,
               aprox_type=aprox_type,
               data_set=data_sets),
        # expand(ROOT_FILE+"pkl/{data_set}/mmuotbary_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
        #         debiasing=debiasing,
        #         epsilon=epsilon,
        #         rho=rho,
        #         aprox_type=aprox_type,
        #         data_set=data_sets),
        # expand(ROOT_FILE+"pkl/{data_set}/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
        #        debiasing=debiasing,
        #        epsilon=epsilon,
        #        rho=rho,
        #        aprox_type=aprox_type,
        #        data_set=data_sets),
        expand(ROOT_FILE+"pkl/{data_set}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
               debiasing=debiasing,
               epsilon=epsilon,
               rho=rho,
               aprox_type=aprox_type,
               data_set=data_sets)

rule generate_datasets:
    output:
        pkl=ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl",
        l2_plot=ROOT_FILE+"ensemble_data/plot_set_{data_set}.png",
    script:
        ROOT_FILE+"generate_set.py"

rule true_spread_skill:
    input:
        pkl=ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl"
    output:
        skill_scores=ROOT_FILE+"ensemble_data/true_skill_scores_{data_set}.png"
    script:
        ROOT_FILE+"point_wise_spread_skill.py"

rule observation_muot_cost:
    input:
        pkl=ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl"
    output:
        ROOT_FILE+"pkl/{data_set}/observation_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}.pkl"
    script:
        ROOT_FILE+"observation_mmuot_cost.py"

rule barycentre_calculation:
    input:
        pkl=ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl",
    output:
        ROOT_FILE+"pkl/{data_set}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
        ROOT_FILE+"figs/{data_set}/barycentres_convergence_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.png",   
        ROOT_FILE+"figs/{data_set}/barycentres_plot_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.png",    
        ROOT_FILE+"pkl/{data_set}/barycentres_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
    script:
        ROOT_FILE+"barycentre_calculation.py"
    
rule mmuot_barycentre_calculation:
    input:
        pkl=ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl",
    output:
        ROOT_FILE+"pkl/{data_set}/mmuotbary_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
        ROOT_FILE+"figs/{data_set}/mmuotbary_convergence_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.png",   
        ROOT_FILE+"figs/{data_set}/mmuotbary_plot_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.png",    
        ROOT_FILE+"pkl/{data_set}/mmuotbary_costs_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
    script:
        ROOT_FILE+"mmuot_barycentres.py"

rule barycentre_mmuot:
    input:
        ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl",
        ROOT_FILE+"pkl/{data_set}/barycentres_output_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
    output:
        ROOT_FILE+"pkl/{data_set}/barycentre_mmuot_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
    script:
        ROOT_FILE+"barycentre_mmuot_cost.py"

rule observation_pairwise_cost:
    input:
        pkl=ROOT_FILE+"ensemble_data/ensemble_dataset_{data_set}.pkl",
    output:
        ROOT_FILE+"pkl/{data_set}/observation_cost_eps_{epsilon}_rho_{rho}_aprox_{aprox_type}_debiasing_{debiasing}.pkl",
    script:
        ROOT_FILE+"observation_pairwise_cost.py"
