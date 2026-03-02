import pwbarycentres as pwb
from graph_dp import SinkhornDataProcessor
from utils import *
import numpy as np
import pickle
import torch

import os 

members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
path_to_data = '/home/jjf817/macomp001/jjf817/PhD_jobs/ICP_Cases/MescoVict_cases/'

set_number = int(snakemake.wildcards['data_set'])

data_list = []
dictionary_in_time = {}

rng = np.random.default_rng(seed=12345)


if set_number == 1:
    # ---------------------------------------
    # Set 1: Orientation and boundary
    # ---------------------------------------
    count = 0
    # 1) generate circle at centre
    data_list = []
    for mem in members:
        F, X = generate_circle_ensemble(mem, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)))

        data_list.append([F, None])

    # generate the observation too as a random perturbation
    O, X = generate_circle_ensemble(0, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)))

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )

    count += 1

    # 2) generate circle off centre
    data_list = []
    for mem in members:
        F, X = generate_circle_ensemble(mem, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        F = np.roll(F, 40, axis=1)
        F = np.roll(F, 40, axis=0)
        data_list.append([F, None])

    # generate the observation too as a random perturbation
    O, X = generate_circle_ensemble(0, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)))
    O = np.roll(O, 40, axis=1)
    O = np.roll(O, 40, axis=0)

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )

    count += 1

    #3) generatre ellipse at centre
    data_list = []
    for mem in members:
        F, X = generate_ellipse_ensemble(mem, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)))

        data_list.append([F, None])

    # generate the observation too as a random perturbation
    O, X = generate_ellipse_ensemble(0, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)))

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )

    count += 1

    # 4) generatre ellipse at angle
    data_list = []
    for mem in members:
        F, X = generate_ellipse_ensemble(mem, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)), case='E2')

        data_list.append([F, None])

    # generate the observation too as a random perturbation
    O, X = generate_ellipse_ensemble(0, 30, path_to_data, seed_pert=rng.integers(0,int(1e7)), case='E2')

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )

    count += 1

    dictionary_in_time['times'] = [k for k in range(count)]

elif set_number == 2:
    # ---------------------------------------
    # Set 2: Rotated ellipses
    # ---------------------------------------
    times = [0, 1, 2, 3, 4]
   
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_rotated_ellipse_ensemble(mem, np.pi*t/16, path_to_data, seed_pert=rng.integers(0,int(1e7)))

            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_rotated_ellipse_ensemble(0, np.pi*t/16, path_to_data, seed_pert=rng.integers(0,int(1e7)))

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 3:
    # ---------------------------------------
    # Set 3: double penalty smaller envelope for generating random circle centres.
    # ---------------------------------------
    times = [0, 20, 40, 60, 80, 100]

    for t in times:
        data_list = []
        for mem in members:
            O, F, X = generate_double_penalty_circle_ensemble(mem, t, 20, path_to_data, seed_pert=rng.integers(0,int(1e7)))

            data_list.append([F, None])

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 4:
    # ---------------------------------------
    # Set 4: double penalty Larger envelope for generating random circle centres.
    # ---------------------------------------
    times = [0, 20, 40, 60, 80, 100]
   
    for t in times:
        data_list = []
        for mem in members:
            O, F, X = generate_double_penalty_circle_ensemble(mem, t, 40, path_to_data, seed_pert=rng.integers(0,int(1e7)))

            data_list.append([F, None])

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 5:
    # ---------------------------------------
    # Set 5: Cone section moving outwards 45 degree
    # ---------------------------------------
    times = [0, 20, 40, 60, 80, 100]
    
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_cone_section_ensemble(mem, r1=t+10, r0=t, angle=np.deg2rad(45), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))

            data_list.append([F, None])
        
        O, X = generate_cone_section_ensemble(0, r1=t+20, r0=t, angle=np.deg2rad(45), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 6:
    # ---------------------------------------
    # Set 6: Cone section moving outwards 22.5 degree
    # ---------------------------------------
    times = [0, 20, 40, 60, 80, 100]
    
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_cone_section_ensemble(mem, r1=t+10, r0=t, angle=np.deg2rad(22.5), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))

            data_list.append([F, None])
        
        O, X = generate_cone_section_ensemble(0, r1=t+20, r0=t, angle=np.deg2rad(22.5), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 7:
    # ---------------------------------------
    # Set 7: Cone section moving outwards 90 degree
    # ---------------------------------------

    times = [0, 20, 40, 60, 80, 100]

    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_cone_section_ensemble(mem, r1=t+10, r0=t, angle=np.deg2rad(90), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))

            data_list.append([F, None])
        
        O, X = generate_cone_section_ensemble(0, r1=t+20, r0=t, angle=np.deg2rad(90), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )

    dictionary_in_time['times'] = times

elif set_number == 8:
    # ---------------------------------------
    # Set 8: Barycentre off the support 
    # ---------------------------------------
    count = 0

    # 1) Anulus
    data_list = []
    for mem in members:
        F, X = generate_cone_section_ensemble(mem, r1=60, r0=40, angle=0, path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))
        F = np.roll(F, 50, axis=1) # roll back to centre
        data_list.append([F, None])

    O, X = generate_cone_section_ensemble(0, r1=60, r0=40, angle=0, path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))
    O = np.roll(O, 50, axis=1) # roll back to centre

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )
    count += 1 

    # 2) Half cone section at 90 degrees
    data_list = []
    for mem in members:
        F, X = generate_cone_section_ensemble(mem, r1=60, r0=40, angle=np.deg2rad(90), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))

        data_list.append([F, None])

    O, X = generate_cone_section_ensemble(0, r1=60, r0=40, angle=np.deg2rad(90), path_to_data=path_to_data, seed_pert=rng.integers(0,int(1e7)))
    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )
    count += 1 

    # 3) two clusters - 1 event per

    data_list = []
    for mem in members:

        cluster = rng.choice([-50, 50], size=1)
        F, X = generate_circle_ensemble(mem, 20, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        F = np.roll(F, cluster, axis=1)

        data_list.append([F, None])

    cluster = rng.choice([-50, 50], size=1)
    O, X = generate_circle_ensemble(0, 20, path_to_data, seed_pert=rng.integers(0,int(1e7)))
    O = np.roll(O, cluster, axis=1)

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )

    count += 1 

    # 4) two cluseters - 2 events per
    data_list = []
    for mem in members:

        F, X = generate_two_circle_event_clusters(mem, 20, [-50, 0], [50, 0], path_to_data, seed_pert=rng.integers(0,int(1e7)))
        data_list.append([F, None])

    O, X = generate_two_circle_event_clusters(0, 20, [-50, 0], [50, 0], path_to_data, seed_pert=rng.integers(0,int(1e7)))

    dictionary_in_time[count] = dict(
        observation = [O, None],
        forecasts = data_list
        )

    count += 1 

    dictionary_in_time['times'] = [k for k in range(count)]

elif set_number == 9:
    # ---------------------------------------
    # Set 9: Clustering/bifurcation test
    # ---------------------------------------

    # I should abstract this...
    times = [0, 1, 2, 3, 4]
    # whether to roll forward to backwards 50 
    clusters = [50, -50, -50, -50, -50, -50, -50, -50, -50, -50]

    # Gather time series data
    data_list = []
    dictionary_in_time = {}
    for t in times:
        data_list = []
        for k, mem in enumerate(members):
            cluster = clusters[k]
            F, X = generate_circle_ensemble(mem, 20, path_to_data, seed_pert=rng.integers(0,int(1e7)))
            F = np.roll(F, cluster, axis=1)

            data_list.append([F, None])
        
        # shuffle over a member
        clusters[t+1] = 50
        cluster = -50
        O, X = generate_circle_ensemble(0, 20, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        O = np.roll(O, cluster, axis=1)

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 10:
    # ---------------------------------------
    # Set 10: noise test
    # ---------------------------------------
    times = [1, 2, 4, 8]

    for t in times:
        data_list = []
        for mem in members:
            O, F, X = generate_double_penalty_circle_ensemble_with_noise(mem, 80, 20, 0.01*t, path_to_data)

            data_list.append([F, None])

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 11:
    # ---------------------------------------
    # Set 11: P1 and P2?
    # ---------------------------------------
    times = [0, 1]
  
    for t in times:
        data_list = []
        for mem in members:
            O, F, X = generate_case_ensemble('P2', path_to_data)

            data_list.append([F, None])

        if t == 1:
            # add zero memeber - P1
            data_list[0][0] *= 0 

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 12:
    # ---------------------------------------
    # Set 12: Intensity bias, equal mass ==> spatial bias
    # ---------------------------------------
    times = [0, 5, 10, 15, 20, 25, 30]

    for t in times:
        data_list = []
        O, X = generate_circle_ensemble(0, 0, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        mass = O.sum()
        for mem in members:
            # rnage changed with t
            dia_vals = np.arange(40-t, 40+t+1)
            F, X = generate_circle_ensemble(mem, 0, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=rng.choice(dia_vals))
            F /= F.sum()
            F *= mass
            data_list.append([F, None])

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 13:
    # ---------------------------------------
    # Set 13: Spatial bias, circular cases for increasing spread (correct support)
    # ---------------------------------------
    times = [0, 10, 20, 30, 40, 50]
   
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_circle_ensemble(mem, t, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=None)

            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_circle_ensemble(0, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 14:
    # ---------------------------------------
    # Set 14: Spatial bias, circular cases; all too small
    # ---------------------------------------
    times = [0, 10, 20, 30, 40, 50]
 
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_circle_ensemble(mem, t, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=20)

            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_circle_ensemble(0, t, path_to_data)

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 15:
    # ---------------------------------------
    # Set 15: Spatial bias, circular cases; all too large
    # ---------------------------------------

    times = [0, 10, 20, 30, 40, 50]
  
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_circle_ensemble(mem, t, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=60)

            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_circle_ensemble(0, t, path_to_data)

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 16:
    # ---------------------------------------
    # Set 16: Spatial bias, circular cases; random diameters
    # ---------------------------------------
    times = [0, 10, 20, 30, 40, 50]
    
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_circle_ensemble(mem, t, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=rng.choice(np.arange(20, 60)))

            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_circle_ensemble(0, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 17:
    # ---------------------------------------
    # Set 17: intensity error; ellipsises with different heights
    # ---------------------------------------

    times = [0, 1, 2, 3, 4, 5]

    for t in times:
        data_list = []
        amp_scale = [-0.05*t, 0.05*t]

        for mem in members:
            F, X = generate_ellipse_ensemble(mem, 0, path_to_data, seed_pert=rng.integers(0,int(1e7)), case='E1', amp_scale=amp_scale)
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_ellipse_ensemble(0, 0, path_to_data, seed_pert=rng.integers(0,int(1e7)), case='E1', amp_scale=amp_scale)

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 18:
    # ---------------------------------------
    # Set 18: intensity error; two step ellipse in a circle
    # ---------------------------------------

    times = [0, 10, 20, 30, 40, 50]
  
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_intensity_errors_ensemble(mem, rng.normal(scale=0.05*t), 1*t//2, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_intensity_errors_ensemble(0, rng.normal(scale=0.05*t), 1*t//2, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)

        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 19:
    # ---------------------------------------
    # Set 19: extreme event; IN observation
    # ---------------------------------------

    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   
    for t in times:
        data_list = []
        for mem in members:
            if mem in [k for k in range(t)]:
                F, X = generate_extreme_event(mem, 2, 5, 5, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
            else:
                F, X = generate_intensity_errors_ensemble(mem, 0, 0, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_extreme_event(0, 2, 5, 5, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 20:
    # ---------------------------------------
    # Set 20: extreme event; Not in observation
    # ---------------------------------------

    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   
    for t in times:
        data_list = []
        for mem in members:
            if mem in [k for k in range(t)]:
                F, X = generate_extreme_event(mem, 2, 5, 5, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
            else:
                F, X = generate_intensity_errors_ensemble(mem, 0, 0, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_extreme_event(0, 0, 5, 5, path_to_data, seed_pert=rng.integers(0,int(1e7)), diameter=100)
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 21:
    # ---------------------------------------
    # Set 21: Multiscale ellipses with time evolution
    # ---------------------------------------


    # I should abstract this...
    times = [0, 5, 10, 15, 20]
    members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path_to_data = '/home/jjf817/macomp001/jjf817/PhD_jobs/ICP_Cases/MescoVict_cases/'

    # Gather time series data
    data_list = []
    dictionary_in_time = {}
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_multiscale_ellipse_ensemble(mem, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_multiscale_ellipse_ensemble(0, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times


elif set_number == 22:
    # ---------------------------------------
    # Set 22: multiscale under spread : large error!
    # ---------------------------------------

    # I should abstract this...
    times = [0, 5, 10, 15, 20]
    members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path_to_data = '/home/jjf817/macomp001/jjf817/PhD_jobs/ICP_Cases/MescoVict_cases/'

    # Gather time series data
    data_list = []
    dictionary_in_time = {}
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_multiscale_ellipse_ensemble(mem, t/2, path_to_data, seed_pert=rng.integers(0,int(1e7)))
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_multiscale_ellipse_ensemble(0, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times
elif set_number == 23:
    # ---------------------------------------
    # Set 23: multiscale under spread : larger error!
    # ---------------------------------------

    # I should abstract this...
    times = [0, 5, 10, 15, 20]
    members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path_to_data = '/home/jjf817/macomp001/jjf817/PhD_jobs/ICP_Cases/MescoVict_cases/'

    # Gather time series data
    data_list = []
    dictionary_in_time = {}
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_multiscale_ellipse_ensemble(mem, t/3, path_to_data, seed_pert=rng.integers(0,int(1e7)))
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_multiscale_ellipse_ensemble(0, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

elif set_number == 24:
    # ---------------------------------------
    # Set 24: multiscale over spread : larger spread!
    # ---------------------------------------

    # I should abstract this...
    times = [0, 5, 10, 15, 20]
    members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path_to_data = '/home/jjf817/macomp001/jjf817/PhD_jobs/ICP_Cases/MescoVict_cases/'

    # Gather time series data
    data_list = []
    dictionary_in_time = {}
    for t in times:
        data_list = []
        for mem in members:
            F, X = generate_multiscale_ellipse_ensemble(mem, t, path_to_data, seed_pert=rng.integers(0,int(1e7)))
            data_list.append([F, None])
        
        # generate the observation too as a random perturbation
        O, X = generate_multiscale_ellipse_ensemble(0, t/2, path_to_data, seed_pert=rng.integers(0,int(1e7)))
        dictionary_in_time[t] = dict(
            observation = [O, None],
            forecasts = data_list
            )
    dictionary_in_time['times'] = times

else:
    raise ValueError('Set number not recognised {}'.format(set_number))

dictionary_in_time['members'] = members
dictionary_in_time['grid'] = X

# - ---------------------------------------                         
# Plot the density fields
times = dictionary_in_time['times']
M = len(dictionary_in_time['members'])
rows = (len(times)//4 + 1)
cols = 4

fig = plt.figure(figsize=(cols * 5, rows * 4))

for t in times:
    obs = dictionary_in_time[t]['observation']
    forecasts_mean = np.stack([k[0] for k in dictionary_in_time[t]['forecasts']], axis=-1).mean(-1)
    ax = fig.add_subplot(rows, cols, times.index(t)+1)
    ax.set_title(f't = {t}')
    ax.set_facecolor("#f9f6f1")


    im = ax.pcolormesh(
        X[:, :, 0],
        X[:, :, 1],
        np.ma.masked_where(forecasts_mean <= 0, forecasts_mean),
        cmap='Blues',
        shading='auto'
    )
    plt.colorbar(im, ax=ax, label='Forecast Mean')

    ax.contour(
        X[:, :, 0],
        X[:, :, 1],
        obs[0],
        levels=[np.unique(obs[0])[1]-1e-9] if len(np.unique(obs[0])) == 2 else [np.unique(obs[0])[1]-1e-9, np.unique(obs[0])[2]-1e-9],
        # cmap='copper',
        colors="#b98224",
        linestyles='dashdot',
        linewidths=2,
    )

plt.savefig(snakemake.output[1])

# make arrays contigous
for t in times:
    obs = dictionary_in_time[t]['observation']
    obs[0] = np.ascontiguousarray(obs[0])
    forecasts = dictionary_in_time[t]['forecasts']
    for k in range(len(forecasts)):
        forecasts[k][0] = np.ascontiguousarray(forecasts[k][0])

with open(snakemake.output[0], 'wb') as f:
    pickle.dump(dictionary_in_time, f)
