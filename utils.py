import torch
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from time import process_time
import pickle
from mmuot import mmuot_dual_cost, generate_mmuotdataprocessor_star_graph, mmuot_sinkhorn_loop, generate_mmuot_debiasing_dp

import numpy as np
import netCDF4 as nc


def guassian_kernel(mu, sigma, N=500, range=(-3, 3)):
    density = ((torch.linspace(1 / (2 * N), 1 - 1 / (2 * N), N) - 0.5) / 0.5) * (
        range[1]
    )
    density = torch.exp(-0.5 * ((density - mu) / sigma) ** 2)
    density = density / density.sum()

    return [
        density.to(torch.float64),
        torch.cartesian_prod(
            torch.linspace(1 / (2 * N), 1 - 1 / (2 * N), N),
            torch.Tensor([0.0]),
        )
        .view(N, 2)
        .to(torch.float64),
    ]


def plot_densities(data, title="Densities", file_title=None, figsize=(8, 5)):

    plt.figure(figsize=figsize, dpi=200)

    line_styles = ["--", "-.", ":", (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (1, 1))]

    for i, (density, grid) in enumerate(data):
        # detach for plotting
        density = density.detach().cpu()
        grid = grid.detach().cpu()

        x = grid[:, 0].numpy()
        y = density.numpy()
        plt.plot(
            x, y, label=f"Density {i+1}", linestyle=line_styles[i % len(line_styles)]
        )

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if file_title is None:
        plt.savefig("density_all_together.pdf")
    else:
        plt.savefig(file_title)


# ----------------------------------------------------
#  True Guassian barycentre; given in paper
# ----------------------------------------------------


def true_sigma_barycentre(sigma_array, weights, epsilon):

    assert weights.shape == sigma_array.shape

    if len(torch.unique(sigma_array)) == 1:
        return sigma_array[0]

    def LHS(S):
        return weights.dot(torch.sqrt((epsilon / 2) ** 2 + 4 * sigma_array**2 * S**2))

    def RHS(S):
        return torch.sqrt((epsilon / 2) ** 2 + 4 * S**4)

    f = lambda S: LHS(S) - RHS(S)

    return root_scalar(
        f, bracket=[sigma_array.min(), sigma_array.max()], method="bisect"
    ).root


def true_gaussian_barycentre(mu_array, sigma_array, weights, epsilon, N=500):

    assert weights.sum() == 1, "Weights must sum to 1"

    mu_average = weights.dot(mu_array)
    sigma_average = true_sigma_barycentre(sigma_array, weights, epsilon)

    return guassian_kernel(mu_average, sigma_average, N=N)


def generate_guassian_densities_with_true_barycentre(
    mu_array, sigma_array, weights, epsilon, N=500, range=(-3, 3)
):
    data = []

    true_density, true_grid = true_gaussian_barycentre(
        mu_array, sigma_array, weights, epsilon, N=N
    )
    data.append([true_density, true_grid])

    for mu, sigma in zip(mu_array, sigma_array):
        density, grid = guassian_kernel(mu, sigma, N=N, range=range)
        data.append([density, grid])

    return data


def generate_guassian_densities_with_uniform_barycentre(
    mu_array, sigma_array, weights, epsilon, N=500, range=(-3, 3)
):
    data = []

    true_density, true_grid = true_gaussian_barycentre(
        mu_array, sigma_array, weights, epsilon, N=N
    )
    # Not using the actual true one
    true_density = torch.ones_like(true_density)
    true_density /= true_density.sum()
    data.append([true_density, true_grid])

    for mu, sigma in zip(mu_array, sigma_array):
        density, grid = guassian_kernel(mu, sigma, N=N, range=range)
        data.append([density, grid])

    return data


def generate_circle_ensemble(member, r, path_to_data, seed_pert=1, diameter=None):
    rng_state = np.random.RandomState((member + r**2)*12357*seed_pert)
    
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'C1', 'C1', path_to_data=path_to_data
    )
    if diameter is not None:
        centre = [100, 100]
        X_precipitation *= 0  # reset to zero
        for i,j in np.ndindex(X_precipitation.shape):
            if (i - centre[0])**2 + (j - centre[1])**2 <= (diameter/2)**2:
                X_precipitation[i,j] = 1
            else:
                X_precipitation[i,j] = 0

        # scale back 
        X_precipitation /= 1873.5

    # Create same with rejction sampling + shift
    dx = r + 1
    dy = r + 1
    while dx**2 + dy**2 > r**2:
        dx = rng_state.randint(-r, r+1, size=1) 
        dy = rng_state.randint(-r, r+1, size=1) 

    X_precipitation = np.roll(X_precipitation, dx, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    return X_precipitation, X_coordinates


def generate_intensity_errors_ensemble(member, amp, r, path_to_data, seed_pert=1, diameter=100, x_shift=-20):
    """
    amp is proportional to 1, and then the whole field is normalised by 1873.5 at the end
    """
    rng_state = np.random.RandomState((member + r**2+diameter)*1357*seed_pert)

    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'E5', 'E5', path_to_data=path_to_data
    )
    # Create same with rejction sampling + shift
    if r > 0:
    
        dx = r + 1
        dy = r + 1
        while dx**2 + dy**2 > r**2:
            dx = rng_state.randint(-r, r+1, size=1) 
            dy = rng_state.randint(-r, r+1, size=1) 
    else:
        dx, dy = 0, 0

    X_precipitation = np.roll(X_precipitation, dx + x_shift, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    if diameter is not None:
        centre = [100, 100]
        X_precipitation[X_precipitation > 0 ] = amp # reset the ellispe to amps
        for i,j in np.ndindex(X_precipitation.shape):
            if (i - centre[0])**2 + (j - centre[1])**2 <= (diameter/2)**2:
                X_precipitation[i,j] += 1
            else:
                X_precipitation[i,j] = 0

        # scale back 
        X_precipitation /= 1873.5

    return X_precipitation, X_coordinates


def generate_extreme_event(member, amp, r, ext_diam, path_to_data, seed_pert=1, diameter=100, x_shift=-20):
    rng_state = np.random.RandomState((member + r**2 + ext_diam)*123*seed_pert)
    
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'C1', 'C1', path_to_data=path_to_data
    )
    X_precipitation *= 0  # reset to zero

    if ext_diam is not None:
        centre = [100, 100]
        X_precipitation *= 0  # reset to zero
        for i,j in np.ndindex(X_precipitation.shape):
            if (i - centre[0])**2 + (j - centre[1])**2 <= (ext_diam/2)**2:
                X_precipitation[i,j] = 1
            else:
                X_precipitation[i,j] = 0

        # scale back at the end
        # X_precipitation /= 1873.5
    
    # Create same with rejction sampling + shift
    if r > 0:
        dx = r + 1
        dy = r + 1
        while dx**2 + dy**2 > r**2:
            dx = rng_state.randint(-r, r+1, size=1) 
            dy = rng_state.randint(-r, r+1, size=1) 
    else:
        dx, dy = 0, 0

    X_precipitation = np.roll(X_precipitation, dx + x_shift, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    if diameter is not None:
        centre = [100, 100]
        X_precipitation[X_precipitation > 0 ] = amp 
        for i,j in np.ndindex(X_precipitation.shape):
            if (i - centre[0])**2 + (j - centre[1])**2 <= (diameter/2)**2:
                X_precipitation[i,j] += 1
            else:
                X_precipitation[i,j] = 0


    else:
        raise Warning("You should really have a diameter for the extreme event!")
    
    # scale back 
    X_precipitation /= 1873.5
    
    return X_precipitation, X_coordinates

def generate_two_circle_event_clusters(member, r, centre1, centre2, path_to_data, seed_pert=1):
    rng_state = np.random.RandomState((member + r**2)*2227*seed_pert)
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'C1', 'C1', path_to_data=path_to_data
    )

    # Create same with rejction sampling + shift
    dx = r + 1
    dy = r + 1
    while dx**2 + dy**2 > r**2:
        dx = rng_state.randint(-r, r+1, size=1) 
        dy = rng_state.randint(-r, r+1, size=1) 

    temp = X_precipitation.copy()
    X_precipitation = np.roll(X_precipitation, dx + centre1[0], axis=1)
    X_precipitation = np.roll(X_precipitation, dy + centre1[1], axis=0)

    # and again for second event
    # Create same with rejction sampling + shift
    dx = r + 1
    dy = r + 1
    while dx**2 + dy**2 > r**2:
        dx = rng_state.randint(-r, r+1, size=1) 
        dy = rng_state.randint(-r, r+1, size=1) 

    temp = np.roll(temp, dx + centre2[0], axis=1)
    temp = np.roll(temp, dy + centre2[1], axis=0)
    X_precipitation += temp

    return X_precipitation, X_coordinates


def generate_ellipse_ensemble(member, r, path_to_data, seed_pert=1, case='E1', amp_scale=0):
    rng_state = np.random.RandomState((member + r**2)*3334*seed_pert)
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        case, case, path_to_data=path_to_data
    )

    if amp_scale != 0:
        X_precipitation *= rng_state.normal(loc=X_precipitation.max(), scale=amp_scale)

    # Create same with rejction sampling + shift
    if r > 0:
        dx = r + 1
        dy = r + 1
        while dx**2 + dy**2 > r**2:
            dx = rng_state.randint(-r, r+1, size=1) 
            dy = rng_state.randint(-r, r+1, size=1)
    else:
        dx, dy = 0, 0

    X_precipitation = np.roll(X_precipitation, dx, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    return X_precipitation, X_coordinates


def generate_double_penalty_circle_ensemble(member, t, r, path_to_data):
    rng_state = np.random.RandomState(int(member + t**2 + r)*12333)

    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'C1', 'C1', path_to_data=path_to_data
    )

    # Create same with rejction sampling + shift
    dx = r + 1
    dy = r + 1
    while dx**2 + dy**2 > r**2:
        dx = rng_state.randint(-r, r+1, size=1) 
        dy = rng_state.randint(-r, r+1, size=1) 

    # move them back 50 points
    Y_precipitation = np.roll(Y_precipitation, -50, axis=1)
    X_precipitation = np.roll(X_precipitation, dx + t - 50, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    return Y_precipitation,  X_precipitation, X_coordinates


def generate_case_ensemble(case, path_to_data):
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        case, case, path_to_data=path_to_data
    )

    return Y_precipitation,  X_precipitation, X_coordinates

def generate_double_penalty_circle_ensemble_with_noise(member, t, r, noise, path_to_data):
    rng_state = np.random.RandomState(int(member + t**2 + r)*353535)
    
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'C1', 'C1', path_to_data=path_to_data
    )

    # Create same with rejction sampling + shift
    dx = r + 1
    dy = r + 1
    while dx**2 + dy**2 > r**2:
        dx = rng_state.randint(-r, r+1, size=1) 
        dy = rng_state.randint(-r, r+1, size=1) 

    # move them back 50 points
    Y_precipitation = np.roll(Y_precipitation, -50, axis=1)
    X_precipitation = np.roll(X_precipitation, dx + t - 50, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    # add noise
    N = np.prod(X_precipitation.shape)
    k = int(noise * N)
    max_val = X_precipitation.max() # it'll be the same everywhere


    # reject sampling to add noise only where zero
    while k > 0:
        idx = rng_state.choice(N, size=1, replace=False)
        if X_precipitation.flat[idx] == 0:
            X_precipitation.flat[idx] = max_val
            k -= 1

    return Y_precipitation,  X_precipitation, X_coordinates


def generate_cone_section_ensemble(member, r1, r0, angle, path_to_data):
    rng_state = np.random.RandomState(int(member + r0+r1**2*angle)*4567)
    X_coordinates, X_precipitation, Y_coordinates, Y_precipitation, mass_x, mass_y = load_test_fields_bias_scaling(
        'C1', 'C1', path_to_data=path_to_data
    )


    # Create same with rejction sampling + shift
    dx = r1 + 1
    dy = r1 + 1
    def condition(dx, dy):
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        if angle==0:
            return (r0 <= r <= r1) 
        else:
            return (r0 <= r <= r1) and ( -angle <= theta <= angle)

    while not condition(dx, dy):
        dx = rng_state.randint(-r1, r1+1, size=1) 
        dy = rng_state.randint(-r1, r1+1, size=1) 

    # move them back 50 points
    # Y_precipitation = np.roll(Y_precipitation, -50, axis=1)
    X_precipitation = np.roll(X_precipitation, dx - 50, axis=1)
    X_precipitation = np.roll(X_precipitation, dy, axis=0)

    return X_precipitation, X_coordinates

# ELLIPSE
def ellipse_mask(center, axes, angle, shape=(200,200)):
    cx, cy = center
    axis_x, axis_y = axes
    a = axis_x / 2
    b = axis_y / 2
    theta = angle
    xx, yy = np.meshgrid(
        np.linspace(1, shape[0], shape[0]),
        np.linspace(1, shape[1], shape[1]),
        indexing='ij'
    )

    # Translate coordinates to center the ellipse at the origin
    x = xx - cx
    y = yy - cy

    # rotate
    xp =  np.cos(theta) * x + np.sin(theta) * y
    yp = -np.sin(theta) * x + np.cos(theta) * y

    mask = (xp**2) / a**2 + (yp**2) / b**2 <= 1
    ellipse = np.zeros_like(xx)
    ellipse[mask] = 1.0

    return ellipse.T

def random_circluar_pert(X, r, rng_state):
    # I should pass on a random number generator
    dx = r + 1
    dy = r + 1
    while dx**2 + dy**2 > r**2:
            dx = rng_state.randint(-r, r+1, size=1) 
            dy = rng_state.randint(-r, r+1, size=1) 

    X = np.roll(X, dx, axis=1)
    X = np.roll(X, dy, axis=0)
    return X

def generate_rotated_ellipse_ensemble(member, angle_r, path_to_data, seed_pert=1):
    # same mass but different rotations
    rng_state = np.random.RandomState(int(member + angle_r**2)*5678*seed_pert)

    X_coordinates, _, _, _, _, _ = load_test_fields_bias_scaling(
        'E5', 'E5', path_to_data=path_to_data
    )
    
    X_precipitation = ellipse_mask(
        center=(100, 100),
        axes=(100,40),
        angle= np.pi/4 + rng_state.uniform(-angle_r, angle_r)
        )
    # scale back
    X_precipitation /= 1873.5

    return X_precipitation, X_coordinates

def generate_multiscale_ellipse_ensemble(member, r, path_to_data, seed_pert=1):
    # same mass but different scales
    rng_state = np.random.RandomState(int(member + r**2)*5432*seed_pert)
    
    X_coordinates, X_precipitation, _, _, _, _ = load_test_fields_bias_scaling(
        'E5', 'E5', path_to_data=path_to_data
    )
    fields = []
    # fix average centre of big event
    fields.append(ellipse_mask(
        center=(150,50),
        axes=(100,40),
        angle=np.pi/4
        ))
    
    # medium events
    centres = [(110, 60), (100, 100), (150, 115)]
    for _ in range(3):
        fields.append(ellipse_mask(
        center=centres[_],
        axes=(50,20),
        angle=np.pi/4
        ))

    # small events
    centres = [(70, 160), (90, 150), (30, 140), (80, 170), (60, 90), (40, 120), (10, 110)]
    for _ in range(7):
        fields.append(ellipse_mask(
        center=centres[_],
        axes=(20,5),
        angle=np.pi/4
        ))

    # sum and perturb
    X_precipitation *= 0  # reset to zero
    for ellipse in fields:
        X_precipitation += random_circluar_pert(ellipse, r, rng_state)

    # cap at one so that overlaps are flattened not increased intensity
    # scale back to mass 1
    X_precipitation[X_precipitation>1] = 1.0
    
    X_precipitation /= 1873.5

    return X_precipitation, X_coordinates

def process_array(x, device='cuda:0'):
    # Keeps tensors intact; converts numpy/other types safely
    if torch.is_tensor(x):
        return x.to(dtype=torch.float64, device=device)
    else:
        return torch.as_tensor(x, dtype=torch.float64, device=device)

def load_test_fields_bias_scaling(fieldx: str, fieldy: str, L=200, dtype=torch.float64, path_to_data='',cases_scale=1873.5):

    # Load data
    X_i = nc.Dataset(
        path_to_data + str(fieldx) + ".nc"
    )

    # For the 10 grid point increment cases we luse C1 rolled over 10 grid points.
    if type(fieldy) == int:
        # Repeat the x field then shift it through 'roll' later
        Y_j = nc.Dataset(
        path_to_data + str(fieldx) + ".nc"
    )
    else:
        Y_j = nc.Dataset(
            path_to_data + str(fieldy) + ".nc"
        )

    # Extract scaled fields
    X_coordinates = torch.stack(
        torch.meshgrid(
            torch.tensor(X_i["x"][:].__array__(), dtype=dtype) / L,
            torch.tensor(X_i["y"][:].__array__(), dtype=dtype) / L,
            indexing="xy",
        ),
        axis=2,
    )
    X_precipitation = X_i["var2d"][:].__array__()
    Y_coordinates = torch.stack(
        torch.meshgrid(
            torch.tensor(Y_j["x"][:].__array__(), dtype=dtype) / L,
            torch.tensor(Y_j["y"][:].__array__(), dtype=dtype) / L,
            indexing="xy",
        ),
        axis=2,
    )
    Y_precipitation = Y_j["var2d"][:].__array__()

    mass_x, mass_y = np.sum(X_precipitation), np.sum(Y_precipitation)

    X_precipitation /= cases_scale
    Y_precipitation /= cases_scale

    if type(fieldy) == int:
        return (
            X_coordinates,
            X_precipitation,
            Y_coordinates,
            np.roll(Y_precipitation, fieldy, axis=1),
            mass_x,
            mass_y,
        )
    else:
        return (
            X_coordinates,
            X_precipitation,
            Y_coordinates,
            Y_precipitation,
            mass_x,
            mass_y,
        )

import torch
import matplotlib.pyplot as plt
import numpy as np


def rai_plot_barycentre_densities(data, title="Densities", file_title=None, cmap='viridis', s=1, boot_ind=None):

    plt.set_cmap(cmap)

    no_samples = len(data)
    ncols = 3
    # one extra for the mean
    nrows = int(np.ceil((no_samples + 1)  / ncols))

    mean_average = torch.zeros_like(data[0][0]).squeeze().cpu()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), dpi=200)
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    if boot_ind is None:
        boot_ind = list(range(len(data)))

    for i, (density, grid) in enumerate(data):
        density, grid = density.cpu().view(-1,1), grid.cpu()
        ax = axes[i]
        mask = density.numpy() >= 0
        mask = mask.squeeze()
        sc = ax.scatter(
            grid[mask, 0].numpy(), grid[mask, 1].numpy(), c=density[mask].numpy(), s=s, cmap=cmap
        )
        if i == 0:
            ax.set_title(f"Observation Density ")
        elif i == 1:
            ax.set_title(f"Barycentre Density ")
        else:
            ax.set_title(f"Forecast Density {boot_ind[i-2]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax)

        if i > 1:
            mean_average += density.squeeze()
    
    mean_average /= (no_samples - 2)

    i += 1
    ax = axes[i]
    mask = mean_average.numpy() >= 0
    mask = mask.squeeze()
    sc = ax.scatter(
        grid[mask, 0].numpy(), grid[mask, 1].numpy(), c=mean_average[mask].numpy(), s=s, cmap=cmap
    )
    ax.set_title(f"Ensemble Mean")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if file_title is None:
        plt.savefig("density_all_together.png")
    else:
        plt.savefig(file_title)
    plt.close(fig)


def mmuot_general_costings(
    centre_data,
    leaf_data,
    epsilon, 
    rho, 
    aprox_type, 
    max_iterates, 
    tol, 
    grid, 
    device='cuda:0'
    ):

    cost_dict = {}

    t0 = process_time()
    data_list = [centre_data] + leaf_data
    dp_for_mmuot_bary = generate_mmuotdataprocessor_star_graph(
        data_list, grid=grid, weights=None, cuda_device=device, clear_grid=True
    )

    dp_for_mmuot_bary, conv_list_bary = mmuot_sinkhorn_loop(
        dp_for_mmuot_bary,
        epsilon,
        rho,
        max_iterations=max_iterates,
        tol=tol,
        aprox=aprox_type,
        prod=True,
        convergence_tracking=True,
        verbose=False,
    )

    cost_bary, _ = mmuot_dual_cost(dp_for_mmuot_bary, epsilon, rho=rho, aprox=aprox_type, prod=True)

    t1 = process_time()

    cost_dict['bias'] = cost_bary
    cost_dict['bary_time'] = t1 - t0

    # Debiasing terms:
    tlist = np.zeros(len(data_list))
    debiased_costs = np.zeros(len(data_list))

    for k, data in enumerate(data_list):
        t_debias_start = process_time()
        dp = generate_mmuot_debiasing_dp(data[0], grid=grid, members=len(leaf_data), cuda_device=device, clear_grid=True)

        dp = mmuot_sinkhorn_loop(
            dp,
            epsilon,
            rho,
            max_iterations=max_iterates,
            tol=tol,
            aprox=aprox_type,
            prod=True,
            convergence_tracking=False,
            verbose=False,
        )
        
        cost, _  = mmuot_dual_cost(dp, epsilon, rho=rho, aprox=aprox_type, prod=True)

        cost_dict[f'debias_{k}'] = cost
        debiased_costs[k] = cost

        t_debias_end = process_time()
        tlist[k] = t_debias_end - t_debias_start

    cost_dict['debias_times'] = tlist
    cost = cost_bary - 1/(len(leaf_data)+1)*debiased_costs.sum()
    
    return cost, cost_dict

