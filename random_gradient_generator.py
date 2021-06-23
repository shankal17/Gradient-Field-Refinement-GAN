"""Contains functions to generate a random surface function dataset

Functions
---------
generate_random_function(start, stop, step)
    Generates a grid of points located on a random surface function
save_reandom_function(base_name, num_samples, domain=[-10, 10.1], step=0.1)
    Saves a set of randomly generated surface functions to specified directory
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm


def generate_random_function(start, stop, step):
    """Generates a grid of points located on a random surface function

    Parameters
    ----------
    start : float
        Lower bound of domain (domain is square in xy plane)
    stop : float
        Upper bound of domain (domain is square in xy plane)
    step : float
        Interval between consecutive points along each x and y axis

    Returns
    -------
    numpy.array
        X data
    numpy.array
        Y data
    numpy.array
        Z data
    """

    x = np.arange(start, stop, step)
    y = np.arange(start, stop, step)
    X, Y = np.meshgrid(x, y)
    include = np.random.uniform(0.0, 1.0, 13) // 0.5
    coefficients = np.multiply(include, np.random.uniform(-5.0, 5.0, 13))
    trig_freq_terms = np.random.uniform(0, 2, 4)
    horizontal_transforms = np.random.uniform(-3, 3, 12)

    proposed = np.array([0.01*(X+horizontal_transforms[0])**3, 0.05*(X+horizontal_transforms[1])**2, 0.1*(X+horizontal_transforms[2]), 0.01*(Y+horizontal_transforms[3])**3, 0.05*(Y+horizontal_transforms[4])**2, 0.1*(Y+horizontal_transforms[5]), 1, 0.01*(X+horizontal_transforms[6])*(Y+horizontal_transforms[7]), 5*np.sin(trig_freq_terms[0]*(X+horizontal_transforms[8])), 5*np.sin(trig_freq_terms[1]*(Y+horizontal_transforms[9])), 5*np.cos(trig_freq_terms[2]*(X+horizontal_transforms[10])), 5*np.cos(trig_freq_terms[3]*(Y+horizontal_transforms[11]))])
    Z = np.sum(np.array([coefficients[i]*proposed[i] for i in range(len(proposed))]), axis=0)
    return X, Y, Z

def save_random_function(base_name, num_samples, domain=[-10, 10.1], step=0.1):
    """Generates a grid of points located on a random surface function

    Parameters
    ----------
    base_name : float
        Base name of function file names
    num_samples : int
        Number of functions to generate and save
    domain : list, optional
        Domain of both the x and y axis
    step : float, optional
        Interval between consecutive points along each x and y axis
    """

    for i in tqdm(range(num_samples)):
        x, y, z = generate_random_function(domain[0], domain[1], step)
        dy, dx = np.gradient(z)
        stacked_decomposed_grad = np.array([x, y, z, dx, dy], dtype=np.float32)
        name = base_name + str(i)
        np.save(name, stacked_decomposed_grad)

if __name__ == '__main__':
    save_random_function('Random_Surfaces/func_', 50000, domain=(-4, 4), step=0.1)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    loaded_function = np.load('Random_Surfaces/func_0.npy')
    x = loaded_function[0]
    y = loaded_function[1]
    z = loaded_function[2]
    dx = loaded_function[3]
    dy = loaded_function[4]
    dz = np.zeros_like(dx)
    surf = ax.plot_surface(x, y, z, linewidth=0)
    # ax.quiver(x, y, np.ones_like(x)*z.min(), dx, dy, np.zeros_like(dx), length=0.05)
    print(dz.shape)
    plt.show()