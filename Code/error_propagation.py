import multiprocessing as mp
import sys

import numpy as np
import matplotlib.pyplot as plt

import config
import eom
import axion_mass
import g_star
import potential

theta_i = 1
f_a = 1e9 * 1e12
M_pl_err = 0.000014e19 * 1e9 / np.sqrt(8*np.pi)
model = eom.Model(axion_mass.m_a_from_chi_general, g_star.matched, potential.cosine, None)
num_samples = int(sys.argv[1])
num_threads = int(sys.argv[2])

def sample(nothing):
    params = config.Parameter()
    params.M_pl = np.random.normal(loc=params.M_pl, scale=M_pl_err)
    model.parameter = params
    solver = model.get_solver(theta_i, f_a)
    return solver.compute_density_parameter()

with mp.Pool(num_threads) as poolparty:
    samples = poolparty.map(sample, [None]*num_samples)

plt.hist(samples)
plt.xlabel(r"$\Omega_a h^2$")
plt.ylabel("Count")
plt.show()


