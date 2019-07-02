import multiprocessing as mp
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

import config
import eom
import axion_mass
import g_star
import potential

num_samples = int(sys.argv[1])
num_threads = int(sys.argv[2])
theta_i = float(sys.argv[3])
f_a = float(sys.argv[4]) * 1e9 # input in GeV
filename = sys.argv[5]

model = eom.Model(axion_mass.m_a_from_chi_general, g_star.matched, potential.cosine, None)

# use the fllowing directly: h T0 rho_c

def sample(i):
    print(i)
    params = config.Parameter()

    params.M_pl = np.random.normal(loc=params.M_pl, scale=params.M_pl_err)
    params.Lambda_QCD = np.random.normal(loc=params.Lambda_QCD, scale=params.Lambda_QCD_err)
    params.m_u = np.random.normal(loc=params.m_u, scale=params.m_u_err)
    params.m_d = np.random.normal(loc=params.m_d, scale=params.m_d_err)
    params.m_pi0 = np.random.normal(loc=params.m_pi0, scale=params.m_pi0_err)
    params.f_pi0 = np.random.normal(loc=params.f_pi0, scale=params.f_pi0_err)
    params.T0 = np.random.normal(loc=params.T0, scale=params.T0_err)

    model.parameter = params
    solver = model.get_solver(theta_i, f_a)
    return solver.compute_density_parameter()

with mp.Pool(num_threads) as poolparty:
    samples = poolparty.map(sample, range(num_samples))

np.savetxt(filename, samples)

