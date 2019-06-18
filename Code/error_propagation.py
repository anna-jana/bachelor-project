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
f_a = 1e12 * 1e9

model = eom.Model(axion_mass.m_a_from_chi_general, g_star.matched, potential.cosine, None)
M_pl_err = 0.000014e19 * 1e9 / np.sqrt(8*np.pi)
Lambda_QCD_err = 20e6

# what about the twosided error?
m_u_err = (0.49 + 0.26) / 2 * 1e6
m_d_err = (0.48 + 0.17) / 2 * 1e6
m_pi0_err = 0.0005 * 1e6
f_pi0_err = 5e6

# use the fllowing directly: h T0 rho_c

def sample(i):
    print(i)
    params = config.Parameter()

    params.M_pl = np.random.normal(loc=params.M_pl, scale=M_pl_err)
    params.Lambda_QCD = np.random.normal(loc=params.Lambda_QCD, scale=Lambda_QCD_err)
    params.m_u = np.random.normal(loc=params.m_u, scale=m_u_err)
    params.m_d = np.random.normal(loc=params.m_d, scale=m_d_err)
    params.m_pi0 = np.random.normal(loc=params.m_pi0, scale=m_pi0_err)
    params.f_pi0 = np.random.normal(loc=params.f_pi0, scale=f_pi0_err)

    model.parameter = params
    solver = model.get_solver(theta_i, f_a)
    return solver.compute_density_parameter()

if __name__ == "__main__":
    num_samples = int(sys.argv[1])
    num_threads = int(sys.argv[2])
    with mp.Pool(num_threads) as poolparty:
        samples = poolparty.map(sample, range(num_samples))
    np.savetxt("samples.txt", samples)

