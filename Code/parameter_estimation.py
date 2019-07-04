import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import emcee
import corner

import axion_mass
import potential
import g_star
import eom
import config

parameter = config.Parameter()

def ln_likelihood(THETA):
    print("*", end=""); sys.stdout.flush()
    model = eom.Model(axion_mass.m_a_from_chi_general, g_star.matched, potential.cosine, parameter)
    theta_i, f_a, parameter.M_pl, parameter.Lambda_QCD, parameter.m_u, parameter.m_d, parameter.m_pi0, parameter.f_pi0, parameter.T0, parameter.rho_c = THETA
    solver = model.get_solver(theta_i, f_a)
    density_parameter_computed = solver.compute_density_parameter()
    return - (density_parameter_computed - parameter.Omega_DM_h_sq)**2 / (2 * parameter.Omega_DM_h_sq_err**2)

parameter_names = ["theta_i", "f_a", "M_pl", "Lambda_QCD", "m_u", "m_d", "m_pi", "f_pi", "T0", "rho_c"]
errors = [0.0, 0.0, parameter.M_pl_err, parameter.Lambda_QCD_err, parameter.m_u_err, parameter.m_d_err,
          parameter.m_pi0_err, parameter.f_pi0_err, parameter.T0_err, parameter.rho_c_err]

inital_guess = np.array((
    1, 1e12 * 1e9, parameter.M_pl, parameter.Lambda_QCD,
    parameter.m_u, parameter.m_d, parameter.m_pi0,
    parameter.f_pi0, parameter.T0, parameter.rho_c
))

# TODO: we dont care about normalization, right?

ln_err = (
    np.log(parameter.M_pl_err) + np.log(parameter.Lambda_QCD_err) +
    np.log(parameter.m_u_err) + np.log(parameter.m_d_err) +
    np.log(parameter.m_pi0_err) + np.log(parameter.f_pi0_err) +
    np.log(parameter.T0_err) + np.log(parameter.rho_c_err)
)

def ln_prior(THETA):
    theta_i, f_a, M_pl, Lambda_QCD, m_u, m_d, m_pi0, f_pi0, T0, rho_c = THETA
    if np.all(np.array(THETA) > 0) and -np.pi <= theta_i <= np.pi and 9 <= np.log10(f_a / 1e9) <= 18:
        return ln_err
    else:
        return - np.inf # = log 0

def ln_prob(THETA):
    lp = ln_prior(THETA)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(THETA)

num_walkers = len(inital_guess) * 2 * 10
steps = 200
eq_steps = 50
num_threads = 4
ndim = len(inital_guess) # 2 + 2 + 3 + 3
# pos = [res["x"] + 1e-4 * np.random.randn(ndim) for i in range(num_walkers)]

if __name__ == "__main__":
    pos = [inital_guess + 1e-4 * np.random.randn(ndim) * inital_guess for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    samples = sampler.chain[:, eq_steps:, :].reshape((-1, ndim))

    # chain: (num_walker, steps, n_dim = #parameters)

    #                          mean value, lower error, upper error
    parameter = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    # save the data
    filename = config.data_path + "/parameter.npz"
    np.savez(filename, parameter=list(parameter), samples=samples)
