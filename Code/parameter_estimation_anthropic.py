import sys
import copy

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
import solver

parameter = config.Parameter()
p = copy.deepcopy(parameter)

# fixed theta value
theta_i = 1e-4

def log_gaussian(x, x_mean, x_stdev):
    return - (x - x_mean)**2 / (2 * x_stdev**2) - 0.5 * np.log(2 * np.pi * x_stdev**2)

def ln_likelihood(THETA):
    log_f_a, p.M_pl, p.Lambda_QCD, p.m_u, p.m_d, p.m_pi0, p.f_pi0, p.T0, p.rho_c = THETA
    density_parameter_computed = solver.compute_relic_density(p, theta_i, 10**log_f_a)
    return log_gaussian(density_parameter_computed, parameter.Omega_DM_h_sq, parameter.Omega_DM_h_sq_err)

parameter_names = ["log_f_a", "M_pl", "Lambda_QCD", "m_u", "m_d", "m_pi", "f_pi", "T0", "rho_c"]
errors = [-1, parameter.M_pl_err, parameter.Lambda_QCD_err, parameter.m_u_err, parameter.m_d_err,
          parameter.m_pi0_err, parameter.f_pi0_err, parameter.T0_err, parameter.rho_c_err]
mean_values = np.array((17, parameter.M_pl, parameter.Lambda_QCD, parameter.m_u, parameter.m_d,
    parameter.m_pi0, parameter.f_pi0, parameter.T0, parameter.rho_c))

def make_initial_guess():
    return np.concatenate([
        [np.random.uniform(9, 18) + 9],
        # [np.random.uniform(17, 18) + 9],
        np.random.normal(loc=mean_values[1:], scale=errors[1:]),
    ])

def ln_prior(THETA):
    log_f_a, M_pl, Lambda_QCD, m_u, m_d, m_pi0, f_pi0, T0, rho_c = THETA
    if np.all(np.array(THETA[2:]) > 0) and 9 <= log_f_a - 9 <= 18:
        return (
            log_gaussian(M_pl, parameter.M_pl, parameter.M_pl_err) +
            log_gaussian(Lambda_QCD, parameter.Lambda_QCD, parameter.Lambda_QCD_err) +
            log_gaussian(m_u, parameter.m_u, parameter.m_u_err) +
            log_gaussian(m_d, parameter.m_d, parameter.m_d_err) +
            log_gaussian(m_pi0, parameter.m_pi0, parameter.m_pi0_err) +
            log_gaussian(f_pi0, parameter.f_pi0, parameter.f_pi0_err) +
            log_gaussian(T0, parameter.T0, parameter.T0_err) +
            log_gaussian(rho_c, parameter.rho_c, parameter.rho_c_err)
        )
    else:
        return - np.inf # = log 0

def ln_prob(THETA):
    lp = ln_prior(THETA)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(THETA)

ndim = len(make_initial_guess())
num_walkers = ndim * 2 * 5 * 2
steps = 5000 * 2
num_threads = 4

if __name__ == "__main__":
    pos = [make_initial_guess() for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    filename = config.data_path + "/anthropic_parameter.npz"
    np.savez(filename, samples=sampler.chain)
