import sys
import copy
import math

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
import solver2

parameter = config.Parameter()

def log_gaussian(x, x_mean, x_stdev):
    return - (x - x_mean)**2 / (2 * x_stdev**2) - 0.5 * np.log(2 * np.pi * x_stdev**2)

def ln_likelihood(THETA):
    theta_i, log_f_a, Delta_N_eff, mu, zeta = THETA
    micro_quark_mass = mu / 2
    N_f = 1
    N_mu = 3
    kappa = (11 * N_mu - 2 * N_f) / 6
    c_N = 0.26 / (zeta**(N_mu - 2) * math.factorial(N_mu - 1) * math.factorial(N_mu - 2))
    eta = kappa + N_f / 2 - 2
    m_psi = mu / 2
    p = (Delta_N_eff, mu, micro_quark_mass, zeta, N_f, N_mu, kappa, c_N, eta, m_psi)
    density_parameter_computed = solver2.compute_relic_density(*p, theta_i, 10**log_f_a)
    return log_gaussian(density_parameter_computed, parameter.Omega_DM_h_sq, parameter.Omega_DM_h_sq_err)

parameter_names = ["Delta_N_eff", "mu", "zeta"]

zeta_mean = 1.34
zeta_err = 0.01

Delta_N_eff_mean = 2.99
Delta_N_eff_err = 0.34

def make_initial_guess():
    theta_i = np.random.uniform(0, np.pi)
    log_f_a = np.random.uniform(15, 18) + 9
    Delta_N_eff = np.random.normal(loc=3.04 - Delta_N_eff_mean, scale=Delta_N_eff_err)
    mu = 10**np.random.uniform(2, 3)
    zeta = np.random.uniform(zeta_mean - zeta_err, zeta_mean + zeta_err)
    return (theta_i, log_f_a, Delta_N_eff, mu, zeta)

def ln_prior(THETA):
    theta_i, log_f_a, Delta_N_eff, mu, zeta = THETA
    if np.all(np.array(THETA[2:]) > 0) and 0 < theta_i <= np.pi and 9 <= log_f_a - 9 <= 18 and \
            2 <= np.log10(mu) <= 3 and abs(zeta - zeta_mean) <= zeta_err:
        return log_gaussian(Delta_N_eff, Delta_N_eff_mean, Delta_N_eff_err)
    else:
        return - np.inf # = log 0

def ln_prob(THETA):
    lp = ln_prior(THETA)
    like = ln_likelihood(THETA)
    if not np.isfinite(lp) or not np.isfinite(like):
        return -np.inf
    return lp + like

ndim = 2 + 3
num_walkers = ndim * 2 * 5
steps = 5000 * 4
num_threads = 4

if __name__ == "__main__":
    pos = [make_initial_guess() for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    # chain: (num_walker, steps, n_dim = #parameters)
    # samples = sampler.chain[:, eq_steps:, :].reshape((-1, ndim))
    # save the data
    filename = config.data_path + "/micro_qcd_parameter.npz"
    np.savez(filename, samples=sampler.chain)
