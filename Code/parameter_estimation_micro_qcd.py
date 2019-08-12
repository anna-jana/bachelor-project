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
import time_temp

def log_gaussian(x, x_mean, x_stdev):
    return - (x - x_mean)**2 / (2 * x_stdev**2) - 0.5 * np.log(2 * np.pi * x_stdev**2)

def compute_Omega(theta_i, f_a, Delta_N_eff, mu, zeta):
    print("*", end="")
    sys.stdout.flush()
    g = g_star.make_micro(Delta_N_eff)
    T_i = time_temp.find_T_osc(f_a, lambda T, f: axion_mass.micro_m_a(T, f, mu, zeta), g)
    if T_i <= config.parameter.T_eq:
        print("WARNING: f_a = %e, theta_i = %e, %e vs %e, %e vs %e, %e vs %e oscillation after T_eq = %e > T_i = %e" %
                (f_a, theta_i, Delta_N_eff, Delta_N_eff_mean, mu, 1e2, zeta, zeta_mean, config.parameter.T_eq, T_i))
        return np.nan
    T_f = config.parameter.T0
    m_i = axion_mass.micro_m_a(T_i, f_a, mu, zeta)
    m_f = axion_mass.micro_m_a(T_f, f_a, mu, zeta)
    g_s_f = g.g_s(T_f)
    g_s_i = g.g_s(T_i)
    rho = 0.5 * m_i * m_f * theta_i**2 * g_s_f / g_s_i * (T_f / T_i)**3
    Omega = rho / config.parameter.rho_c * config.parameter.h**2
    return Omega

def ln_likelihood(THETA):
    theta_i, log_f_a, Delta_N_eff, mu, zeta = THETA
    f_a = 10**log_f_a
    density_parameter_computed = compute_Omega(theta_i, f_a, Delta_N_eff, mu, zeta)
    return log_gaussian(density_parameter_computed, config.parameter.Omega_DM_h_sq, config.parameter.Omega_DM_h_sq_err)

parameter_names = ["Delta_N_eff", "mu", "zeta"]

zeta_mean = 1.34
zeta_err = 0.01

N_eff_mean = 2.99
N_eff_err = 0.34
Delta_N_eff_mean = 3.04 - N_eff_mean
Delta_N_eff_err = N_eff_err

def make_initial_guess():
    theta_i = np.random.uniform(0, np.pi)
    log_f_a = np.random.uniform(15, 18) + 9
    Delta_N_eff = np.random.normal(loc=Delta_N_eff_mean, scale=Delta_N_eff_err)
    mu = 10**np.random.uniform(2, 3)
    zeta = np.random.uniform(zeta_mean - zeta_err, zeta_mean + zeta_err)
    return (theta_i, log_f_a, Delta_N_eff, mu, zeta)

def ln_prior(THETA):
    theta_i, log_f_a, Delta_N_eff, mu, zeta = THETA
    if np.all(np.array(THETA[2:]) > 0) and 0 < theta_i <= np.pi and 15 <= log_f_a - 9 <= 18 and \
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
num_threads = 1

if __name__ == "__main__":
    pos = [make_initial_guess() for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    np.savez(config.data_path + "/micro_qcd_parameter.npz", samples=sampler.chain)
