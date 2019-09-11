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

def compute_Omega(theta_i, f_a, T_ratio, mu, zeta, debug_print=True):
    if debug_print:
        print("*", end="")
        sys.stdout.flush()
    g = g_star.make_micro_from_T_ratio(T_ratio, mu)
    T_i = time_temp.find_T_osc(f_a, lambda T, f: axion_mass.micro_m_a(T, f, mu, zeta), g)
    if T_i <= config.parameter.T_eq:
        print("WARNING: oscillation starts below T_eq")
        return np.nan
    T_f = config.parameter.T0
    m_i = axion_mass.micro_m_a(g_star.compute_T_ratio(T_i, T_ratio, mu) * T_i, f_a, mu, zeta)
    m_f = axion_mass.micro_m_a(g_star.compute_T_ratio(T_f, T_ratio, mu) * T_f, f_a, mu, zeta)
    g_s_f = g.g_s(T_f)
    g_s_i = g.g_s(T_i)
    A_i = theta_i * f_a
    rho = 0.5 * m_i * m_f * A_i**2 * g_s_f / g_s_i * (T_f / T_i)**3
    Omega = rho / config.parameter.rho_c * config.parameter.h**2
    return Omega

# N_neutrino = 3.045 # from http://pdg.lbl.gov/2019/reviews/rpp2018-rev-neutrinos-in-cosmology.pdf, page 2
# Delta_N_eff_BBM_mean = 2.91 - N_neutrino # https://arxiv.org/pdf/1502.01589.pdf, page 49, (75)
# Delta_N_eff_BBM_err = 0.37
# Delta_N_eff_CMB_mean = 3.04 - N_neutrino # https://arxiv.org/pdf/1502.01589.pdf, page 43, (60d)
# Delta_N_eff_CMB_err = 0.18
# T_BBM = 1e6
# T_CMB = config.parameter.T0


N_neutrino = 3.045 # from http://pdg.lbl.gov/2019/reviews/rpp2018-rev-neutrinos-in-cosmology.pdf, page 2
# He + TT + lowP + BAO
# Delta_N_eff_BBM_mean = 3.14 - N_neutrino # https://arxiv.org/pdf/1502.01589.pdf, page 49, (75)
# Delta_N_eff_BBM_err = 0.44
Delta_N_eff_BBM_mean = 3.01 - N_neutrino # https://arxiv.org/pdf/1502.01589.pdf, page 49, (75)
Delta_N_eff_BBM_err = 0.38
# TT + lowP + BAO <- most robust result?
Delta_N_eff_CMB_mean = 3.15 - N_neutrino # https://arxiv.org/pdf/1502.01589.pdf, page 43, (60d)
Delta_N_eff_CMB_err = 0.23
T_BBM = 1e6
T_CMB = config.parameter.T0

def ln_likelihood(THETA):
    # theta_i, log_f_a, T_ratio, mu, zeta = THETA
    theta_i, log_f_a, T_ratio, log_mu, zeta = THETA
    mu = 10**log_mu
    f_a = 10**log_f_a
    density_parameter_computed = compute_Omega(theta_i, f_a, T_ratio, mu, zeta)
    Delta_N_eff_at_BBM = g_star.compute_Delta_N_eff(T_ratio, mu, T_BBM)
    Delta_N_eff_at_CMB = g_star.compute_Delta_N_eff(T_ratio, mu, T_CMB)
    # if Delta_N_eff_at_CMB > Delta_N_eff_CMB_mean + Delta_N_eff_CMB_err:
    #     return - np.inf
    # if Delta_N_eff_at_BBM > Delta_N_eff_BBM_mean + Delta_N_eff_BBM_err:
    #     return - np.inf
    return (
        log_gaussian(density_parameter_computed, config.parameter.Omega_DM_h_sq, config.parameter.Omega_DM_h_sq_err)
        + log_gaussian(Delta_N_eff_at_BBM, Delta_N_eff_BBM_mean, Delta_N_eff_BBM_err)
        + log_gaussian(Delta_N_eff_at_CMB, Delta_N_eff_CMB_mean, Delta_N_eff_CMB_err)
    )

parameter_names = ["theta_i", "log f_a", "T_ratio", "mu", "zeta"]

zeta_mean = 1.34
zeta_err = 0.01

log_f_a_min, log_f_a_max = 15 + 9, 17 + 9
log_mu_min, log_mu_max = 1, 3
T_ratio_min, T_ratio_max = 1 / 6, 1 / 3

def make_initial_guess():
    theta_i = np.random.uniform(0, np.pi)
    log_f_a = np.random.uniform(log_f_a_min, log_f_a_max)
    # mu = 10**np.random.uniform(log_mu_min, log_mu_max)
    log_mu = np.random.uniform(log_mu_min, log_mu_max)
    T_ratio = np.random.uniform(T_ratio_min, T_ratio_max)
    zeta = np.random.uniform(zeta_mean - zeta_err, zeta_mean + zeta_err)
    return (theta_i, log_f_a, T_ratio, log_mu, zeta)
    # return (theta_i, log_f_a, T_ratio, mu, zeta)

def ln_prior(THETA):
    # theta_i, log_f_a, T_ratio, mu, zeta = THETA
    theta_i, log_f_a, T_ratio, log_mu, zeta = THETA
    # log_mu_min <= np.log10(mu) <= log_mu_max and abs(zeta - zeta_mean) <= zeta_err \
    if 0 < theta_i <= np.pi and log_f_a_min <= log_f_a <= log_f_a_max and \
       log_mu_min <= log_mu <= log_mu_max and abs(zeta - zeta_mean) <= zeta_err \
       and T_ratio_min <= T_ratio <= T_ratio_max:
        return 0
    else:
        return - np.inf # = log 0

def ln_prob(THETA):
    lp = ln_prior(THETA)
    like = ln_likelihood(THETA)
    if not np.isfinite(lp) or not np.isfinite(like):
        return -np.inf
    return lp + like

ndim = 2 + 3
# num_walkers = ndim * 2 * 5 * 2 * 2
# steps = 5000 * 4 * 4

num_walkers = ndim * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2
steps = 5000 * 2 * 2
num_threads = 4

if __name__ == "__main__":
    pos = [make_initial_guess() for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    np.savez(config.data_path + "/T_ratio_micro_qcd_parameter.npz", samples=sampler.chain)
