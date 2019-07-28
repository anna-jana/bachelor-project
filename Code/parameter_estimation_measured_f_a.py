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

def log_gaussian(x, x_mean, x_stdev):
    return - (x - x_mean)**2 / (2 * x_stdev**2) - 0.5 * np.log(2 * np.pi * x_stdev**2)

def ln_likelihood(THETA):
    # print("*", end=""); sys.stdout.flush()
    # print(THETA)
    theta_i, f_a, p.M_pl, p.Lambda_QCD, p.m_u, p.m_d, p.m_pi0, p.f_pi0, p.T0, p.rho_c = THETA
    # model = eom.Model(axion_mass.m_a_from_chi_general, g_star.matched, potential.cosine, p)
    # solver = model.get_solver(theta_i, 10**log_f_a)
    # density_parameter_computed = solver.compute_density_parameter()
    density_parameter_computed = solver.compute_relic_density(p, theta_i, f_a)
    return log_gaussian(density_parameter_computed, parameter.Omega_DM_h_sq, parameter.Omega_DM_h_sq_err)

measured_f_a = 1e12 * 1e9
Q = 1e6
experimental_f_a_error = measured_f_a / Q

parameter_names = ["theta_i", "f_a", "M_pl", "Lambda_QCD", "m_u", "m_d", "m_pi", "f_pi", "T0", "rho_c"]
errors = [0.0, experimental_f_a_error, parameter.M_pl_err, parameter.Lambda_QCD_err, parameter.m_u_err, parameter.m_d_err,
          parameter.m_pi0_err, parameter.f_pi0_err, parameter.T0_err, parameter.rho_c_err]

inital_guess = np.array((
    1, measured_f_a, parameter.M_pl, parameter.Lambda_QCD,
    parameter.m_u, parameter.m_d, parameter.m_pi0,
    parameter.f_pi0, parameter.T0, parameter.rho_c
))

def make_initial_guess():
    ans = inital_guess.copy()
    ans[0] = np.random.uniform(0, np.pi) # theta_i
    ans[1:] = np.random.normal(loc=ans[1:], scale=errors[1:])
    return ans

def ln_prior(THETA):
    theta_i, f_a, M_pl, Lambda_QCD, m_u, m_d, m_pi0, f_pi0, T0, rho_c = THETA
    if np.all(np.array(THETA[2:]) > 0) and 0 < theta_i <= np.pi:
        return (
            log_gaussian(M_pl, parameter.M_pl, parameter.M_pl_err) +
            log_gaussian(Lambda_QCD, parameter.Lambda_QCD, parameter.Lambda_QCD_err) +
            log_gaussian(m_u, parameter.m_u, parameter.m_u_err) +
            log_gaussian(m_d, parameter.m_d, parameter.m_d_err) +
            log_gaussian(m_pi0, parameter.m_pi0, parameter.m_pi0_err) +
            log_gaussian(f_pi0, parameter.f_pi0, parameter.f_pi0_err) +
            log_gaussian(T0, parameter.T0, parameter.T0_err) +
            log_gaussian(rho_c, parameter.rho_c, parameter.rho_c_err) +
            log_gaussian(f_a, measured_f_a, experimental_f_a_error)
        )
    else:
        return - np.inf # = log 0

def ln_prob(THETA):
    lp = ln_prior(THETA)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(THETA)

num_walkers = len(inital_guess) * 2 * 5 * 2
steps = 5000 * 2
num_threads = 4
ndim = len(inital_guess) # 2 + 2 + 3 + 3

if __name__ == "__main__":
    pos = [make_initial_guess() for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    # chain: (num_walker, steps, n_dim = #parameters)
    # samples = sampler.chain[:, eq_steps:, :].reshape((-1, ndim))
    # save the data
    filename = config.data_path + "/parameter_f_a_measured.npz"
    np.savez(filename, samples=sampler.chain)
