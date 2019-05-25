"""
This module implements the solver for the axion eom and the computation of the relic density
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

from config import model
import time_temp
import T_osc_solver

temperature_unit = 1e12 # eV

def sim_axion_field_evo_T(theta_i, f_a, m_a_fn, g_model, from_T_osc=5, to_T_osc=0.2, num_pts_to_return=400): # for testing
    T_osc = T_osc_solver.find_T_osc(f_a, m_a_fn, g_model)
    T_range = (from_T_osc * T_osc / temperature_unit, to_T_osc * T_osc / temperature_unit)
    T = np.linspace(*T_range, num_pts_to_return)
    sol = inte.solve_ivp(lambda T, y: axion_eom_T_rhs(T, y, f_a, m_a_fn, g_model), T_range, (theta_i, 0), t_eval=T)
    return T * temperature_unit, sol.y[0, :]

def axion_eom_T_rhs(T, y, f_a, m_a_fn, g_model):
    """
    Computes the rhs of the eom of the axion field as a function of the temperature,
    given the temperature in eV, y = [theta(T), dtheta/dT(T)], g_model is a GStarModel,
    m_a_fn : T x f_a -> m_a [eV]
    """
    theta, dthetadT = y
    assert np.isfinite(theta)
    H = time_temp.hubble_parameter_in_rad_epoch(T * temperature_unit, g_model)
    dtdT = time_temp.dtdT(T * temperature_unit, g_model) * temperature_unit
    d2tdT2 = time_temp.d2tdT2(T * temperature_unit, g_model) * temperature_unit**2
    m_a = m_a_fn(T * temperature_unit, f_a)
    d2thetadT2 = - (3 * H * dtdT - d2tdT2 / dtdT) * dthetadT - m_a**2 * dtdT**2 * np.sin(theta)
    return [dthetadT, d2thetadT2]

def find_axion_field_osc_vals(theta_i, f_a, m_a_fn, g_model, from_T_osc=5, avg_start=0.8, avg_stop=0.6, N=300):
    """
    Solve the EOM for the axion field the oscillation and follow then num_zero_crossing_to_do zero crossings
    Then average the energy density over num_osc_to_avg
    Takes the initial field value theta_i, the axion decay constant f_a,
    the axion mass as a function of temperature m_a_fn : f_a x T -> m_a and the model
    the the eff. rel. dof. : GStarModel.
    """
    # set up the solver
    T_osc = T_osc_solver.find_T_osc(f_a, m_a_fn, g_model) / temperature_unit
    T_start = from_T_osc * T_osc
    dT = (T_osc - T_start) / N
    solver = inte.ode(axion_eom_T_rhs).set_integrator("dopri5").set_f_params(f_a, m_a_fn, g_model).set_initial_value((theta_i, 0), T_start)

    # integrate to oscillation regime (first zero crossing)
    while solver.y[0] > 0:
        solver.integrate(solver.t + dT)
    T_s = solver.t
    solver.integrate(avg_start * T_s)

    # collect values
    dT = (avg_stop - avg_start) * T_s / N
    T_values, theta_values, dthetadT_values = [], [], []
    for i in range(N):
        solver.integrate(solver.t + dT)
        T_values.append(solver.t); theta_values.append(solver.y[0]); dthetadT_values.append(solver.y[1])

    return np.array(T_values) * temperature_unit, np.array(theta_values), np.array(dthetadT_values) / temperature_unit

def compute_density_parameter_from_field(T, theta, dthetadT, f_a, m_a_fn, g_model):
    """
    Compute the density parameter of the axions from the simulated field.
    Takes T in eV, theta, dthetadT per 1/eV, f_a in eV, m_a_fn : T x f_a -> m_a, g_model : GStarModel
    returns the density parameter for the axions Omega_a_h_sq_today * h**2
    """
    # compute averaged n/s
    delta_T = T[-1] - T[0]
    m_a = m_a_fn(T, f_a)
    dtdT = time_temp.dtdT(T, g_model)
    g_s = g_model.g_s(T)
    n_over_s_at_each_T = 45 / (2 * np.pi**2) * f_a**2 / (m_a * g_s * T**3) * (0.5 * (dthetadT / dtdT)**2 + m_a * (1 - np.cos(theta)))
    n_over_s = inte.simps(n_over_s_at_each_T, T) / delta_T
    # scale to today
    s_today = 2 * np.pi**2 / 45 * 43 / 11 * model.T0**3
    n_a_today = n_over_s * s_today
    rho_a_today = m_a_fn(model.T0, f_a) * n_a_today
    # compute density parameter
    Omega_a_h_sq_today = model.h**2 * rho_a_today / model.rho_c
    return Omega_a_h_sq_today

def compute_density_parameter(theta_i, f_a, m_a_fn, g_model):
    T, theta, dthetadT = find_axion_field_osc_vals(theta_i, f_a, m_a_fn, g_model)
    return compute_density_parameter_from_field(T, theta, dthetadT, f_a, m_a_fn, g_model)

