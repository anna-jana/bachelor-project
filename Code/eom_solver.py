"""
This module implements the solver for the axion eom and the computation of the relic density
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

import config
import time_temp
import T_osc_solver

temperature_unit = 1e12 # eV

class EOMSolver:
    def __init__(self, m_a_fn, g_model, potential_model, parameter=None):
        self.theta_i = None
        self.f_a = None
        self.m_a_fn = m_a_fn
        self.g_model = g_model
        self.potential_model = potential_model
        if parameter is None:
            self.parameter = config.parameter
        else:
            self.parameter = parameter

    def axion_eom_T_rhs(self, T, y):
        """
        Computes the rhs of the eom of the axion field as a function of the temperature,
        given the temperature in eV, y = [theta(T), dtheta/dT(T)], g_model is a GStarModel,
        m_a_fn : T x f_a -> m_a [eV]
        """
        theta, dthetadT = y
        assert np.isfinite(theta)
        H = time_temp.hubble_parameter_in_rad_epoch(T * temperature_unit, self.g_model)
        dtdT = time_temp.dtdT(T * temperature_unit, self.g_model) * temperature_unit
        d2tdT2 = time_temp.d2tdT2(T * temperature_unit, self.g_model) * temperature_unit**2
        m_a = self.m_a_fn(T * temperature_unit, self.f_a)
        d2thetadT2 = - (3 * H * dtdT - d2tdT2 / dtdT) * dthetadT - m_a**2 * dtdT**2 * self.potential_model.dVdtheta(theta)
        return [dthetadT, d2thetadT2]

    def find_axion_field_osc_vals(self, from_T_osc=5, avg_start=0.8, avg_stop=0.6, N=300):
        """
        Solve the EOM for the axion field the oscillation and follow then num_zero_crossing_to_do zero crossings
        Then average the energy density over num_osc_to_avg
        Takes the initial field value theta_i, the axion decay constant f_a,
        the axion mass as a function of temperature m_a_fn : f_a x T -> m_a and the model
        the the eff. rel. dof. : GStarModel.
        """
        assert self.f_a is not None and self.theta_i is not None
        # set up the ode solver
        T_osc = T_osc_solver.find_T_osc(self.f_a, self.m_a_fn, self.g_model) / temperature_unit
        T_start = from_T_osc * T_osc
        dT = (T_osc - T_start) / N
        solver = inte.ode(self.axion_eom_T_rhs).set_integrator("dopri5").set_initial_value((self.theta_i, 0), T_start)

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

    def compute_density_parameter_from_field(self, T, theta, dthetadT):
        """
        Compute the density parameter of the axions from the simulated field.
        Takes T in eV, theta, dthetadT per 1/eV, f_a in eV, m_a_fn : T x f_a -> m_a, g_model : GStarModel
        returns the density parameter for the axions Omega_a_h_sq_today * h**2
        """
        # compute averaged n/s
        delta_T = T[-1] - T[0]
        m_a = self.m_a_fn(T, self.f_a)
        dtdT = time_temp.dtdT(T, self.g_model)
        g_s = self.g_model.g_s(T)
        n_over_s_at_each_T = 45 / (2 * np.pi**2) * self.f_a**2 / (m_a * g_s * T**3) * \
                (0.5 * (dthetadT / dtdT)**2 + m_a**2 * self.potential_model.V(theta))
        n_over_s = inte.simps(n_over_s_at_each_T, T) / delta_T
        # scale to today
        s_today = 2 * np.pi**2 / 45 * 43 / 11 * self.parameter.T0**3
        n_a_today = n_over_s * s_today
        rho_a_today = self.m_a_fn(self.parameter.T0, self.f_a) * n_a_today
        # compute density parameter
        Omega_a_h_sq_today = self.parameter.h**2 * rho_a_today / self.parameter.rho_c
        return Omega_a_h_sq_today

