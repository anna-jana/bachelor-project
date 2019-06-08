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
        theta, dthetadT = y
        assert np.isfinite(theta)
        H = time_temp.hubble_parameter_in_rad_epoch(T * temperature_unit, self.g_model)
        dtdT = time_temp.dtdT(T * temperature_unit, self.g_model) * temperature_unit
        d2tdT2 = time_temp.d2tdT2(T * temperature_unit, self.g_model) * temperature_unit**2
        m_a = self.m_a_fn(T * temperature_unit, self.f_a)
        d2thetadT2 = - (3 * H * dtdT - d2tdT2 / dtdT) * dthetadT - m_a**2 * dtdT**2 * self.potential_model.dVdtheta(theta)
        return [dthetadT, d2thetadT2] # list bc. the solver needs a list for some reason

    def find_axion_field_osc_vals(self, from_T_osc=5, avg_start=0.8, avg_stop=0.6, N=300, eps=1e-5, num_crossings=3):
        assert self.f_a is not None and self.theta_i is not None
        # set up the ode solver
        T_osc = T_osc_solver.find_T_osc(self.f_a, self.m_a_fn, self.g_model) / temperature_unit
        T_start = from_T_osc * T_osc
        dT = (T_osc - T_start) / N
        solver = inte.ode(self.axion_eom_T_rhs).set_integrator("dopri5", nsteps=10000).set_initial_value((self.theta_i, 0), T_start)

        # integrate to oscillation regime (first zero crossing)
        while solver.y[0] > 0:
            solver.integrate(solver.t + dT)
        T_s = solver.t
        # solver.integrate(avg_start * T_s) # TODO: do I need this?

        delta_T = (avg_stop - avg_start) * T_s
        dT = delta_T / N
        last_n_over_s = np.NAN

        while True:
            T_values, theta_values, dthetadT_values = [], [], []
            zero_crossings = 0
            prev_sign = np.sign(solver.y[0])

            for i in range(N):
                solver.integrate(solver.t + dT)
                # count zero crossings
                sign = np.sign(solver.y[0])
                if prev_sign != sign:
                    zero_crossings += 1
                prev_sign = sign
                # collect values
                T_values.append(solver.t); theta_values.append(solver.y[0]); dthetadT_values.append(solver.y[1])

            T = np.array(T_values) * temperature_unit
            theta = np.array(theta_values)
            dthetadT = np.array(dthetadT_values) / temperature_unit

            if solver.t * temperature_unit < self.parameter.T_eq:
                print("warning: integration over T_eq")
                return T, theta, dthetadT, np.NAN

            n_over_s = self.compute_n_over_s(T, theta, dthetadT)
            d_n_over_s_dT = abs(last_n_over_s - n_over_s) / delta_T

            # n/s has to be conserved and we need to integrate several oscillations
            if d_n_over_s_dT < eps and zero_crossings > num_crossings:
                return T, theta, dthetadT, n_over_s

            last_n_over_s = n_over_s

    def compute_n_over_s(self, T, theta, dthetadT):
        delta_T = T[-1] - T[0]
        m_a = self.m_a_fn(T, self.f_a)
        dtdT = time_temp.dtdT(T, self.g_model)
        g_s = self.g_model.g_s(T)
        n_over_s_at_each_T = 45 / (2 * np.pi**2) * self.f_a**2 / (m_a * g_s * T**3) * \
                (0.5 * (dthetadT / dtdT)**2 + m_a**2 * self.potential_model.V(theta))
        return inte.simps(n_over_s_at_each_T, T) / delta_T

    def compute_density_parameter_from_n_over_s(self, n_over_s):
        s_today = 2 * np.pi**2 / 45 * 43 / 11 * self.parameter.T0**3
        n_a_today = n_over_s * s_today
        rho_a_today = self.m_a_fn(self.parameter.T0, self.f_a) * n_a_today
        Omega_a_h_sq_today = self.parameter.h**2 * rho_a_today / self.parameter.rho_c
        return Omega_a_h_sq_today

    def compute_density_parameter(self, **kwargs):
        return self.compute_density_parameter_from_n_over_s(self.find_axion_field_osc_vals(**kwargs)[-1])


