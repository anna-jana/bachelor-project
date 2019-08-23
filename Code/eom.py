"""
This module implements the solver for the axion eom and the computation of the relic density
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

import config
import time_temp

class Model:
    def __init__(self, m_a_fn, g_model, potential_model, parameter=config.parameter):
        self.m_a_fn = m_a_fn
        self.g_model = g_model
        self.potential_model = potential_model
        self.parameter = parameter

    def get_solver(self, theta_i, f_a, **solver_opts):
        return Solver(self, theta_i, f_a, **solver_opts)


class Solver:
    def __init__(self, model, theta_i, f_a, temperature_unit=1e12, from_T_osc=5, avg_start=0.8, avg_stop=0.6, N=300, eps=1e-5, num_crossings=3):
        self.model = model
        self.theta_i = theta_i
        self.f_a = f_a
        self.temperature_unit = temperature_unit
        self.from_T_osc = from_T_osc
        self.avg_start = avg_start
        self.avg_stop = avg_stop
        self.N = N
        self.eps = eps
        self.num_crossings = num_crossings
        self.T_s = None
        self.T_osc = time_temp.find_T_osc(self.f_a, self.model.m_a_fn, self.model.g_model) / temperature_unit
        self.T_start = from_T_osc * self.T_osc
        self.dT = (self.T_osc - self.T_start) / N
        self.solver = inte.ode(self.axion_eom_T_rhs).set_integrator("dopri5", nsteps=10000).set_initial_value((self.theta_i, 0), self.T_start)

    def axion_eom_T_rhs(self, T, y):
        theta, dthetadT = y
        assert np.isfinite(theta)
        H = time_temp.hubble_parameter_in_rad_epoch(T * self.temperature_unit, self.model.g_model)
        dtdT = time_temp.dtdT(T * self.temperature_unit, self.model.g_model, self.model.parameter) * self.temperature_unit
        d2tdT2 = time_temp.d2tdT2(T * self.temperature_unit, self.model.g_model, self.model.parameter) * self.temperature_unit**2
        m_a = self.model.m_a_fn(T * self.temperature_unit, self.f_a)
        d2thetadT2 = - (3 * H * dtdT - d2tdT2 / dtdT) * dthetadT - m_a**2 * dtdT**2 * self.model.potential_model.dVdtheta(theta)
        return [dthetadT, d2thetadT2] # list bc. the solver needs a list for some reason

    def solve_to_osc(self):
        # integrate to oscillation regime (first zero crossing)
        while self.solver.y[0] > 0:
            self.solver.integrate(self.solver.t + self.dT)
        # solver.integrate(avg_start * T_s) # TODO: do I need this?

    def advance(self, amount, steps):
        theta = np.empty(steps)
        T = np.empty(steps)
        dT = amount / steps
        for i in range(steps):
            self.solver.integrate(self.solver.t + dT)
            theta[i] = self.solver.y[0]
            T[i] = self.solver.t
        return T * self.temperature_unit, theta


    def field_to_osc(self):
        theta = []; T = []
        while self.solver.y[0] > 0:
            self.solver.integrate(self.solver.t + self.dT)
            theta.append(self.solver.y[0]); T.append(self.solver.t)
        return np.array(T) * self.temperature_unit, theta

    def find_const_n_over_s(self):
        T_s = self.solver.t
        delta_T = (self.avg_stop - self.avg_start) * T_s
        self.dT = delta_T / self.N
        last_n_over_s = np.NAN

        while True:
            T_values, theta_values, dthetadT_values = [], [], []
            zero_crossings = 0
            prev_sign = np.sign(self.solver.y[0])

            for i in range(self.N):
                self.solver.integrate(self.solver.t + self.dT)
                # count zero crossings
                sign = np.sign(self.solver.y[0])
                if prev_sign != sign:
                    zero_crossings += 1
                prev_sign = sign
                # collect values
                T_values.append(self.solver.t); theta_values.append(self.solver.y[0]); dthetadT_values.append(self.solver.y[1])

            T = np.array(T_values) * self.temperature_unit
            theta = np.array(theta_values)
            dthetadT = np.array(dthetadT_values) / self.temperature_unit

            if self.solver.t * self.temperature_unit < self.model.parameter.T_eq:
                print("warning: integration over T_eq")
                return T, theta, dthetadT, np.NAN

            n_over_s = self.compute_n_over_s(T, theta, dthetadT)
            d_n_over_s_dT = (last_n_over_s - n_over_s) / (self.temperature_unit * delta_T)

            # n/s has to be conserved and we need to integrate several oscillations
            if abs(d_n_over_s_dT) < self.eps and zero_crossings > self.num_crossings:
                return T, theta, dthetadT, n_over_s


            last_n_over_s = n_over_s

    def compute_n_over_s(self, T, theta, dthetadT):
        delta_T = T[-1] - T[0]
        m_a = self.model.m_a_fn(T, self.f_a)
        dtdT = time_temp.dtdT(T, self.model.g_model, self.model.parameter)
        g_s = self.model.g_model.g_s(T)
        n_over_s_at_each_T = 45 / (2 * np.pi**2) * self.f_a**2 / (m_a * g_s * T**3) * \
                (0.5 * (dthetadT / dtdT)**2 + m_a**2 * self.model.potential_model.V(theta))
        return inte.simps(n_over_s_at_each_T, T) / delta_T

    def compute_density_parameter_from_n_over_s(self, n_over_s):
        s_today = 2 * np.pi**2 / 45 * 43 / 11 * self.model.parameter.T0**3
        n_a_today = n_over_s * s_today
        rho_a_today = self.model.m_a_fn(self.model.parameter.T0, self.f_a) * n_a_today
        Omega_a_h_sq_today = self.model.parameter.h**2 * rho_a_today / self.model.parameter.rho_c
        return Omega_a_h_sq_today

    def compute_density_parameter(self):
        self.solve_to_osc()
        T, theta, dthetadT, n_over_s = self.find_const_n_over_s()
        return self.compute_density_parameter_from_n_over_s(n_over_s)
