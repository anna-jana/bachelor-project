"""
Plot the analytical solution for the Klein Gordon Eq.
in the case of an universe with a ~ t^p like radiation domination
and for V = 1/2*m_a**2*phi**2 with a constant mass.
"""

# coding: utf-8
from __future__ import division, print_function

import sys, os, time, math
import numpy as np
from numpy import pi, sin, cos, tan, tanh, sqrt, exp, log
from scipy.integrate import odeint
from scipy.constants import G, c, hbar, Boltzmann as kB
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jvp, yvp # first and second bessel functions

# parameter
N = 300 # number of steps
t_init = 1.0 # inital time
# t_end = 10.0 # end of the simulation
m_a = 1e-20 # axion mass [eV]
f_a = 1e10 # axion decay constant
# theta_init = 1.0 # initial theta value after symmetry breaking
phi_init = 10**9 * 10**16 # [eV]
# phi_init == theta_init * f_a
theta_init = phi_init / f_a
p = 1/2 # radiation dominated
n = (3*p - 1) / 2 # nth bessel function
# scale parameter
a0 = 1.0
a_end = 1e3
t_end = t_init * (a_end / a0)**(1/p)
t = np.linspace(t_init, t_end, N) # time
a = a0*(t/t_init)**p
a_prime = a / a0 # sclae factor relative to the initial scale factor
a_dot = a0*p*(t/t_init)**(p - 1)
H = a_dot / a
bessel_arg_init = m_a * t_init

# initial conditions
# from psi dot = 0
alpha = (-3/2*p + 1/2) / t_init
beta = t_init**(-3/2*p + 1/2)
A = alpha*jv(n, bessel_arg_init) + beta*jvp(n, bessel_arg_init)
B = alpha*yv(n, bessel_arg_init) + beta*yvp(n, bessel_arg_init)

# from psi = f_a * theta
gamma = a0**(-3/2)*(t / t_init)**(-3/2*p + 1/2)
C = gamma * jv(n, bessel_arg_init)
D = gamma * yv(n, bessel_arg_init)

# compute coeffs
det = B*C - A*D
C1 = B*f_a*theta_init / det
C2 = A*f_a*theta_init / det

# compute analytic solution for m_a = const and V = 1/2*m_a**2*phi**2 as well as a = a0 t**p
# for the axion background field and its time derivative
phi = a**(-3/2) * (t / t_init)**(1/2) * (C1*jv(n, m_a * t) + C2*yv(n, m_a * t))
kappa = a0**(-3/2) / t_init**(-3/2*p + 1/2)
phi_dot = kappa * ((-3/2*p + 1/2)*t**(-3/2*p - 1/2)*(C1*jv(n, m_a * t) + C2*yv(n, m_a * t)) + \
                   t**(-3/2*p + 1/2) * m_a * (C1*jvp(n, m_a * t) + C2*yvp(n, m_a * t)))

# compute density and related quantities
rho_a = 1/2*phi_dot**2 + 1/2*m_a**2*phi**2 # denity
P_a = 1/2*phi_dot**2 - 1/2*m_a**2*phi**2 # pressure
w_a = P_a / rho_a # eq. of state
rho_c = 3 * H**2 / (8*pi*G) # critical density
density_parameter = rho_a / rho_c

# plot all the stuff
# field
plt.subplot(2, 2, 1)
plt.semilogx(a_prime, phi)
plt.ylabel(r"Axion Field $\phi$")
# TODO: plot the a_osc position using a vertical dashed line

# mass vs hubble
plt.subplot(2, 2, 2)
plt.loglog(a_prime, H, label="Hubble")
plt.loglog(a_prime, np.ones(a_prime.size) * m_a / 2, label=r"$m_a / 2$")
plt.legend()

# eos
plt.subplot(2, 2, 3)
plt.semilogx(a_prime, w_a)
plt.ylabel("Equation of State $w$")
plt.xlabel("Scale Factor $a / a_i$")

# density
plt.subplot(2, 2, 4)
plt.loglog(a_prime, rho_a, label="Exact Density")
plt.legend()

plt.show()
