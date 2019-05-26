import numpy as np
import matplotlib.pyplot as plt

from config import plot_path, model
from util import show
import density_plot

n = 4
C = 0.018
g_star = 61.75
kappa = ((10 * model.M_pl**2 / (np.pi**2 * g_star))**(1/4) * C**(1/2) * model.Lambda_QCD**(n/2) * (6e-10)**(1/2))**(1/(1 + n/2))
g_star_s0 = 3.91
g_star_s_osc = g_star
f_c = 1.44
E1 = 1e16 * 1e9
E2 = 6e-10
iota = (
    C * E1**2 * E2**2 * model.T0**3 * f_c * g_star_s0 * model.Lambda_QCD**n /
    (2 * g_star_s_osc * model.rho_c * kappa**(3 + n)) *
    model.h**2
)

def compute_analytic_relic_density(theta_i, f_a):
    F_A, THETA_I = np.meshgrid(f_a, theta_i)
    Omega_a_h_sq = iota * (F_A / (1e16 * 1e9))**(7/6) * THETA_I**2
    return Omega_a_h_sq
