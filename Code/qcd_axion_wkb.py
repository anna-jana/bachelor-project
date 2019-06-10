import numpy as np
import sympy as sp
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

if __name__ == "__main__":
    show("kappa =", kappa)
    show("iota =", iota)

    # ## Compute $\kappa$ and $\iota$
    E1, E2, n, C, f_c, f_a, Lambda_QCD, kappa, N, rho_c, g_0, g_osc, T_0, gamma, theta_i, eV =  sp.symbols("E1, E2, n, C, f_c, f_a, Lambda_QCD, kappa, N, rho_c, g_0, g_osc, T_0, gamma, theta_i, eV")
    GeV = 10**9 * eV
    T_osc = kappa * (E1 / (f_a / N))**(1/(1 + n/2)/2)
    #m_a = 6*10**(-10) * eV * 10**16 * GeV / (f_a / N)
    m_a = E2 * E1 / (f_a / N)
    zeta = C * (Lambda_QCD / T_osc)**n # ok
    m_a_osc = m_a * zeta # ok
    # s up to a constant factor
    s0 = g_0 * T_0**3 # ok
    s_osc = g_osc * T_osc**3 # ok
    n_a = f_c / 2 * m_a_osc * (f_a / N)**2 * theta_i**2
    rho_a = n_a * m_a * s0 / s_osc * gamma
    Omega_a = rho_a / rho_c # ok
    print("Omega_a =", Omega_a)

    # ## Plot WKB Approximation
    # plot
    N = 300
    f_a = np.logspace(9, 18, N) * 1e9 # [eV]
    theta_i = np.logspace(-5, 0, N) # [RAD]
    Omega_a_h_sq = compute_analytic_relic_density(theta_i, f_a)
    fig = density_plot.plot_density(theta_i, f_a, Omega_a_h_sq, plot_type="contourf")
    plt.savefig(plot_path + "/qcd_relic_denstiy_wkb_plot.pdf")
