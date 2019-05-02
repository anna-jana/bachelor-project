"""
This module provides different ways of calculating the axion mass at a given temperature.
On the one hand there are analytic approximations given as python function encoding the formulas
and on the other hand there are are numerical functions for $\chi_mathrm{top} = m_a(T)^2 f_a^2$
computed with lattice QCD from some paper.
References are (hopfully!) given.
All functions follow the signature
m_a_from_somthing(T_in_eV, f_a_in_eV) -> mass in eV
"""

import numpy as np
import scipy.constants as c
from scipy.interpolate import PchipInterpolator

import config
model = config.Model()

################################## m_a at T = 0 ####################################
###### marsh review ######
def m_a_at_abs_zero_from_marsh(f_a):
    return 6e-10 * 1e16 * 1e9 / f_a

###### shellard ######
# correct
def m_a_at_abs_zero_from_shellard(f_a):
    prefactor = model.m_pi0 * model.f_pi0 * np.sqrt(model.m_u * model.m_d) / (model.m_u + model.m_d)
    # print(prefactor)
    return prefactor / f_a

############################## m_a at low T < Lambda_QCD ##########################
###### shellard #######
Lambda_shellard = 400e6

# checked
def m_a_at_low_T_from_shellard(T, f_a):
    m_a = np.sqrt(
            1.46e-3 * Lambda_shellard**4 *
            (1 + 0.5*T/Lambda_shellard) /
            (1 + (3.53*T/Lambda_shellard)**7.48)
        ) / f_a
    return m_a

############################ m_a at high T > Lambda_QCD #########################
######## fox et al. ########
def m_a_at_high_T_from_fox(T, f_a, with_correction):
    Lambda = model.Lambda_QCD
    # Lambda = Lambda_shellard
    C = 0.018
    n_fox = 4
    d = 1.2
    alpha = 0.5
    # m_a = np.sqrt(2) * C * m_a_at_abs_zero_from_marsh(f_a) * (Lambda / T)**n_fox
    m_a = m_a_at_abs_zero_from_shellard(f_a) * C * (Lambda / 200e6)**alpha * (Lambda / T)**n_fox
    if with_correction:
        correction_factor = (1 - np.log(Lambda / T))**d
        m_a *= correction_factor
    return m_a

######## shellard et al. ##########
# approximation to full DGA result
n_shellard = 6.68
alpha_a = 1.68e-7

def m_a_at_high_T_from_shellard(T, f_a):
    m_a = np.sqrt(alpha_a) * Lambda_shellard**2 / (f_a * (T / Lambda_shellard)**(n_shellard/2))
    return m_a

######################## numerical function from Borsamyi et al. ########################
# data points for $chi_\mathrm{top}$ from Borsamyi et al. S10.3
# T[MeV], - log10( \chi / fm^-4 )
data = np.array([
    [100, -1.66 ],
    [120, -1.65],
    [140, -1.75],
    [170, -2.18],
    [200, -2.72],
    [240, -3.39],
    [290, -4.11],
    [350, -4.74],
    [420, -5.34],
    [500, -5.90],
    [600, -6.49],
    [720, -7.08],
    [860, -7.67],
    [1000, -8.17],
    [1200, -8.79],
    [1500, -9.56],
    [1800, -10.20],
    [2100, -10.75],
    [2500, -11.38],
    [3000, -12.05 ],
])
# load data extracted from plot
file_data = np.loadtxt("chi_data.dat")

def convert_chi_data(T_MeV, chi_fm):
    sort_perm = np.argsort(T_MeV)
    T = 1e6 * T_MeV[sort_perm]
    chi = (1 / c.elementary_charge * c.hbar * c.c / 1e-15)**4 * chi_fm[sort_perm]
    return T, chi

T_paper, chi_paper = convert_chi_data(data[:, 0], 10**data[:, 1])
T_plot, chi_plot = convert_chi_data(file_data[:, 0], file_data[:, 1])
T, chi = convert_chi_data(np.concatenate([data[:, 0], file_data[:, 0]]), np.concatenate([10**data[:, 1], file_data[:, 1]]))

chi_interp = PchipInterpolator(T, chi)
chi_interp_paper = PchipInterpolator(T_paper, chi_paper)
chi_interp_plot = PchipInterpolator(T_plot, chi_plot)

def m_a_from_chi(T, f_a):
    return np.sqrt(chi_interp_paper(T)) / f_a

