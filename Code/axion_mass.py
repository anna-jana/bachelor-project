r"""
This module provides different ways of calculating the axion mass at a given temperature.
On the one hand there are analytic approximations given as python function encoding the formulas
and on the other hand there are are numerical functions for $\chi_mathrm{top} = m_a(T)^2 f_a^2$
computed with lattice QCD from some paper.
References are (hopfully!) given.
All functions follow the signature
m_a_from_somthing(T_in_eV, f_a_in_eV, *args, **kwargs) -> mass in eV
"""

import numpy as np
import scipy.constants as c
from scipy.interpolate import PchipInterpolator

from config import model
import config

######################## numerical function from Borsamyi et al. ########################
# data points for $chi_\mathrm{top}$ from Borsamyi et al. S10.3
# T[MeV], - log10( \chi / fm^-4 )
data = np.array([
    [100, -1.66],
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
    [3000, -12.05],
])

# load data extracted from plot
file_data = np.loadtxt(config.data_path + "/chi_data.dat")

def convert_chi_data(T_MeV, chi_fm):
    sort_perm = np.argsort(T_MeV)
    T = 1e6 * T_MeV[sort_perm]
    chi = (1 / c.elementary_charge * c.hbar * c.c / 1e-15)**4 * chi_fm[sort_perm]
    return T, chi

T_paper, chi_paper = convert_chi_data(data[:, 0], 10**data[:, 1])
T_plot, chi_plot = convert_chi_data(file_data[:, 0], file_data[:, 1])
T_data, chi_data = convert_chi_data(np.concatenate([data[:, 0], file_data[:, 0]]), np.concatenate([10**data[:, 1], file_data[:, 1]]))

chi_interp = PchipInterpolator(T_data, chi_data)
chi_interp_paper = PchipInterpolator(T_paper, chi_paper)
chi_interp_plot = PchipInterpolator(T_plot, chi_plot)

def m_a_from_chi(T, f_a, chi_interp_to_use=None):
    """
    Compute axion mass from chi data from Borsamyi et als lattice qcd simulations
    """
    if chi_interp_to_use is None:
       chi = chi_interp_paper
    return np.sqrt(chi(T)) / f_a

def m_a_from_chi_general(T, f_a, chi_interp_to_use=None):
    """
    Compute m_a fro chi data from Borsamyi's lattice qcd simulations but use m_a(T = 0) below its domain and
    m_a = m_a_shellard for T above its domain.
    """
    if chi_interp_to_use is None:
       chi = chi_interp_paper
    return np.where(T < T_paper[0], m_a_at_abs_zero_from_shellard(f_a),
            np.where(T > T_paper[-1], m_a_at_high_T_from_fox(T, f_a, True), m_a_from_chi(T, f_a)))

################################## m_a at T = 0 ####################################
def m_a_at_abs_zero_from_marsh(f_a):
    """
    T = 0 axion mass from review by marsh
    """
    return 6e-10 * 1e16 * 1e9 / f_a

###### shellard (weinberg) ######
def m_a_at_abs_zero_from_shellard(f_a):
    """
    Classic T = 0 axion mass form shellard (original from weinberg)
    """
    return model.m_pi0 * model.f_pi0 * np.sqrt(model.m_u * model.m_d) / (model.m_u + model.m_d) / f_a

######################################## shellard ##########################################
Lambda_shellard = 400e6

###### m_a at low T < Lambda_QCD
def m_a_at_low_T_from_shellard(T, f_a):
    """
    Low temperature fit from IILA by shellard to compute axion mass
    """
    return np.sqrt(1.46e-3 * Lambda_shellard**4 * (1 + 0.5*T/Lambda_shellard) / (1 + (3.53*T/Lambda_shellard)**7.48)) / f_a

###### DGA result fitted to IILA
n_shellard = 6.68
alpha_a = 1.68e-7

def m_a_at_high_T_from_shellard(T, f_a):
    """
    Compute axion mass at high temperatures using the result from shellard.
    """
    return np.sqrt(alpha_a) * Lambda_shellard**2 / (f_a * (T / Lambda_shellard)**(n_shellard/2))

####### full IILA result
T3 = 0.45e9
T4 = 1.2e9
T5 = 4.2e9
T6 = 100e9

max_deg = np.array([3, 2, 2], dtype=np.int)

d_coeff = np.array([[-15.6, -6.68, -0.947, +0.555],
                    [-15.4, -7.04, -0.139, np.NAN], # data in paper is wrong first coeff in this row + -> -
                    [-14.8, -7.47, -0.0757, np.NAN]])

def m_a_full_IILA_shellard(T, f_a):
    """
    Compute the axion mass using the result from the IILA calculation by shellard.
    This function is only defined in a range between T3 and T6 (global constants)
    """
    log_T_over_Lambda = np.log(T / Lambda_shellard)
    def calc_exponent(N_f):
        return sum(d_coeff[N_f - 3, n]*log_T_over_Lambda**n for n in range(0, max_deg[N_f - 3] + 1))
    exponent = np.where(T < T3,
            np.NAN,
            np.where(T < T4,
                calc_exponent(3),
                np.where(T < T5,
                    calc_exponent(4),
                    np.where(T < T6,
                        calc_exponent(5),
                        np.NAN))))
    return np.sqrt(Lambda_shellard**4 * np.exp(exponent)) / f_a

def m_a_shellard(T, f_a):
    """
    Compute the axion mass using the results of m_a_full_IILA_shellard inside its domain and outside using m_a_at_low_T_from_shellard
    """
    return np.where((T > T3) & (T < T6), m_a_full_IILA_shellard(T, f_a), m_a_at_low_T_from_shellard(T, f_a))

############################  fox et al  #########################
######## m_a at high T > Lambda_QCD ########
d = 1.2
C = 0.018
n_fox = 4
alpha = 0.5

def __m_a_at_high_T_from_fox(T, f_a, my_callibration_factor):
    """
    Internal function to compute the axion mass using the n = 4 result (fox) but with a given callibration_factor
    to normalize to T = 0 mass
    """
    return my_callibration_factor * m_a_at_abs_zero_from_shellard(f_a) * C * (model.Lambda_QCD / 200e6)**alpha * (model.Lambda_QCD / T)**n_fox

Lambda_callibration = 100e6
callibration_factor = m_a_from_chi(Lambda_callibration, 1.0) / __m_a_at_high_T_from_fox(Lambda_callibration, 1.0, 1.0)

def m_a_at_high_T_from_fox(T, f_a, with_correction):
    """
    Compute axion mass at high temperatures using the result from fox. The result is normalized to the T = 0 result from the lattice
    """
    m_a = __m_a_at_high_T_from_fox(T, f_a, callibration_factor)
    if with_correction:
        correction_factor = (1 - np.log(model.Lambda_QCD / T))**d
        m_a *= correction_factor
    return m_a

def m_a_fox(T, f_a):
    """
    Compute m_a using the results from fox. It uses the correction factor if T > Lambda_QCD and m_a = m_a(T = 0) below T = 100MeV
    """
    return np.where(T > model.Lambda_QCD, m_a_at_high_T_from_fox(T, f_a, True),
            np.where(T < 100e6, m_a_at_abs_zero_from_shellard(f_a), m_a_at_high_T_from_fox(T, f_a, False)))

