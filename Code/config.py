import configparser
import sys
import numpy as np
import scipy.constants as c
import astropy.units as u
import astropy.constants as ac

class Parameter:
    def __init__(self):
        # general constants
        m_pl = 1.22093e19
        self.M_pl = m_pl * 1e9 / np.sqrt(8*np.pi)
        self.G = 6.70861e-39 # not used
        self.Lambda_QCD = 200e6

        # cosmological parameters
        self.h = 0.673
        self.z_eq = 3360
        self.T0 = 2.7255 * c.Boltzmann  / c.elementary_charge
        self.Omega_DM = 0.265
        self.Omega_rad = 5.38e-5

        self.H0 = ((100 * self.h * u.km / u.second / u.Mpc * ac.hbar).to("eV") / u.eV).to_value()
        self.rho_c = rho_c = 3 * self.H0**2 * self.M_pl**2
        self.Omega_DM_h_sq = self.Omega_DM * self.h**2

        self.g_star_R_today = 3.36
        self.z_eq = 3365
        self.T_eq = (30 / np.pi**2 * self.rho_c * self.Omega_rad / self.g_star_R_today)**(1/4) * (1 + self.z_eq)

        # particles
        self.m_u = 2.3e6
        self.m_d = 4.8e6
        self.m_s = 95e6
        self.m_pi0 = 134.9770e6
        self.f_pi0 = 130e6 / 2**0.5 # PDG uses different convention

        # errors from PDG
        self.M_pl_err = 0.000014e19 * 1e9 / np.sqrt(8*np.pi)
        self.Lambda_QCD_err = 20e6
        self.m_u_err = (0.49 + 0.26) / 2 * 1e6
        self.m_d_err = (0.48 + 0.17) / 2 * 1e6
        self.m_pi0_err = 0.0005 * 1e6
        self.f_pi0_err = 5e6
        self.T0_err = 6e-5 * c.Boltzmann / c.elementary_charge
        self.Omega_DM_h_sq_err = 0.00002
        self.h_err = 0.0009
        self.H0_err = ((100 * self.h_err * u.km / u.second / u.Mpc * ac.hbar).to("eV") / u.eV).to_value()
        self.rho_c_err = np.sqrt((3 * self.H0**2 * 2 * self.M_pl * self.M_pl_err)**2 + (3 * 2 * self.H0 * self.M_pl**2 * self.H0_err)**2)

    def __str__(self):
        return "Model " + str(self.__dict__)

model = Parameter()
parameter = model

plot_path = "../Plots"
data_path = "../Data"
