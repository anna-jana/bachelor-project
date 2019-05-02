import configparser
import sys
import numpy as np
import scipy.constants as c
import astropy.units as u
import astropy.constants as ac

class Model:
    def __init__(self, model_filename = "model.ini"):

        # setup config parser
        parser = configparser.ConfigParser()
        parser.read(model_filename)

        # constants
        self.c = parser.getfloat("constants", "c")
        self.M_pl = parser.getfloat("constants", "M_pl") * 1e9 / np.sqrt(8*np.pi)
        self.G = parser.getfloat("constants", "G")
        self.Lambda_QCD = parser.getfloat("constants", "Lambda_QCD")

        # cosmological parameters
        self.h = parser.getfloat("cosmology", "h")
        self.z_eq = parser.getfloat("cosmology", "z_eq")
        self.T0 = parser.getfloat("cosmology", "T0") * c.Boltzmann  / c.elementary_charge
        self.Omega_DM = parser.getfloat("cosmology", "Omega_DM")
        self.Omega_rad = parser.getfloat("cosmology", "Omega_rad")

        self.H0 = ((100 * self.h * u.km / u.second / u.Mpc * ac.hbar).to("eV") / u.eV).to_value()
        self.rho_c = rho_c = 3 * self.H0**2 * self.M_pl**2
        self.Omega_DM_h_sq = self.Omega_DM * self.h**2

        # particles
        self.m_u = parser.getfloat("particles", "m_u")
        self.m_d = parser.getfloat("particles", "m_d")
        self.m_s = parser.getfloat("particles", "m_s")
        self.m_pi0 = parser.getfloat("particles", "m_pi0")

        # PDG uses different convention
        self.f_pi0 = parser.getfloat("particles", "f_pi0") / 2**0.5

        # g_star model

        # m_a(T) model

        # potential model

        # ODE solver options

        # scaling options

        # f_a, theta_i paramter options

        # error estimation

    def __str__(self):
        return "Model " + str(self.__dict__)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        model = Model()
    else:
        model_filename = sys.argv[1]
        model = Model(model_filename)

