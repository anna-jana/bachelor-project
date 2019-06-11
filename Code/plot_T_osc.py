import numpy as np
import matplotlib.pyplot as plt

import axion_mass
from config import model, plot_path
import T_osc_solver
import g_star

def plot_T_osc(m_a_model, label):
    g_model = g_star.borsamyi_table
    f_a = np.logspace(9, 18, 400) * 1e9 # eV
    # f_a = np.logspace(-18, 18, 400) * 1e9 # eV
    T_osc = np.array(list(map(lambda f_a: T_osc_solver.find_T_osc(f_a, m_a_model, g_model), f_a)))
    plt.loglog(f_a / 1e9, T_osc / 1e6, label=label)
    # plt.loglog(f_a / 1e9, T_osc * c.elementary_charge / c.Boltzmann, label=label)
    plt.xlabel(r"$f_a [\mathrm{GeV}]$", fontsize=16)
    plt.ylabel(r"$T_\mathrm{osc} [\mathrm{MeV}]$", fontsize=16)
    # plt.ylabel(r"$T_\mathrm{osc} [\mathrm{K}]$", fontsize=16)

plot_T_osc(axion_mass.m_a_fox, "DIGA (fox)")
plot_T_osc(axion_mass.m_a_shellard, "IILA (shellard)")
plot_T_osc(axion_mass.m_a_from_chi_general, r"$\chi_\mathrm{Lattice}$ (Bosaminy)")
plt.axhline(model.Lambda_QCD / 1e6, color="black", label=r"$\Lambda_\mathrm{QCD}$")
# plt.axhline(model.Lambda_QCD * c.elementary_charge / c.Boltzmann, color="black", label=r"$\Lambda_\mathrm{QCD}$")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path + "/T_osc_plot.pdf")
