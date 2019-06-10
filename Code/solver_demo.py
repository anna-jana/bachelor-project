import numpy as np
import matplotlib.pyplot as plt

import axion_mass
import g_star
import eom
import potential
from config import plot_path

model = eom.Model(axion_mass.m_a_from_chi_general, g_star.borsamyi_table, potential.cosine)
solver = model.get_solver(1e-5, 1e9 * 1e12)
solver.solve_to_osc()
T, theta, dthetadT, n_over_s = solver.find_const_n_over_s()
plt.semilogx(T, theta)
plt.gca().invert_xaxis()
plt.xlabel("T [eV]")
plt.ylabel(r"$\theta$")
plt.savefig(plot_path + "/n_over_s_avg_field_plot.pdf")

model = eom.Model(axion_mass.m_a_from_chi_general, g_star.borsamyi_table, potential.cosine)
solver = model.get_solver(1e-5, 1e18 * 1e9, temperature_unit=1e15)
T, theta = solver.field_to_osc()
plt.semilogx(T, theta)
plt.gca().invert_xaxis()
plt.xlabel("T [eV]")
plt.ylabel(r"$\theta$")
plt.savefig(plot_path + "/field_before_osc_plot.pdf")
