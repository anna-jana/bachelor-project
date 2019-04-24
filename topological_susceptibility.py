import numpy as np
import scipy.constants as c

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

T_K = data[:, 0] # [MeV]
minus_log10_chi = data[:, 1]

T = c.Boltzmann / c.elementary_charge * T_K
chi = (1 / c.elementary_charge * c.hbar * c.c / 1e-15)**4 * 10 ** (- minus_log10_chi)


