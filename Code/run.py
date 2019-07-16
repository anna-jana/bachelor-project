import numpy as np
import solver
import axion_mass
import g_star
import potential
import eom
import config

N = 100
workers = 4
theta_i = np.concatenate([
    np.logspace(-5, 0, N // 3),
    np.linspace(1, 3, N // 3 + 2)[1:-1],
    np.linspace(3, np.pi, N // 3),
])
f_a = np.concatenate([
    np.logspace(9, 16, N // 2),
    np.logspace(16, 19, N // 2 + 1)[1:],
]) * 1e9

ans = np.array([[solver.compute_relic_density(config.parameter, t, f) for f in f_a] for t in theta_i])

# [  2.18170129e+00   2.62958012e+01   3.89602398e+27   8.11795434e+07
 #          1.66239454e+06   4.74448978e+06   6.10393249e+05   2.46767815e+07
 #             4.45388802e-04   4.56386056e-12]

# theta_i = 2.18170129e+00
# f_a = 10**2.62958012e+01
# solver.compute_relic_density(config.parameter, theta_i, f_a)


