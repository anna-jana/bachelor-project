import multiprocessing as mp
import sys

import numpy as np

import eom_solver
import axion_mass
import potential
import g_star

# I am sorry. This is bc. one can only pass global functions to mp.Pool.map
__global_solver = None

def worker_fn(p):
    # I am sorry
    global __global_solver
    __global_solver.theta_i, __global_solver.f_a = p
    ans = __global_solver.compute_density_parameter()
    print("=", end=""); sys.stdout.flush()
    return ans

def compute_density_parameter(theta_i_range, f_a_range, solver, N=(10,10), num_workers=6):
    for i in range(N[0] * N[1]):
        print("=", end="")
    print("\n" + "-" * 40)
    if isinstance(theta_i_range, tuple):
        theta_i_s = np.logspace(np.log10(theta_i_range[0]), np.log10(theta_i_range[1]), N[0])
    else:
        theta_i_s = theta_i_range
    if isinstance(f_a_range, tuple):
        f_a_s = np.logspace(np.log10(f_a_range[0]), np.log10(f_a_range[1]), N[1]) * 1e9
    else:
        f_a_s = f_a_range
    points = [(theta_i, f_a) for i, theta_i in enumerate(theta_i_s) for j, f_a in enumerate(f_a_s)]
    # I am sorry
    global __global_solver
    __global_solver = solver
    with mp.Pool(num_workers) as poolparty:
        ans = poolparty.map(worker_fn, points, 1)
    Omega_a_h_sq = np.array(ans).reshape(theta_i_s.size, f_a_s.size)
    return theta_i_s, f_a_s, Omega_a_h_sq

def save_data(filename, Omega_a_h_sq, theta_i_s, f_a_s):
    np.savez(filename, Omega_a_h_sq=Omega_a_h_sq, theta_i_s=theta_i_s, f_a_s=f_a_s)

def load_data(filename):
    f = np.load(filename)
    Omega_a_h_sq = f["Omega_a_h_sq"]
    theta_i_s = f["theta_i_s"]
    f_a_s = f["f_a_s"]
    return Omega_a_h_sq, theta_i_s, f_a_s

if __name__ == "__main__":
    N = 5
    solver = eom_solver.EOMSolver(axion_mass.m_a_from_chi_general, g_star.borsamyi_table, potential.cosine)
    theta_i_s, f_a_s, Omega_a_h_sq = compute_density_parameter((1e-5, np.pi), (1e9, 1e19), solver , N=(N,N))
    save_data("cosine.npz", Omega_a_h_sq, theta_i_s, f_a_s)
    solver = eom_solver.EOMSolver(axion_mass.m_a_from_chi_general, g_star.borsamyi_table, potential.harmonic)
    theta_i_s, f_a_s, Omega_a_h_sq = compute_density_parameter((1e-5, np.pi), (1e9, 1e19), solver , N=(N,N))
    save_data("harmomic.npz", Omega_a_h_sq, theta_i_s, f_a_s)



