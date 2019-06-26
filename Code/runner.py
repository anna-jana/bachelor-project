import multiprocessing as mp
import sys

import numpy as np

import eom
import axion_mass
import potential
import g_star
import config

# I am sorry. This is bc. one can only pass global functions to mp.Pool.map
__global_model = None

def worker_fn(p):
    # I am sorry
    global __global_model
    solver = __global_model.get_solver(p[0], p[1])
    ans = solver.compute_density_parameter()
    print("=", end=""); sys.stdout.flush()
    return ans

def compute_density_parameter(theta_i_s, f_a_s, model, num_workers=6):
    print("\n" + "-" * 60)
    for i in range(len(theta_i_s) * len(f_a_s)):
        print("=", end="")
    print("\n" + "-" * 60)
    points = [(theta_i, f_a) for theta_i in theta_i_s for f_a in f_a_s]
    # I am sorry
    global __global_model
    __global_model = model
    with mp.Pool(num_workers) as poolparty:
        ans = poolparty.map(worker_fn, points, 1)
    Omega_a_h_sq = np.array(ans).reshape(theta_i_s.size, f_a_s.size)
    return Omega_a_h_sq

def save_data(filename, Omega_a_h_sq, theta_i_s, f_a_s):
    np.savez(filename, Omega_a_h_sq=Omega_a_h_sq, theta_i_s=theta_i_s, f_a_s=f_a_s)

def load_data(filename):
    f = np.load(filename)
    Omega_a_h_sq = f["Omega_a_h_sq"]
    theta_i_s = f["theta_i_s"]
    f_a_s = f["f_a_s"]
    return Omega_a_h_sq, theta_i_s, f_a_s

if __name__ == "__main__":
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

    m_a_fn = axion_mass.m_a_from_chi_general
    g_star_model = g_star.matched

    model = eom.Model(m_a_fn, g_star_model, potential.cosine)
    Omega_a_h_sq = compute_density_parameter(theta_i, f_a, model, num_workers=workers)
    save_data(config.data_path + "/cosine.npz", Omega_a_h_sq, theta_i, f_a)

    model = eom.Model(m_a_fn, g_star_model, potential.harmonic)
    Omega_a_h_sq = compute_density_parameter(theta_i, f_a, model, num_workers=workers)
    save_data(config.data_path + "/harmonic.npz", Omega_a_h_sq, theta_i, f_a)
