import matplotlib.pyplot as plt
import numpy as np

from config import model

def plot_density(theta_i, f_a, Omega_a_h_sq, levels=10, plot_type="pcolormesh"):
    Omega_a_h_sq = Omega_a_h_sq.copy()
    Omega_a_h_sq[Omega_a_h_sq > model.Omega_DM_h_sq] = 0.0
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    if plot_type == "pcolormesh":
        plt.pcolormesh(f_a / 1e9, theta_i, np.log10(Omega_a_h_sq))
    elif plot_type == "contourf":
        plt.contourf(f_a / 1e9, theta_i, np.log10(Omega_a_h_sq), levels)
    else:
        raise ValueError("invalid plot_type")
    plt.xlabel(r"$f_a / \mathrm{GeV}$", fontsize=15)
    plt.ylabel(r"$\theta_i$", fontsize=15)
    plt.text(0.55, 0.8, r"$\Omega_a h^2 > \Omega_\mathrm{DM} h^2 = 0.12$", transform=ax.transAxes, fontsize=15)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$\log_{10}(\Omega_a h^2)$", fontsize=15)
