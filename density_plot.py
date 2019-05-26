import matplotlib.pyplot as plt
import numpy as np

from config import model

def plot_density(theta_i, f_a, Omega_a_h_sq, levels=10, plot_type="pcolormesh"):
    Omega_a_h_sq = Omega_a_h_sq.copy()
    Omega_a_h_sq[Omega_a_h_sq > model.Omega_DM_h_sq] = 0.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if plot_type == "pcolormesh":
        plt.pcolormesh(f_a / 1e9, theta_i, np.log10(Omega_a_h_sq))
    elif plot_type == "contourf":
        plt.contourf(f_a / 1e9, theta_i, np.log10(Omega_a_h_sq), levels)
    else:
        raise ValueError("invalid plot_type")
    plt.xlabel(r"$f_a / \mathrm{GeV}$")
    plt.ylabel(r"$\theta_i$")
    plt.text(0.55, 0.8, r"$\Omega_a h^2 > \Omega_\mathrm{DM} h^2 = 0.12$", transform=ax.transAxes)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$\log_{10}(\Omega_a h^2)$")
    return fig
