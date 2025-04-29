'''
This program defines the functions that will be used later and plots the analytical expressions of <(4F^{+/x}{X})^2> and <F^{+}{X} (F^{x}{X})^{*} + (F^{+}{X})^{*} F^{x}_{X}>
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc
from multiprocessing import Pool
from tqdm import tqdm
from numba import jit

# Constants
AUs = 499.00478383615643
year = 31558149.763545603
c = 299792458.0
L_m = 2.5e9
L = L_m / c
f_min = 1e-4
f_max = 1.0
num_points = 50  
f = np.logspace(np.log10(f_min), np.log10(f_max), num=num_points)
tm = np.arange(0, 1000, 50)

#LISAmotion
def LISAmotion(tm):
    lamb = 0.0
    kappa = 0.0
    N = len(tm)
    a = AUs
    e = L / (2.0 * np.sqrt(3.0) * a)
    nn = np.array([1, 2, 3])
    Beta = (nn - 1) * 2.0 * np.pi / 3.0 + lamb
    alpha = 2.0 * np.pi * tm / year + kappa
    x = np.zeros((3, N))
    y = np.zeros((3, N))
    z = np.zeros((3, N))
    for i in range(3):
        x[i, :] = a * np.cos(alpha) + a * e * (np.sin(alpha) * np.cos(alpha) * np.sin(Beta[i]) - (1.0 + (np.sin(alpha))**2) * np.cos(Beta[i]))
        y[i, :] = a * np.sin(alpha) + a * e * (np.sin(alpha) * np.cos(alpha) * np.cos(Beta[i]) - (1.0 + (np.cos(alpha))**2) * np.sin(Beta[i]))
        z[i, :] = -np.sqrt(3.0) * a * e * np.cos(alpha - Beta[i])
    n23 = np.array([x[1, :] - x[2, :], y[1, :] - y[2, :], z[1, :] - z[2, :]]) / L
    n31 = np.array([x[2, :] - x[0, :], y[2, :] - y[0, :], z[2, :] - z[0, :]]) / L
    n12 = np.array([x[0, :] - x[1, :], y[0, :] - y[1, :], z[0, :] - z[1, :]]) / L
    return (n23, n31, n12)

#Psi 
def Psi(lam, bet, n, om, L):
    k = -1. * np.array([np.cos(bet) * np.cos(lam), np.cos(bet) * np.sin(lam), np.sin(bet)])
    kn = np.dot(k, n)
    x = om * L
    Gam_rs = np.sinc((1. - kn) * 0.5 * x / np.pi) * np.exp(-0.5j * x * (1. - kn))
    Gam_sr = np.sinc((1. + kn) * 0.5 * x / np.pi) * np.exp(-0.5j * x * (1. + kn))
    Psi_rs = Gam_rs + Gam_sr * np.exp(-1.j * x * (1 - kn))
    return Psi_rs

#F^{+}_rs F^{x}_rs
def Fp_rs(psi, lam, bet, n):
    u = np.array([np.sin(lam), -np.cos(lam), 0.0])
    v = np.array([-np.sin(bet) * np.cos(lam), -np.sin(bet) * np.sin(lam), np.cos(bet)])
    nu = np.dot(u, n)
    nv = np.dot(v, n)
    plus = nu * nu - nv * nv
    cros = 2. * nu * nv
    Fp = np.cos(psi) * plus + np.sin(psi) * cros
    return Fp

def Fc_rs(psi, lam, bet, n):
    u = np.array([np.sin(lam), -np.cos(lam), 0.0])
    v = np.array([-np.sin(bet) * np.cos(lam), -np.sin(bet) * np.sin(lam), np.cos(bet)])
    nu = np.dot(u, n)
    nv = np.dot(v, n)
    plus = nu * nu - nv * nv
    cros = 2. * nu * nv
    Fc = -np.sin(psi) * plus + np.cos(psi) * cros
    return Fc

#F^{+}_{X} F^{x}_{X} 
def XPlus(psi, lam, bet, n31, n12, ti, om, L):
    F13 = Fp_rs(psi, lam, bet, -n31[:, ti])
    F12 = Fp_rs(psi, lam, bet, n12[:, ti])
    Psi13 = Psi(lam, bet, -n31[:, ti], om, L)
    Psi12 = Psi(lam, bet, n12[:, ti], om, L)
    return (F13 * Psi13 - F12 * Psi12)

def XCros(psi, lam, bet, n31, n12, ti, om, L):
    F13 = Fc_rs(psi, lam, bet, -n31[:, ti])
    F12 = Fc_rs(psi, lam, bet, n12[:, ti])
    Psi13 = Psi(lam, bet, -n31[:, ti], om, L)
    Psi12 = Psi(lam, bet, n12[:, ti], om, L)
    return (F13 * Psi13 - F12 * Psi12)

#<F^{+}_{X} (F^{x}_{X})* + (F^{+}_{X})* F^{x}_{X}>
def compute_cross_term(f, n23, n31, n12):
    om = 2.0 * np.pi * f
    ti = np.random.choice(len(n12[0]))
    FX_cross = lambda psi, lam, bet: np.cos(bet) * 2.0 * np.real(
        XPlus(psi, lam, bet, n31, n12, ti, om, L) * np.conj(XCros(psi, lam, bet, n31, n12, ti, om, L))
    )
    res = sc.tplquad(FX_cross, -0.5 * np.pi, 0.5 * np.pi,
                    lambda lam: 0.0, lambda lam: 2. * np.pi,
                    lambda lam, bet: 0.0, lambda lam, bet: 2. * np.pi,
                    epsabs=1e-3, epsrel=1e-3)
    print(f"Frequency {f}: Cross term integral = {res[0]}")
    return res[0] / (8.0 * np.pi**2)

#Compute power
def compute_power(f, n23, n31, n12):
    om = 2.0 * np.pi * f
    ti = np.random.choice(len(n12[0]))
    FXp2 = lambda psi, lam, bet: np.cos(bet) * (np.abs(XPlus(psi, lam, bet, n31, n12, ti, om, L)))**2
    FXc2 = lambda psi, lam, bet: np.cos(bet) * (np.abs(XCros(psi, lam, bet, n31, n12, ti, om, L)))**2
    res1 = sc.tplquad(FXp2, -0.5 * np.pi, 0.5 * np.pi,
                      lambda lam: 0.0, lambda lam: 2. * np.pi,
                      lambda lam, bet: 0.0, lambda lam, bet: 2. * np.pi,
                      epsabs=1e-3, epsrel=1e-3)
    res2 = sc.tplquad(FXc2, -0.5 * np.pi, 0.5 * np.pi,
                      lambda lam: 0.0, lambda lam: 2. * np.pi,
                      lambda lam, bet: 0.0, lambda lam, bet: 2. * np.pi,
                      epsabs=1e-3, epsrel=1e-3)
    print(f"Frequency {f}: XPlus integral = {res1[0]}, XCros integral = {res2[0]}")
    return (res1[0] / (8.0 * np.pi**2), res2[0] / (8.0 * np.pi**2))

#Numba
@jit(nopython=True)
def Fp_rs(psi, lam, bet, n):
    u = np.array([np.sin(lam), -np.cos(lam), 0.0])
    v = np.array([-np.sin(bet) * np.cos(lam), -np.sin(bet) * np.sin(lam), np.cos(bet)])
    nu = np.dot(u, n)
    nv = np.dot(v, n)
    plus = nu * nu - nv * nv
    cros = 2.0 * nu * nv
    Fp = np.cos(psi) * plus + np.sin(psi) * cros
    return Fp

@jit(nopython=True)
def Psi(lam, bet, n, om, L):
    k = np.array([-np.cos(bet) * np.cos(lam), -np.cos(bet) * np.sin(lam), -np.sin(bet)])
    kn = np.dot(k, n)
    x = om * L
    Gam_rs = np.sinc((1. - kn) * 0.5 * x / np.pi) * np.exp(-0.5j * x * (1. - kn))
    Gam_sr = np.sinc((1. + kn) * 0.5 * x / np.pi) * np.exp(-0.5j * x * (1. + kn))
    Psi_rs = Gam_rs + Gam_sr * np.exp(-1.j * x * (1 - kn))
    return Psi_rs

@jit(nopython=True)
def XPlus(psi, lam, bet, n31, n12, ti, om, L):
    F13 = Fp_rs(psi, lam, bet, -n31[:, ti])
    F12 = Fp_rs(psi, lam, bet, n12[:, ti])
    Psi13 = Psi(lam, bet, -n31[:, ti], om, L)
    Psi12 = Psi(lam, bet, n12[:, ti], om, L)
    return F13 * Psi13 - F12 * Psi12

@jit(nopython=True)
def FXp2(psi, lam, bet, n31, n12, ti, om, L):
    result = XPlus(psi, lam, bet, n31, n12, ti, om, L)
    return np.cos(bet) * (np.abs(result))**2

if __name__ == "__main__":
    freqs = np.logspace(np.log10(1e-4), np.log10(1.0), num=50)
    n23, n31, n12 = LISAmotion(tm)
    with Pool(processes=4) as pool:
        results = pool.starmap(compute_power, [(f, n23, n31, n12) for f in tqdm(freqs, desc="Computing power")])
    fplsSA_fPts, fcrsSA_fPts = zip(*results)
    fplsSA_fPts = np.array(fplsSA_fPts)
    fcrsSA_fPts = np.array(fcrsSA_fPts)
    with Pool(processes=4) as pool:
        cross_results = pool.starmap(compute_cross_term, [(f, n23, n31, n12) for f in tqdm(freqs, desc="Computing cross term")])
    cross_fPts = np.array(cross_results)

    print("fplsSA_fPts:", fplsSA_fPts)
    print("fcrsSA_fPts:", fcrsSA_fPts)
    print("cross_fPts:", cross_fPts)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.loglog(freqs, fplsSA_fPts, label=r'$<(4F^{+}_{X})^2>$', color='blue', linewidth=2, linestyle='-', alpha=0.7)
    ax1.loglog(freqs, fcrsSA_fPts, label=r'$<(4F^{x}_{X})^2>$', color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power Response', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which="both", ls="--")
    ax1.set_ylim(1e-4, 1e1)

    ax2 = ax1.twinx()
    ax2.loglog(freqs, np.abs(cross_fPts), label=r'$|<F^{+}_{X} F^{\times *}_{X} + F^{+ *}_{X} F^{\times}_{X}>|$', 
              color='green', linewidth=2, linestyle=':', alpha=0.7)
    ax2.set_ylabel('Cross Term (Absolute)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(1e-25, 1e-15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')
    
    plt.title('Averaged Antenna Response Functions and Cross Term', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/mainfig.png', dpi=300, bbox_inches='tight')
