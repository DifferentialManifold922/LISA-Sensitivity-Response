#This program computes the analytical expression and approximation of <(4F^{+}_{X})^2> and plots 
import numpy as np
import matplotlib.pyplot as plt
import main
from multiprocessing import Pool
from tqdm import tqdm
import time
from scipy.stats import qmc
from scipy.interpolate import interp1d

#global variables
global_n31 = None
global_n12 = None
global_tm = None
global_L = None

#initializaton
def init_globals():
    global global_n31, global_n12, global_tm, global_L
    year = main.year
    global_tm = np.arange(0, 1000, 50)  
    n23, global_n31, global_n12 = main.LISAmotion(global_tm)
    global_L = main.L_m / main.c

#<(F^{+}_{x})^2>
def compute_FXp2_batch(freq_batch):
    global global_n31, global_n12, global_tm, global_L
    results = []
    for f in freq_batch:
        om = 2.0 * np.pi * f
        val = []
        for ntr in range(3):
            ti = np.random.choice(len(global_tm))
            N = 4096
            sampler = qmc.Sobol(d=3, scramble=True)
            samples = sampler.random(n=N)
            psi_samples = samples[:, 0] * (2.0 * np.pi)  
            lam_samples = samples[:, 1] * (2.0 * np.pi)
            bet_samples = -0.5 * np.pi + samples[:, 2] * (0.5 * np.pi - (-0.5 * np.pi))
            integrand_values = np.array([main.FXp2(psi, lam, bet, global_n31, global_n12, ti, om, global_L)
                                        for psi, lam, bet in zip(psi_samples, lam_samples, bet_samples)])
            volume = (2.0 * np.pi - 0.0) * (2.0 * np.pi - 0.0) * (0.5 * np.pi - (-0.5 * np.pi))
            res = np.mean(integrand_values) * volume
            val.append(res)
        results.append(np.mean(val) / (8.0 * np.pi**2))  
    return results

if __name__ == "__main__":
    init_globals()
    freqs = np.logspace(np.log10(1e-4), np.log10(1.0), num=50)  
    freq_batches = [freqs[i:i+1] for i in range(0, len(freqs), 1)]
    start_time = time.time()
    with Pool(processes=4) as pool:
        results = list(tqdm(pool.map(compute_FXp2_batch, freq_batches), total=len(freq_batches), desc="Computing <(F^{+}_X)^2>"))
    AvFXp2 = np.concatenate(results)
    print(f"Total time: {time.time() - start_time} seconds")
    print("Low-frequency results:", AvFXp2[:3])

#constants
    L_m = main.L_m
    c = main.c
    L = L_m / c
    omegaL = 2.0 * np.pi * freqs * L

    correction_term = 16 * (3/20) * (1 / (1 + 0.6 * omegaL**2))

    
    fPts = np.array([0.0001, 0.001, 0.01, 0.1])
    omegaL_pts = 2.0 * np.pi * fPts * L
    correction_term_pts = 16 * (3/20) * (1 / (1 + 0.6 * omegaL_pts**2))

    
    interp_AvFXp2 = interp1d(freqs, AvFXp2, bounds_error=False, fill_value="extrapolate")
    AvFXp2_pts = interp_AvFXp2(fPts)

  
    FigSize = (8, 6)
    plt.figure(figsize=FigSize)
    plt.semilogx(freqs, AvFXp2, 'o-', label=r'$\langle (F^{+}_{X})^2 \rangle$ (Numerical)', 
                 markersize=4, linewidth=1)  
    plt.semilogx(freqs, correction_term, 's--', label=r'$16 \times \frac{3}{20} \times \frac{1}{1 + 0.6 (\omega L)^2}$', 
                 markersize=4, linewidth=1)  
    plt.xlabel(r'Frequency ($Hz$)')
    plt.ylabel('Response')
    plt.title(r'Averaged Antenna Response and Correction Term vs. Frequency')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('results/Numerical_approximation.png', dpi=300, bbox_inches='tight')
    plt.show()

   
    print("\nValues at specified frequencies:")
    for freq, avfxp2_val, corr_val in zip(fPts, AvFXp2_pts, correction_term_pts):
        print(f"f = {freq} Hz:")
        print(f"  AvFXp2 (Numerical) = {avfxp2_val}")
        print(f"  Correction Term = {corr_val}")
