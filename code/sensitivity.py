#This program computes sensitivity (SciRD) and plots
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import main  

#constants
c = 299792458.0
year = 31558149.763545603
AUs = 499.00478383615643
L_m = 2.5e9
L = L_m / c

#Using SciRD
sqSnoise_SciRD = [3e-15, 15e-12]

#frequencies
fMin = 2e-5
fMax = 1
fPts = np.array([0.0001, 0.001, 0.01, 0.1, 1.0])
freqs_main = np.logspace(np.log10(1e-4), np.log10(1.0), 50)  
frequencies_plot = np.logspace(np.log10(fMin), np.log10(fMax), 100) 
frequencies_plot = np.unique(np.sort(np.concatenate([frequencies_plot, fPts, freqs_main]))) 
tm = np.arange(0, 1000, 50)

#S_{OMS} S_{acc}
def PSD_Noise_components(fr, sqSnoise=sqSnoise_SciRD):
    [sqSacc_level, sqSoms_level] = sqSnoise
    Sa_a = sqSacc_level**2 * (1.0 + (0.4e-3/fr)**2) * (1.0 + (fr/8e-3)**4)
    Sa_d = Sa_a * (2.*np.pi*fr)**(-4.)
    Sa_nu = Sa_d * (2.0*np.pi*fr/c)**2
    Soms_d = sqSoms_level**2 * (1. + (2.e-3/fr)**4)
    Soms_nu = Soms_d * (2.0*np.pi*fr/c)**2
    return [Sa_nu, Soms_nu]

#compute
def parallel_noise_calc(chunk):
    return np.array([PSD_Noise_components(f) for f in chunk])

def compute_noise_parallel(frequencies, num_cores=4):
    chunksize = max(1, len(frequencies) // num_cores)
    chunks = np.array_split(frequencies, num_cores)
    with Pool(processes=num_cores) as pool:
        results = pool.map(parallel_noise_calc, chunks, chunksize=chunksize)
    results = np.vstack(results)
    Sa_nu = results[:, 0]
    Soms_nu = results[:, 1]
    return Sa_nu, Soms_nu

#compute<(F^{+}_{X})^2>
def compute_power_parallel(frequencies, n23, n31, n12, num_cores=4):
    n23 = np.ascontiguousarray(n23)  
    n31 = np.ascontiguousarray(n31)
    n12 = np.ascontiguousarray(n12)
    
    chunksize = max(1, len(frequencies) // num_cores)
    try:
        with Pool(processes=num_cores) as pool:
            results = pool.starmap(main.compute_power, 
                                  [(f, n23, n31, n12) for f in tqdm(frequencies, desc="Computing power")],
                                  chunksize=chunksize)
        return np.array([r[0] for r in results])  
    except Exception as e:
        print(f"Parallel computation failed: {e}")
        results = []
        for f in tqdm(frequencies, desc="Computing power (fallback)"):
            try:
                results.append(main.compute_power(f, n23, n31, n12)[0])
            except Exception as e2:
                print(f"Failed at f={f}: {e2}")
                results.append(np.nan)  
        return np.array(results)

#sensitivity
if __name__ == "__main__":
  
    n23, n31, n12 = main.LISAmotion(tm)

    fr = frequencies_plot
    Sa_nu_SciRD, Soms_nu_SciRD = compute_noise_parallel(fr, num_cores=4)
    phiL = 2 * np.pi * fr * L

    AvFXp2 = compute_power_parallel(fr, n23, n31, n12, num_cores=4)
    
    if np.any(np.isnan(AvFXp2)):
        print("Warning: NaN detected in AvFXp2, replacing with interpolated values")
        valid_idx = ~np.isnan(AvFXp2)
        interp_func = interp1d(fr[valid_idx], AvFXp2[valid_idx], kind='cubic', fill_value="extrapolate")
        AvFXp2 = interp_func(fr)

    S_hX_SA = (Soms_nu_SciRD + Sa_nu_SciRD * (3 + np.cos(2 * phiL))) / (phiL**2 * AvFXp2 / 16)
    S_h_SA = S_hX_SA / 2

    fr_pts = fPts
    Sa_nu_SciRD_pts, Soms_nu_SciRD_pts = PSD_Noise_components(fr_pts, sqSnoise_SciRD)
    phiL_pts = 2 * np.pi * fr_pts * L
    AvFXp2_pts = compute_power_parallel(fr_pts, n23, n31, n12, num_cores=4)
    S_hX_SA_pts = (Soms_nu_SciRD_pts + Sa_nu_SciRD_pts * (3 + np.cos(2 * phiL_pts))) / (phiL_pts**2 * AvFXp2_pts / 16)
    S_h_SA_pts = S_hX_SA_pts / 2

    #interpolation
    interp_func = interp1d(fr, S_hX_SA, kind='cubic', fill_value="extrapolate")
    fr_plot = np.logspace(np.log10(fMin), np.log10(fMax), 1000)
    S_hX_SA_plot = interp_func(fr_plot)

    #plot
    plt.figure(figsize=[8, 6])
    plt.loglog(fr_plot, np.sqrt(S_hX_SA_plot), color='#1f77b4', linewidth=2, antialiased=True)
    plt.scatter(fPts, np.sqrt(S_hX_SA_pts), color='#ff7f0e', marker='o', s=50, zorder=10)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel(r'$\sqrt{S_{h,X}}$: LISA Sensitivity TDI 4 links ($1/\sqrt{\text{Hz}}$)', fontsize=12)
    plt.title('Sensitivity curve for a single TDI-X observable', fontsize=14)
    plt.xlim(fMin, fMax)
    plt.ylim(1e-20, 1e-13)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('~/sensitivity.png', dpi=400, bbox_inches='tight')
    plt.close()

    print("\nUsing sqSnoise_SciRD = [3e-15, 15e-12]:")
    print("f (Hz) & S_hX_SA (Hz^-1) & S_h_SA (Hz^-1) & √S_hX_SA (1/√Hz) & √S_h_SA (1/√Hz) \\\\")
    for i in range(len(fPts)):
        print("%.5f & %.6e & %.6e & %.6e & %.6e \\" % (
            fPts[i], 
            S_hX_SA_pts[i], 
            S_h_SA_pts[i],
            np.sqrt(S_hX_SA_pts[i]),
            np.sqrt(S_h_SA_pts[i])
        ))
