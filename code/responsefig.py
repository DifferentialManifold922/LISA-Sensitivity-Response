#This program plots the fig of average GW response for TDI-X2.0 (Numerical and Semi-analytical).
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Constants (consistent with main.py)
c = 299792458.0
L_m = 2.5e9
L = L_m / c
f_min = 1e-4
f_max = 1.0
num_points = 500  
f = np.logspace(np.log10(f_min), np.log10(f_max), num=num_points)
fPts = [0.0001, 0.001, 0.01, 0.1]  # Specified frequencies

#Semi-analytical response for TDI-X2.0
def semi_analytical_response_x20(f, L):
    omega = 2 * np.pi * f
    omega_L = omega * L
    term1 = 64 * omega_L**2
    term2 = np.sin(omega_L)**2 * np.sin(2 * omega_L)**2
    term3 = 3 / 20
    term4 = 1 / (1 + 0.6 * omega_L**2)
    return term1 * term2 * term3 * term4

#Load simulation data
data_path = '/home/taffy/LISA/lisa_sensitivity_snr-master-Data/Data/PSD_LC2_Sim_LISA_XYZAETAmEmTm_1yr_SGWB1_NoNoise_12345x-TDI_X20.npy'
PSD_GWLCX20 = np.load(data_path)

#Get the frequency range from simulation data
f_sim = PSD_GWLCX20[:, 0]
f_sim_min = f_sim.min()
f_sim_max = f_sim.max()

#Interpolate simulation data for smoother curve within simulation frequency range
interp_func = interp1d(f_sim, PSD_GWLCX20[:, 1], kind='cubic', bounds_error=True)
f_smooth = np.logspace(np.log10(max(f_min, f_sim_min)), np.log10(min(f_max, f_sim_max)), num_points)
PSD_GWLCX20_smooth = interp_func(f_smooth)

#Interpolate simulation data at fPts
PSD_GWLCX20_Pts = np.interp(fPts, f_sim, PSD_GWLCX20[:, 1])

#Compute semi-analytical response
RXSAX20 = semi_analytical_response_x20(f, L)
RXSAX20_Pts = semi_analytical_response_x20(np.array(fPts), L)

#Print response values
print("===== TDI-X2.0 Response =====")
print("Frequency (Hz) & Semi-Analytic & Simulation (LISACode) \\\\")
for i in range(len(fPts)):
    print("%.5f & %.6e & %.6e \\\\" % (fPts[i], RXSAX20_Pts[i], PSD_GWLCX20_Pts[i]))

#Plot
plt.figure(figsize=(8, 6))
plt.loglog(f_smooth, PSD_GWLCX20_smooth, color='blue', label='X_2.0(Numerical, Smoothed)')
plt.loglog(f, RXSAX20, color='orange', label='X_2.0(Semi-Analytic)')
plt.loglog(fPts, RXSAX20_Pts, 'or', label='Reference Points (Analytic)')
plt.loglog(fPts, PSD_GWLCX20_Pts, 'ob', label='Reference Points (Numerical)')
plt.xlabel(r'Frequency (Hz)')
plt.ylabel(r'Response to GW $X_{2.0}$')
plt.xlim([f_min, f_max])
plt.ylim([1e-12, 1e2])
plt.legend()
plt.grid(True)
plt.savefig('/home/taffy/LISA/numerical_semi_response.png', dpi=300, bbox_inches='tight')
plt.close()