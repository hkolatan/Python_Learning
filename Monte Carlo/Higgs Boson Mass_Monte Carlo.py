import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("higgs_doublea.txt", sep="\t")

angles = df["angle"]
hist, bins = np.histogram(angles, bins=30, range=(0, np.pi))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.errorbar(bin_centers, hist, yerr=np.sqrt(hist), fmt="o", label="Data")
plt.xlabel("Angle (radians)")
plt.ylabel("Counts")
plt.legend()
plt.show()

#  Higgs bozonunun kütlesi ve hata payı
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

p0 = [300, 1.2, 0.1]
popt, pcov = curve_fit(gauss, bin_centers, hist, p0=p0, sigma=np.sqrt(hist))

plt.errorbar(bin_centers, hist, yerr=np.sqrt(hist), fmt="o", label="Data")
plt.plot(bin_centers, gauss(bin_centers, *popt), label="Fit")
plt.xlabel("Angle (radians)")
plt.ylabel("Counts")
plt.legend()
plt.show()

mass = popt[1]
mass_error = np.sqrt(pcov[1, 1])
print("Higgs mass = {:.2f} ± {:.2f} GeV/c^2".format(mass, mass_error))


# Monte Carlo Simülasyonu
n_mc = 10000
mc_angles = np.zeros(n_mc)
for i in range(n_mc):
    f1, f2 = np.random.uniform(0, 2 * np.pi, size=2)
    mc_angles[i] = np.abs(f1 - f2)

mc_hist, mc_bins = np.histogram(mc_angles, bins=30, range=(0, np.pi))
mc_bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])
plt.plot(mc_bin_centers, mc_hist, label="MC")
plt.plot(bin_centers, gauss(bin_centers, *popt), label="Data")
plt.xlabel("Angle (radians)")
plt.ylabel("Counts")
plt.legend()
plt.show()

mc_mass = np.mean(mc_angles)
mc_mass_error = np.std(mc_angles) / np.sqrt(n_mc)
print("Monte Carlo Higgs mass = {:.2f} ± {:.2f} GeV/c^2".format(mc_mass, mc_mass_error))