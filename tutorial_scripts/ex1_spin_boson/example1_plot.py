import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

file_sb = Path("train.dat.log")  
time_units = "fs"
pop_labels = [r"$\rho_{00}$TT", r"$\rho_{11}$TT"]

file_sb2 = Path("tree2.dat.log")
time_units = "fs"
pop_labels2 = [r"$\rho_{00}Tree$", r"$\rho_{11}$Tree"]


# Drude–Lorentz params (cm⁻¹)
re_d    = [540]
width_d = [70]
# Underdamped Brownian params (cm⁻¹)
freq_b  = [1243]
re_b    = [161.6]
width_b = [10]
# --------------------------------------------------

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", "font.serif": ["Times"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "axes.linewidth": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 300,
})

def load_spinboson_pops(fname, convert_to_ps=False):
    arr = np.genfromtxt(fname, dtype=complex, comments="#", skip_header=1)
    t = arr[:,0].real
    if convert_to_ps:
        t *= 1e-3
    pops = np.vstack((arr[:,1].real, arr[:,4].real)).T
    return t, pops

def drude_sd(omega, re_d, width_d):
    J = np.zeros_like(omega)
    for lam, gamma in zip(re_d, width_d):
        J += 2*lam*gamma*omega/(3.14159*(omega**2 + gamma**2))
    return J

def brownian_sd(omega, re_b, freq_b, width_b):
    J = np.zeros_like(omega, dtype=float)
    for lam, w0, g in zip(re_b, freq_b, width_b):
        w1 = np.sqrt(max(w0**2 - g**2, 0.0))   # ω₁ = √(Ω²−γ²)
        J += (4.0 / np.pi) * lam * g * w0**2 * omega / (
            ((omega + w1)**2 + g**2) * ((omega - w1)**2 + g**2)
        )
    return J

# load data
convert = (time_units.lower() == "ps")
t_sb, p_sb = load_spinboson_pops(file_sb, convert_to_ps=convert)
t_sb2, p_sb2 = load_spinboson_pops(file_sb2, convert_to_ps=convert)

# build spectral densities
omega = np.linspace(0, 2000, 2000)
J_dl = drude_sd(omega, re_d, width_d)
J_bo = brownian_sd(omega, re_b, freq_b, width_b)
J_tot = J_dl + J_bo

# make figure
fig, (ax1, ax2) = plt.subplots(
    2,1, figsize=(3.4, 3.0), gridspec_kw={"hspace":0.4}
)

# --- (a) populations ---
ax1.plot(t_sb, p_sb[:,0], label=r"$\rho_{11}$ TT")
ax1.plot(t_sb, p_sb[:,1], label=r"$\rho_{00}$ TT")

ax1.plot(t_sb2, p_sb2[:,0],'--', label=r"$\rho_{11}$ TTN")
ax1.plot(t_sb2, p_sb2[:,1], '--', label=r"$\rho_{00}$ TTN")

ax1.set_ylabel("Population", labelpad=6)
ax1.set_xlabel("Time (fs)", labelpad=3)
ax1.set_xlim(0, 1000)
ax1.text(-0.12,1.02,"(a)", transform=ax1.transAxes,
         fontsize=9, va="bottom")

handles, labs = ax1.get_legend_handles_labels()
fig.legend(handles, labs, loc="upper center",
           ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=8,
           frameon=False)

# --- (b) total spectral density ---
ax2.plot(omega, J_tot/1000, color="0.2", label=r"$J(\omega)$")
ax2.set_ylabel(r"$J(\omega)\,/\,10^3\mathrm{cm}^{-1}$", labelpad=6)
ax2.set_xlabel(r"Frequency (cm$^{-1}$)", labelpad=6)
ax2.text(-0.12,1.02,"(b)", transform=ax2.transAxes,
         fontsize=9, va="bottom")

# --- inset: zoom DL part ---
axins = inset_axes(ax2, width="40%", height="30%", loc="upper left", borderpad=2)
axins.plot(omega, J_dl, color="C1", linewidth=1.2)
axins.set_xlim(0, 200)
axins.set_ylim(0, J_dl[omega<=200].max()*1.1)
axins.set_xticks([0,100,200])
axins.set_yticks([])

fig.tight_layout(rect=(0,0,1,0.88))
plt.savefig("SpinBoson_populations.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()
