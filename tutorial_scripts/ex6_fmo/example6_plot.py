import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ==================== CONFIGURATION ====================

# --- File Paths ---
files = {
    "Structured": {
        "300K": Path("3_level_FMO_structured_300K.dat.log"),
        "77K":  Path("3_level_FMO_structured_77_2.dat.log")
    },
    "3-Peak": {
        "300K": Path("3_level_FMO_three_peak_300K.dat.log"),
        "77K":  Path("3_level_FMO_threepeak_77K.dat.log")
    },
    "Drude": {
        "300K": Path("3_level_FMO_dl300_2.dat.log"),
        "77K":  Path("fmo_77K_dl.dat.log")
    }
}

# --- Physics Parameters (cm^-1) ---
W_str = np.array([160, 247, 763, 1175, 1356, 1521], dtype=float)
G_str = np.array([133, 53, 76, 29, 29, 15], dtype=float)
L_str = np.array([26.24, 13.832, 101.479, 57.575, 25.764, 9.126], dtype=float)

W_3pk = np.array([160, 247, 763], dtype=float)
G_3pk = np.array([133, 53, 76], dtype=float)
L_3pk = np.array([26.24, 13.832, 101.479], dtype=float)

L_DL  = 35.0
Wc_DL = 106.1767

# --- Plot Styling ---
site_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
model_styles = {"Structured": "-", "3-Peak": "--", "Drude": ":"}
model_colors_sd = {"Structured": "k", "3-Peak": "tab:red", "Drude": "tab:purple"}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", "font.serif": ["Times"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "axes.linewidth": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 300,
})

# ==================== FUNCTIONS ====================

def load_populations(fname):
    try:
        raw = np.genfromtxt(fname, comments="#", dtype=complex)
        t = raw[:, 0].real
        p00 = raw[:, 1].real
        p11 = raw[:, 5].real
        p22 = raw[:, 9].real
        return t, np.vstack([p00, p11, p22]).T
    except OSError:
        t = np.linspace(0, 1000, 100)
        return t, np.zeros((100, 3))

def j_brownian(w, W, G, L):
    """Factored form with ω₁ = √(Ω²−γ²) computed internally."""
    J = np.zeros_like(w, dtype=float)
    for w0, g, lam in zip(W, G, L):
        w1 = np.sqrt(max(w0**2 - g**2, 0.0))   # ω₁ = √(Ω²−γ²)
        J += (4.0 / np.pi) * lam * g * w0**2 * w / (
            ((w + w1)**2 + g**2) * ((w - w1)**2 + g**2)
        )
    return J

def j_drude(w, lam, wc):
    return (2.0 * lam * wc * w) / (w**2 + wc**2)

# ==================== PLOTTING ====================

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(3.4, 3.7), sharex=False,
    gridspec_kw={"height_ratios": [1, 1, 0.85], "hspace": 0.05}
)

# --- LOAD DATA ---
data = {"300K": {}, "77K": {}}
for model in files:
    for temp in ["300K", "77K"]:
        t, pops = load_populations(files[model][temp])
        data[temp][model] = (t, pops)

# --- PANEL (a): 300 K ---
temp = "300K"
for model in files:
    t, pops = data[temp][model]
    for site in range(3):
        ax1.plot(t, pops[:, site],
                 color=site_colors[site],
                 linestyle=model_styles[model],
                 alpha=0.9)

ax1.set_ylabel(r"Population")
ax1.text(0.96, 0.88, r"(a) $300\,\mathrm{K}$",
         transform=ax1.transAxes, ha='right', va='top', fontsize=9)
ax1.set_xticklabels([])
ax1.set_xlim(0, 1000)
ax1.set_ylim(-0.05, 1.05)

# --- PANEL (b): 77 K ---
temp = "77K"
for model in files:
    t, pops = data[temp][model]
    for site in range(3):
        ax2.plot(t, pops[:, site],
                 color=site_colors[site],
                 linestyle=model_styles[model],
                 alpha=0.9)

ax2.set_ylabel(r"Population")
ax2.set_xlabel("Time (fs)", labelpad=1)
ax2.text(0.96, 0.88, r"(b) $77\,\mathrm{K}$",
         transform=ax2.transAxes, ha='right', va='top', fontsize=9)
ax2.set_xlim(0, 1000)
ax2.set_ylim(-0.05, 1.05)

# --- PANEL (c): SPECTRAL DENSITIES ---
w_cm = np.linspace(0, 1700, 1000)
J_str = j_brownian(w_cm, W_str, G_str, L_str)
J_3pk = j_brownian(w_cm, W_3pk, G_3pk, L_3pk)
J_dru = j_drude(w_cm, L_DL, Wc_DL) / np.pi

ax3.plot(w_cm, J_str/1000, color=model_colors_sd["Structured"], linestyle="-", label="6-peak")
ax3.plot(w_cm, J_3pk/1000, color=model_colors_sd["3-Peak"], linestyle="--", label="3-Peak")
ax3.plot(w_cm, J_dru/1000, color=model_colors_sd["Drude"], linestyle=":", label="DL")

ax3.set_ylabel(r"$J(\omega)$ ($10^3$ cm$^{-1}$)")
ax3.set_xlabel(r"Frequency $\omega$ (cm$^{-1}$)", labelpad=1)
ax3.text(0.96, 0.88, r"(c)",
         transform=ax3.transAxes, ha='right', va='top', fontsize=9)
ax3.set_xlim(0, 1700)
ax3.set_ylim(0, 1.01)

ax3.legend(fontsize=7, frameon=False, loc="upper left", ncol=2,
           columnspacing=1.0, handlelength=1.5, borderaxespad=0.2)

# --- GLOBAL LEGEND (Top) ---
legend_elements_top = [
    Line2D([0], [0], color=site_colors[0], lw=1.5, label='Site 1'),
    Line2D([0], [0], color=site_colors[1], lw=1.5, label='Site 2'),
    Line2D([0], [0], color=site_colors[2], lw=1.5, label='Site 3'),
]

fig.legend(handles=legend_elements_top, loc='upper center',
           bbox_to_anchor=(0.53, 0.95), ncol=3, fontsize=8, frameon=False,
           columnspacing=1.0, handlelength=1.5)

fig.tight_layout(rect=[0, 0, 1, 0.94])

pos2 = ax2.get_position()
pos3 = ax3.get_position()
desired_gap = 0.08
new_bottom = pos2.y0 - pos3.height - desired_gap
ax3.set_position([pos3.x0, new_bottom, pos3.width, pos3.height])

plt.savefig("FMO_Comparison_Squeezed.png", bbox_inches='tight', dpi=300)
plt.show()
