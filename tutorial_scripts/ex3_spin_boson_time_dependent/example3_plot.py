import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes # Not needed anymore

# -------------------- CONFIGURATION & PARAMETERS --------------------

# --- 1. Laser Physics Constants ---
Hartree_to_cm = 219474.6313705
fs_to_au      = 41.341374576
cm_to_Ha      = 1.0 / Hartree_to_cm

# System Parameters
delta_Eps_cm = 1500.0           
delta_Eps_au = delta_Eps_cm * cm_to_Ha
mu_au        = 3.31356454592    

# Drive parameters
FWHM_fs = 5.0
sigma_fs = FWHM_fs / (2 * np.sqrt(2 * np.log(2))) 
t0_fs = 6.0                                       

# Laser Amplitude Calculation
sigma_au = sigma_fs * fs_to_au
target_envelope_area = np.pi 
E0_au = target_envelope_area / (mu_au * sigma_au * np.sqrt(2 * np.pi))
interaction_max_cm = (mu_au * E0_au) * Hartree_to_cm

# --- 2. Full Laser Function ---
def get_laser_field(t_fs_array):
    t_au = t_fs_array * fs_to_au
    envelope = np.exp(-0.5 * ((t_fs_array - t0_fs) / sigma_fs)**2)
    carrier = np.cos(delta_Eps_au * t_au)
    return -interaction_max_cm * envelope * carrier

# --- 3. Style Settings ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", "font.serif": ["Times"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "axes.linewidth": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 300,
})

# -------------------- DATA PROCESSING --------------------

# Load Data
try:
    data = np.loadtxt("laser_example3.dat.log", dtype=np.complex128)
    t_plot = data[:, 0].real
    pop_exc = data[:, 1].real
    coh_eg = data[:, 2] 
    coh_mag = np.abs(coh_eg)
    pop_gnd = data[:, 4].real
except OSError:
    print("Could not load file.")
    exit()

# Laser Trace
laser_vals = get_laser_field(t_plot)
norm_factor = 0.4 / np.max(np.abs(laser_vals))
laser_plot_visual = laser_vals * norm_factor

# -------------------- PLOTTING --------------------

# Create Figure
# Reduced height from 4.8 to 3.2 since we removed one panel
fig = plt.figure(figsize=(3.4, 3.2))

# Define ONE GridSpec for the two remaining panels
# Adjusted bottom to 0.15 to leave room for x-axis labels
gs = fig.add_gridspec(nrows=2, ncols=1, left=0.15, right=0.95, bottom=0.15, top=0.88, hspace=0.0)

# Create Axes
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1) # Share X axis

# Helper for inside labels with small background to prevent overlap
def add_inside_label(ax, text, x=0.02, y=0.90):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# --- PANEL (a): POPULATIONS ---
l1, = ax1.plot(t_plot, pop_exc, color='tab:blue', label=r"$\rho_{11}$")
l2, = ax1.plot(t_plot, pop_gnd, '--', color='tab:orange', label=r"$\rho_{00}$")
l_las, = ax1.plot(t_plot, laser_plot_visual, 'k-', linewidth=0.5, alpha=0.7, label=r"E(t)")
ax1.fill_between(t_plot, 0, laser_plot_visual, color='k', alpha=0.05)
ax1.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

ax1.set_ylabel("Population", labelpad=2) 
ax1.set_xlim(t_plot[0], t_plot[-1])
# Hide X ticks for the top plot since they share axis
ax1.tick_params(labelbottom=False) 
add_inside_label(ax1, "(a)")

# --- PANEL (b): COHERENCES ---
l3, = ax2.plot(t_plot, coh_mag, color='tab:purple', label=r"$|\rho_{10}|$")

ax2.set_ylabel("Coherence", labelpad=2)
ax2.set_xlabel("Time (fs)", labelpad=2)
ax2.set_xlim(t_plot[0], t_plot[-1])
add_inside_label(ax2, "(b)")

# -------------------- LEGEND --------------------
handles = [l1, l2, l_las, l3]
labels  = [h.get_label() for h in handles]

fig.legend(handles, labels, 
           loc="upper center",
           bbox_to_anchor=(0.55, 0.99), 
           ncol=4, 
           fontsize=8,
           frameon=False, 
           columnspacing=1.0,
           handletextpad=0.4) 

plt.savefig("two_panel_shared_axis.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()
