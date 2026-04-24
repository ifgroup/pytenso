import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

path='two_baths.dat.log'
plt.rcParams["font.family"] = "serif"

def load_spinboson_pops(fname, convert_to_ps=False):
    try:
        arr = np.genfromtxt(fname, dtype=complex, comments="#", skip_header=1)
    except OSError:
        print("Could not load file.")
        exit()
    t = arr[:,0].real
    if convert_to_ps:
        t *= 1e-3
    pops = np.vstack((arr[:,1].real, arr[:,4].real)).T
    return t, pops

time_series, population_series = load_spinboson_pops(path)

fig, (ax1) = plt.subplots(
    1,1, figsize=(3.4, 2.0), gridspec_kw={"hspace":0.9}
)
# --- (a) populations ---
ax1.plot(time_series, population_series[:,0], label=r"$\rho_{00}$")
ax1.plot(time_series, population_series[:,1], label=r"$\rho_{11}$")

ax1.set_ylabel("Population", labelpad=6)
ax1.set_xlabel("Time (fs)", labelpad=6)
ax1.set_xlim(0, 500)

handles, labs = ax1.get_legend_handles_labels()
fig.legend(handles, labs, loc="upper center",
           ncol=2, bbox_to_anchor=(0.5, 1.00), fontsize=8,
           frameon=False)

fig.tight_layout(rect=(0,0,1,0.88))
plt.savefig("SpinBoson_populations.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()
