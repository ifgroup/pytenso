import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

reference='heom.dat.log'
mctdh='mctdh.dat.log'

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

th, seriesh = load_spinboson_pops(reference)
tm, seriesm = load_spinboson_pops(mctdh)

fig, (ax1) = plt.subplots(
    1,1, figsize=(3.4, 2.0), gridspec_kw={"hspace":0.9}
)
ax1.plot(th, seriesh[:,0], label=r"HEOM")
ax1.plot(tm, seriesm[:,0], label=r"MCTDH",dashes=(2,2))

ax1.set_ylabel("Population", labelpad=6)
ax1.set_xlabel("Time (fs)", labelpad=6)
ax1.set_xlim(0, 200)

handles, labs = ax1.get_legend_handles_labels()
fig.legend(handles, labs, loc="upper center",
           ncol=2, bbox_to_anchor=(0.5, 1.00), fontsize=8,
           frameon=False)

fig.tight_layout(rect=(0,0,1,0.88))
plt.savefig("heom_v_mctdh.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()
