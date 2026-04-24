from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath
from tenso.prototypes.bath import gen_bcf
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path
import re


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    out = os.path.splitext(os.path.basename(__file__))[0] 

# Plotting script
# ===================== FILE LISTS =====================
TEMP_FILES = [
    r"T_50K.dat.log",
    r"T_100K.dat.log",
    r"T_200K.dat.log",
    r"T_300K.dat.log",
    r"T_400K.dat.log",
]
REORG_FILES = [
    r"Reorg_5.dat.log",
    r"Reorg_15.dat.log",
    r"Reorg_30.dat.log",
    r"Reorg_50.dat.log",
    r"Reorg_70.dat.log",
    r"Reorg_90.dat.log",
    r"Reorg_120.dat.log",
]
# ===================== SETTINGS =====================
time_units = "fs"        # ps or fs
skip_header = 1
YMAX = 1.0
OUT_BASENAME = "concurrence"
USE_TEX = False          
# Colormaps
CMAP_W, CMAP_T, CMAP_L, CMAP_F = "plasma", "cividis", "magma", "viridis"
# Límits X
XMAX_ABC, XMAX_D = 150.0, 50.0
# Inset (panel d)
INSET_X, INSET_Y = (20, 25), (0.2, 0.4)

# ====== Nature Physics–ish style ======
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.family": "serif", "font.serif": ["Times"], "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "lines.linewidth": 1.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "figure.dpi": 300,
    })
else:
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "lines.linewidth": 1.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "figure.dpi": 300,
    })

# ====== Helpers ======
def safe_savefig(fig, basename):
    try:
        fig.savefig(f"{basename}.pdf", bbox_inches="constrained")
        fig.savefig(f"{basename}.png", dpi=450, bbox_inches="constrained")
    except Exception as e:
        print("[warn] PDF failed, retrying without TeX:", e)
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "stix"})
        fig.savefig(f"{basename}.pdf", bbox_inches="constrained")
        fig.savefig(f"{basename}.png", dpi=450, bbox_inches="constrained")

def load_rho_series(fname: Path, convert_to_ps=False):
    arr = np.genfromtxt(fname, dtype=complex, comments="#", skip_header=skip_header)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    t = arr[:, 0].real
    if convert_to_ps: t *= 1e-3
    data = arr[:, 1:]
    M = data.shape[1]; N = int(np.sqrt(M))
    if N*N != M: raise ValueError(f"{fname}: got {M} cols after time, not N^2.")
    rho = data.reshape(-1, N, N)
    return t, rho, N

def concurrence_wotters(rho4):
    sy = np.array([[0, -1j], [1j, 0]]); Y = np.kron(sy, sy)
    R = rho4 @ Y @ rho4.conjugate() @ Y
    evals = np.linalg.eigvals(R)
    evals = np.real(np.clip(evals, 0, None))
    lam = np.sort(np.sqrt(evals))[::-1]
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))

def concurrence_series(rho):
    if rho.shape[1:] != (4, 4): return None
    C = np.empty(rho.shape[0])
    for i in range(rho.shape[0]): C[i] = concurrence_wotters(rho[i])
    return np.clip(C, 0, None)

def _basename_no_dat(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".log"): name = name[:-4]
    if name.lower().endswith(".dat"): name = name[:-4]
    return name.replace("Brwonian", "Brownian")


def extract_num_for_panel(p: Path, key: str):
    name = _basename_no_dat(p)
    if key == "T":
        m = re.match(r"T_(\d+)K$", name, re.I)
    elif key == "L":
        m = re.match(r"Reorg_(\d+)$", name, re.I)
    return None if not m else float(m.group(1))


# ====== Figure ======
PANELS = [
    {"title": "Temperature sweep",  "files": TEMP_FILES,    "letter": "b", "key": "T", "cmap": CMAP_T, "cbar": r"$T\ (\mathrm{K})$"},
    {"title": "Reorganization",     "files": REORG_FILES,   "letter": "c", "key": "L", "cmap": CMAP_L, "cbar": r"$\lambda\ (\mathrm{cm}^{-1})$"},
]

fig, axes = plt.subplots(3, 1, figsize=(4.4, 6.3),
                         gridspec_kw={"hspace": 0.3, "wspace": 0.36},
                         layout="tight")
axes = axes.ravel()
convert = (time_units.lower() == "ps")

t, rho, N = populations = load_rho_series('population_example.dat.log')

ax = axes[0]
ax.plot(t,rho[:,0,0],label=r"$|11\rangle$")
ax.plot(t,rho[:,1,1],label=r"$|01\rangle$")
ax.plot(t,rho[:,2,2],dashes=(3,3),label=r"$|10\rangle$")
ax.plot(t,rho[:,3,3],label=r"$|00\rangle$")
ax.set_xlim(0,150) # Hard-coded xmax in this case
ax.legend(loc='best',bbox_to_anchor=(0.05,0.95,0.9,0.1),ncols=4)
ax.set_ylabel(f"Population", labelpad=7)
ax.text(-0.16, 1.06, f"(a)", transform=ax.transAxes,
            fontsize=9, va="bottom")


for ax, panel, xmax in zip(axes[1:3], PANELS[:2], [XMAX_ABC]*2):
    print(ax)
    files = [Path(f) for f in panel["files"]]
    vals = [extract_num_for_panel(p, panel["key"]) for p in files]
    vals = [v for v in vals if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(panel["cmap"])

    for p in files:
        v = extract_num_for_panel(p, panel["key"])
        if v is None: continue
        try:
            t, rho, N = load_rho_series(p, convert_to_ps=convert)
        except Exception as e:
            print(f"[skip] {p}: {e}"); continue
        if N != 4: print(f"[skip] {p}: N={N}"); continue
        C = concurrence_series(rho);  ax.plot(t, C, color=cmap(norm(v)), alpha=0.98)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label(panel["cbar"], rotation=90, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_ylabel("Concurrence", labelpad=7)
    if YMAX is not None: ax.set_ylim(0, YMAX)
    ax.set_xlim(0, xmax)
    ax.margins(x=0.02)
    ax.text(-0.16, 1.06, f"({panel['letter']})", transform=ax.transAxes,
            fontsize=9, va="bottom")

ax.set_xlabel(f"Time ({time_units})", labelpad=7)
plt.savefig(OUT_BASENAME, dpi=450, transparent=True)
plt.show()
