import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Parameters ───────────────────────────────────────────────────────
data_dir_new = Path('.')   # update if files are in a subdirectory
dims_new     = [2, 4, 6, 10, 14, 25]
ranks_new    = [5, 10, 15, 20, 25, 40]
time_units   = 'fs'

# ── Load data ────────────────────────────────────────────────────────
def load_new(fname):
    p = data_dir_new / fname
    if not p.exists():
        print(f'WARNING: {p} not found')
        return None
    arr = np.genfromtxt(p, dtype=complex, comments='#', skip_header=1)
    return arr[:, 0].real, arr[:, 1].real

dim_data_new        = [load_new(f'convergence_{i}.dat.log')  for i in range(len(dims_new))]
rank_data_tree_new  = [load_new(f'tree2_rank{r}.dat.log')    for r in ranks_new]
rank_data_train_new = [load_new(f'train_rank{r}.dat.log')    for r in ranks_new]

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", "font.serif": ["Times"],
    "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11, "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "xtick.minor.size": 2,
    "xtick.major.width": 0.8, "xtick.minor.width": 0.6,
    "ytick.major.size": 3.5, "ytick.minor.size": 2,
    "ytick.major.width": 0.8, "ytick.minor.width": 0.6,
    "legend.frameon": False, "legend.fontsize": 10, "figure.dpi": 300,
})

cmap = cm.coolwarm

def panel_label(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))

def mean_err(rho, rho_ref):
    n = min(len(rho), len(rho_ref))
    return np.mean(np.abs(rho[:n] - rho_ref[:n]))

colors_dim_all  = [cmap(i / (len(dims_new)  - 1)) for i in range(len(dims_new))]
colors_rank_all = [cmap(i / (len(ranks_new) - 1)) for i in range(len(ranks_new))]

t_ref_dim,   rho_ref_dim   = dim_data_new[-1]
t_ref_tree,  rho_ref_tree  = rank_data_tree_new[-1]
t_ref_train, rho_ref_train = rank_data_train_new[-1]

# ── Layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(3.5, 7.2))
gs_top = fig.add_gridspec(3, 1, top=0.96, bottom=0.36, hspace=0.0)
gs_bot = fig.add_gridspec(1, 1, top=0.29, bottom=0.07)

ax_a = fig.add_subplot(gs_top[0])
ax_b = fig.add_subplot(gs_top[1], sharex=ax_a)
ax_c = fig.add_subplot(gs_top[2], sharex=ax_a)
ax_d = fig.add_subplot(gs_bot[0])

# ── (a) dim convergence ───────────────────────────────────────────────
for color, dim, entry in zip(colors_dim_all[:-1], dims_new[:-1], dim_data_new[:-1]):
    if entry is None: continue
    ax_a.plot(entry[0], entry[1], color=color, label=f'Depth = {dim}')
ax_a.plot(t_ref_dim, rho_ref_dim, color=colors_dim_all[-1], lw=2.0,
          label=f'Depth = {dims_new[-1]} (ref)')
ax_a.set_ylabel('Population', labelpad=6)
ax_a.set_xlim(0, 100)
plt.setp(ax_a.get_xticklabels(), visible=False)
ax_a.tick_params(bottom=False)
panel_label(ax_a, '(a)')
ax_a.legend(loc='upper right', ncol=2, fontsize=8)

# ── (b) tree2 rank convergence ────────────────────────────────────────
for color, rank, entry in zip(colors_rank_all[:-1], ranks_new[:-1], rank_data_tree_new[:-1]):
    if entry is None: continue
    ax_b.plot(entry[0], entry[1], color=color, label=f'Rank = {rank}')
ax_b.plot(t_ref_tree, rho_ref_tree, color=colors_rank_all[-1], lw=2.0,
          label=f'Rank = {ranks_new[-1]} (ref)')
ax_b.set_ylabel('Population', labelpad=6)
ax_b.set_xlim(0, 100)
plt.setp(ax_b.get_xticklabels(), visible=False)
ax_b.tick_params(bottom=False)
panel_label(ax_b, '(b) BTT')
ax_b.legend(loc='upper right', ncol=2, fontsize=8)

# ── (c) train rank convergence ────────────────────────────────────────
for color, rank, entry in zip(colors_rank_all[:-1], ranks_new[:-1], rank_data_train_new[:-1]):
    if entry is None: continue
    ax_c.plot(entry[0], entry[1], color=color, label=f'Rank = {rank}')
ax_c.plot(t_ref_train, rho_ref_train, color=colors_rank_all[-1], lw=2.0,
          label=f'Rank = {ranks_new[-1]} (ref)')
ax_c.set_xlabel('Time (fs)', labelpad=4)
ax_c.set_ylabel('Population', labelpad=6)
ax_c.set_xlim(0, 100)
panel_label(ax_c, '(c) TT')

# ── (d) time-averaged |Δρ₀₀| vs parameter value ──────────────────────
dim_errors   = [mean_err(e[1], rho_ref_dim)   if e else np.nan for e in dim_data_new[:-1]]
tree_errors  = [mean_err(e[1], rho_ref_tree)  if e else np.nan for e in rank_data_tree_new[:-1]]
train_errors = [mean_err(e[1], rho_ref_train) if e else np.nan for e in rank_data_train_new[:-1]]

ax_d.semilogy(dims_new[:-1],  dim_errors,   'o-', color='C0', label='Depth')
ax_d.semilogy(ranks_new[:-1], tree_errors,  's-', color='C1', label='Rank (BTT)')
ax_d.semilogy(ranks_new[:-1], train_errors, '^-', color='C2', label='Rank (TT)')

ax_d.set_xlabel('Parameter value', labelpad=4)
ax_d.set_ylabel(r'$\langle|\Delta\rho_{11}|\rangle_t$', labelpad=6)
ax_d.set_xlim(0, 35)
panel_label(ax_d, '(d)')
ax_d.legend(loc='upper right')

plt.savefig('convergence_new_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()
