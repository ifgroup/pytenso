from math import ceil
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath
from tenso.prototypes.bath import gen_bcf

wfn = np.array([1.0, 0.0], dtype=np.complex128)

# ================== (a) DIM CONVERGENCE ========================
# System from convergence_plot.py: DL-only bath, symmetric TLS
bath_dim = gen_bcf(
    re_d=[540],
    width_d=[70],
    temperature=300,
    decomposition_method='Pade',
    n_ltc=1,
)

sys_ham_dim = np.array([[0.0, 300.0], [300.0, 0.0]], dtype=np.complex128)
sys_op_dim  = np.array([[0.0, 1.0],  [1.0,  0.0]], dtype=np.complex128)

end_time_dim = 100.0   # fs
dt_dim       = 0.05    # fs

dims = [2]

print('='*50)
print('Running dim-convergence simulations')
print('='*50)
for ii, dim in enumerate(dims):
    fname = f'convergence_{ii}'
    propagator = system_multibath(
        fname=fname,
        init_rdo=np.outer(wfn, wfn.conj()),
        sys_ham=sys_ham_dim,
        sys_ops=[sys_op_dim],
        bath_correlations=[bath_dim],
        dim=dim,
        end_time=end_time_dim,
        step_time=dt_dim,
    )
    progress_bar = tqdm(propagator, total=ceil(end_time_dim / dt_dim),
                        desc=f'dim={dim}')
    for _t in progress_bar:
        progress_bar.set_description(f'dim={dim} @{_t:.2f} fs')

# ================== (b) RANK CONVERGENCE =======================
# Original system: DL + Brownian bath, asymmetric TLS
bath_rank = gen_bcf(
    re_d=[540],
    width_d=[70],
    freq_b=[1663],
    re_b=[330],
    width_b=[4],
    temperature=300,
    decomposition_method='Pade',
    n_ltc=1,
)

sys_ham_rank = np.array([[1500/2, 300.0], [300.0, -1500/2]], dtype=np.complex128)
sys_op_rank  = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

end_time_rank = 100.0  # fs
dt_rank       = 0.05     # fs

ranks         = [40]
frame_methods = ['train', 'tree2']
dim_rank      = 25   # fixed dim for rank sweep

print('='*50)
print('Running rank-convergence simulations')
print('='*50)
for method in frame_methods:
    for rank in ranks:
        fname = f'{method}_rank{rank}'
        propagator = system_multibath(
            fname=fname,
            init_rdo=np.outer(wfn, wfn.conj()),
            sys_ham=sys_ham_rank,
            sys_ops=[sys_op_rank],
            bath_correlations=[bath_rank],
            dim=dim_rank,
            end_time=end_time_rank,
            step_time=dt_rank,
            frame_method=method,
            rank=rank,
            stepwise_method='simple',
            ps_method='ps1',
        )
        progress_bar = tqdm(propagator, total=ceil(end_time_rank / dt_rank),
                            desc=f'{method} rank={rank}')
        for _t in progress_bar:
            progress_bar.set_description(f'{method} rank={rank} @{_t:.2f} fs')
