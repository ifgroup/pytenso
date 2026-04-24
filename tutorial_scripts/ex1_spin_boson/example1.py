from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath # Propagator
from tenso.prototypes.bath import gen_bcf # Bath correlation function generator
import matplotlib.pyplot as plt

out_tree = 'tree2' # Output directory
out_name = 'train' # Output name

outs = {
    'tree2': out_tree,
    'train': out_name
}

bath_simulation = gen_bcf(
    re_d=[540],    # Reorganization energy in cm-1
    width_d=[70],  # Width of the DL spectral density in cm-1
    freq_b=[1243], # Central Frequency of the Brownian Spectral density in cm-1
    re_b=[161.6], # Reorganization energy of the Brownian Spectral density in cm-1
    width_b=[10],  # Width of the spectral density in cm-1
    temperature=300, # Temperature in Kelvin
    decomposition_method='Pade', # Decomposition method for the bath correlation function
    n_ltc=1, # Number of low-temperature correction terms
)
# Initialize system Hamiltonian
H = np.array([[1500/2, 600/2], [600/2, -1500/2]], dtype=np.complex128)

end_time = 1000.0 # End time in fs
dt = 1
wfn = np.array([1.0, 0.0], dtype=np.complex128)
# Run calculation s for the tree and train configuation
for out in outs:
    propagator = system_multibath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        sys_ham=np.array([[1500/2, 600/2], [600/2, -1500/2]], dtype=np.complex128),
        sys_ops=[np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)],
        bath_correlations=[bath_simulation],
        dim=20,
        end_time=end_time,
        step_time=dt,
        frame_method=out,
    )
    progress_bar = tqdm(propagator, total=ceil(end_time / dt))

    for _t in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')
