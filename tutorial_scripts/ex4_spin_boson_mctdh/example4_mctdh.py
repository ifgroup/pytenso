# coding: utf-8
from math import ceil
import numpy as np
from tqdm import tqdm
# Import MCTDH module rather than HEOM modules
from tenso.prototypes.mctdh import system_multibath
from tenso.prototypes.bath import gen_star_boson

out = "mctdh"
# Generate the type of bath discretization required for MCTDH
bath_simulation = gen_star_boson(
    re_b=[200], 
    width_b=[10], 
    freq_b=[50],
    temperature=300, 
    cutoff=200, # Maximum frequency in the bath discretization
    n_discretization=50, # Number of discrete modes for the bath
    discretization_method='TEDOPA',
)

end_time = 200
dt = 1 
wfn = np.array([0.0, 1.0], dtype=np.complex128) # Initial wavefunction

# The propagator requires an initial wavefunction, not density matrix
propagator = system_multibath( 
    fname=out, 
    init_wfn=wfn, # Set system initial condition
    sys_ham=np.array([[-750.0,300.0],[300.0,750.0]], dtype=np.complex128), 
    sys_ops=[np.array([[-0.5,0.0], [0.0,0.5]], dtype=np.complex128)], 
    baths=[bath_simulation], # Note the different name for the bath input
    end_time=end_time, 
    dim=8,
    step_time=dt,) 
progress_bar = tqdm(propagator,total=ceil(end_time/dt))
for _t in progress_bar:
    progress_bar.set_description(f'@{_t: .2f} fs')
    


