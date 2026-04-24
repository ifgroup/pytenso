from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath
from tenso.prototypes.bath import gen_bcf
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path
import re


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    out = os.path.splitext(os.path.basename(__file__))[0] 
dt=0.5
end_time=150



sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I = np.eye(2, dtype=np.complex128)
# Define parameters
omega = 20  
# Define single qubit Hamiltonian
H1 = (omega / 2) * sigma_z + omega/2 * I
# Construct two-qubit Hamiltonian without interaction
h = np.kron(H1, I) + np.kron(I, H1)

Q_s1=np.kron(sigma_x,I)
Q_s2=np.kron(I,sigma_x)

sys_ops = [Q_s1,Q_s2]

rdo_0 = np.array([[0, 0, 0, 0],
                  [0, 0.5, -0.5, 0],
                  [0, -0.5, 0.5,0],
                  [0, 0, 0,0]],
                 dtype=np.complex128) 

bath = gen_bcf(
    re_d=[50], 
    width_d=[30],
    temperature=50,
    decomposition_method='Pade',
    n_ltc=3,
)


out="population_example"
propagator = system_multibath(
        fname=out,
        init_rdo=rdo_0,
        sys_ham=h,
        sys_ops=sys_ops,
        bath_correlations=[bath]*2,
        end_time=end_time,
        step_time=dt,
        dim=20,
)

progress_bar = tqdm(propagator, total=ceil(end_time / dt)) 
for _t in (progress_bar):
    progress_bar.set_description(f'@{_t:.2f} fs')


RR=[5,15,30,50,70,90,120]
for R in RR:
    print("R_%d begins" % R)

    bath = gen_bcf(
        re_d=[R], 
        width_d=[30],
        temperature=300,
        decomposition_method='Pade',
        n_ltc=1,
    )

    out = "Reorg_%d" % R
    propagator = system_multibath( 
        fname=out,
        init_rdo=rdo_0,
        sys_ham=h,
        sys_ops=sys_ops,
        bath_correlations=[bath] * 2, 
        end_time=end_time,
        step_time=dt,
        dim=20
    )

    progress_bar = tqdm(propagator, total=ceil(end_time / dt)) 
    for _t in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')

    print("R_%d is completed" % R)

TT=[50,100,200,300,400]
for Temp in TT:
    print("T_%d begins" % Temp)

    bath = gen_bcf(
        re_d=[50], 
        width_d=[30],
        temperature=Temp,
        decomposition_method='Pade',
        n_ltc=3,
    )

    out = "T_%dK" % Temp
    propagator = system_multibath( 
        fname=out,
        init_rdo=rdo_0,
        sys_ham=h,
        sys_ops=sys_ops,
        bath_correlations=[bath] * 2, 
        end_time=end_time,
        step_time=dt,
        dim=20
    )

    progress_bar = tqdm(propagator, total=ceil(end_time / dt)) 
    for _t in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')

    print("T_%d is completed" % Temp)

