# coding: utf-8

from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.mctdh import spin_boson
from tenso.prototypes.bath import gen_star_boson

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    out = os.path.splitext(os.path.basename(__file__))[0]

    # Bath settings:
    bath = gen_star_boson(
        re_d=[715.73],
        width_d=[54.45],
        freq_v=[1663, 1416, 1376, 1243, 1193, 784, 665, 442],
        re_v=[330.0, 25.6, 186.0, 161.7, 77.3, 26.5, 32.0],
        temperature=300,
        cutoff=100,
        n_discretization=16,
    )
    print(bath, flush=True)

    # System settings:
    wfn = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    end_time = 400.0
    dt = 1.0
    propagator = spin_boson(
        fname=out,
        init_wfn=wfn,
        sys_ham=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        sys_op=np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
        bath=bath,
        save_checkpoint_to_file=True,
        dim=5,
        step_time=dt,
    )

    progress_bar = tqdm(propagator, total=ceil(end_time / dt))
    for _t in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')
