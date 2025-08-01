# coding: utf-8

from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import spin_boson
from tenso.prototypes.bath import gen_bcf

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    out = os.path.splitext(os.path.basename(__file__))[0]

    # Bath settings:
    bath = gen_bcf(
        re_d=[200],
        width_d=[100],
        temperature=300,
        decomposition_method='Pade',
        n_ltc=5,
    )
    print(bath, flush=True)

    end_time = 1000.0
    dt = 1.0
    wfn = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    propagator = spin_boson(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        sys_ham=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        sys_op=np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
        bath_correlation=bath,
        dim=5,
        end_time=end_time,
        dt=dt,
        save_checkpoint_to_file=True,
    )

    steps = ceil(end_time / dt)
    progress_bar = tqdm(propagator, total=steps)

    for _t in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')
