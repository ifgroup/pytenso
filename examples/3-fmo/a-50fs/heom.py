# coding: utf-8

from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath
from tenso.prototypes.bath import gen_bcf

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    out = os.path.splitext(os.path.basename(__file__))[0]

    bath = gen_bcf(
        re_d=[35.0],
        width_d=[106.17674918],
        temperature=77,
        decomposition_method='Pade',
        n_ltc=5,
    )
    h = np.array([[200, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
                  [-87.7, 320, 30.8, 8.2, 0.7, 11.8, 4.3],
                  [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6],
                  [-5.9, 8.2, -53.5, 110, -70.7, -17, -63.3],
                  [6.7, 0.7, -2.2, -70.7, 270, 71.1, -1.3],
                  [-13.7, 11.8, -9.6, -17, 71.1, 420, 39.7],
                  [-9.9, 4.3, 6, -63.3, -1.3, 39.7, 230]],
                 dtype=np.complex128)
    avg = np.diag(h).mean()
    h -= avg * np.eye(7)

    # the Hamiltonian of the FMO complex
    sys_ops = []
    end_time = 1000.0
    dt = 0.1
    for i in range(7):
        op_i = np.zeros((7, 7))
        op_i[i, i] = 1.0
        sys_ops.append(op_i)
    wfn = np.zeros(7)
    wfn[0] = 1.0
    propagator = system_multibath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        sys_ham=h,
        sys_ops=sys_ops,
        bath_correlations=[bath] * 7,
        end_time=end_time,
        step_time=dt,
    )

    progress_bar = tqdm(propagator, total=ceil(end_time / dt))
    for _t in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')
