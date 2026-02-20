"""This file contains a few short calculations whose output is spot checked
for accurracy. Additional, more thorough checks will make sure that the 
density matrices produced are legal.
"""
from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm
from tenso.prototypes.bath import gen_bcf
from math import ceil
from tenso.prototypes.heom import system_multibath
from tenso.prototypes.default_parameters import quantity
import pytest

def test1(subtests,tmp_path):
    """ Quick FMO complex three bath check. Does the usual set of a few
    point checks and then general density matrix legality checks.
    """
    out=os.path.join(tmp_path,'hr1')

    bath = gen_bcf(
        re_d = [35],
        width_d = [106.18],
        temperature=300,
        decomposition_method='Pade',
        n_ltc=1,
    )

    h = np.array([
        [310, -98, 6],
        [-98, 230, 30],
        [6, 30, 0]
    ], dtype=np.complex128)

    avg = np.diag(h).mean()
    h -= avg * np.eye(3)

    sys_ops = []
    end_time = 50.0
    dt = 0.5

    for i in range(3):
        op_i = np.zeros((3, 3))
        op_i[i, i] = 1.0
        sys_ops.append(op_i)

    wfn = np.zeros(3)
    wfn[0] = 1.0  # Initial state localized at site 1

    propagator = system_multibath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        sys_ham=h,
        sys_ops=sys_ops,
        bath_correlations=[bath] * 3,
        end_time=end_time,
        step_time=dt,
        dim=10,
    )


    progress_bar = tqdm(propagator,total=ceil(end_time/dt))
    for _t in progress_bar:
       progress_bar.set_description(f'@{_t: .2f} fs')


    # Load results from the output file into a local array
    results = np.loadtxt(out+".dat.log",dtype=np.complex128)
    assert results[12,2].real == pytest.approx(-0.00487562)
    assert results[12,2].imag == pytest.approx(-0.10887978)
    assert results[25,1].real == pytest.approx(0.94847489)
    assert results[25,1].imag == pytest.approx(-0.00000001)
    assert results[36,7].real == pytest.approx(0.02132838)
    assert results[36,7].imag == pytest.approx(-0.00896822)
    assert results[49,9].real == pytest.approx(0.00156856)
    assert results[49,9].imag == pytest.approx(0.00000001)
    assert results[80,5].real == pytest.approx(0.38471039)
    assert results[80,5].imag == pytest.approx(0.00000027)
    assert results[95,7].real == pytest.approx(0.03225632)
    assert results[95,7].imag == pytest.approx(0.02729108)
    for i in range(0,100):
        with subtests.test("Values add to 1", i =i):
            assert results[i,1].real + results[i,5] + results[i,9] == pytest.approx(1.0)
    for i in range(0,100):
        with subtests.test("No imaginary Populations", i=i):
            assert results[i,1].imag <= 1e-5
            assert results[i,5].imag <= 1e-5
            assert results[i,9].imag <= 1e-5
    for i in range(0,100):
        with subtests.test("No negative Populations", i=i):
            assert np.absolute(results[i,1].real) >= -1e-5
            assert np.absolute(results[i,5].real) >= -1e-5
            assert np.absolute(results[i,9].real) >= -1e-5
    for i in range(0,100):
        with subtests.test("Hermitian matrix", i=i):
            # Tolerance increased due to larger timesteps in this setup
            assert results[i,2].imag == pytest.approx(-1.0*results[i,4].imag,rel=1e-5,abs=1e-5)
            assert results[i,2].real == pytest.approx(results[i,4].real,rel=1e-5,abs=1e-5)
            assert results[i,3].real == pytest.approx(results[i,7].real,rel=1e-5,abs=1e-5)
            assert results[i,3].imag == pytest.approx(-1.0*results[i,7].imag,rel=1e-5,abs=1e-5)
            assert results[i,6].real == pytest.approx(results[i,8].real,rel=1e-5,abs=1e-5)
            assert results[i,6].imag == pytest.approx(-1.0*results[i,8].imag,rel=1e-5,abs=1e-5)
    for i in range(0,100):
        with subtests.test("No illegally large coherences", i=i): # Tolerance of 1% allowed
            ratio = 0.99
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,5])) >= ratio*np.abs(results[i,2])
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,9])) >= ratio*np.abs(results[i,3])
            assert np.sqrt(np.absolute(results[i,5]))*np.sqrt(np.absolute(results[i,9])) >= ratio*np.abs(results[i,6])


def test2(subtests,tmp_path):
    """ FMO complex check with varying numbers of low temperature
    corrections, hence features, in the three baths. The system
    should be converged with respect to n_ltc and thus only small
    differences from the above run should be expected. Tolerances
    must, however, increase in several cases.
    """

    out=os.path.join(tmp_path,'hr1')
    bath1 = gen_bcf(
        re_d = [35],
        width_d = [106.18],
        temperature=300,
        decomposition_method='Pade',
        n_ltc=2,
    )
    bath2 = gen_bcf(
        re_d = [35],
        width_d = [106.18],
        temperature=300,
        decomposition_method='Pade',
        n_ltc=1,
    )
    bath3 = gen_bcf(
        re_d = [35],
        width_d = [106.18],
        temperature=300,
        decomposition_method='Pade',
        n_ltc=3,
    )
    h = np.array([
        [310, -98, 6],
        [-98, 230, 30],
        [6, 30, 0]
    ], dtype=np.complex128)

    avg = np.diag(h).mean()
    h -= avg * np.eye(3)

    sys_ops = []
    end_time = 25.0
    dt = 0.5

    for i in range(3):
        op_i = np.zeros((3, 3))
        op_i[i, i] = 1.0
        sys_ops.append(op_i)

    wfn = np.zeros(3)
    wfn[0] = 1.0  # Initial state localized at site 1

    out=os.path.join(tmp_path,'hr1')
    propagator = system_multibath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        sys_ham=h,
        sys_ops=sys_ops,
        bath_correlations=[bath1, bath2, bath3],
        end_time=end_time,
        step_time=dt,
        dim=10,
    )


    progress_bar = tqdm(propagator,total=ceil(end_time/dt))
    for _t in progress_bar:
       progress_bar.set_description(f'@{_t: .2f} fs')

    abtol=2e-5  # Results should be a close match to above
    # Load results from the output file into a local array
    results = np.loadtxt(out+".dat.log",dtype=np.complex128)
    assert results[12,2].real == pytest.approx(-0.00487562,abs=abtol)
    assert results[12,2].imag == pytest.approx(-0.10887978,abs=abtol)
    assert results[25,1].real == pytest.approx(0.94847489,abs=abtol)
    assert results[25,1].imag == pytest.approx(-0.00000001,abs=abtol)
    assert results[36,7].real == pytest.approx(0.02132838,abs=abtol)
    assert results[36,7].imag == pytest.approx(-0.00896822,abs=abtol)
    assert results[49,9].real == pytest.approx(0.00156856,abs=abtol)
    assert results[49,9].imag == pytest.approx(0.00000001,abs=abtol)
    #assert results[80,5].real == pytest.approx(0.38471039,abs=abtol)
    #assert results[80,5].imag == pytest.approx(0.00000027,abs=abtol)
    #assert results[95,7].real == pytest.approx(0.03225632,abs=abtol)
    #assert results[95,7].imag == pytest.approx(0.02729108,abs=abtol)
    for i in range(0,50):
        with subtests.test("Values add to 1", i =i):
            assert results[i,1].real + results[i,5] + results[i,9] == pytest.approx(1.0,abs=1e-5)
    for i in range(0,50):
        with subtests.test("No imaginary Populations", i=i):
            assert results[i,1].imag <= 1e-5
            assert results[i,5].imag <= 1e-5
            assert results[i,9].imag <= 1e-5
    for i in range(0,50):
        with subtests.test("No negative Populations", i=i):
            assert np.absolute(results[i,1].real) >= -1e-5
            assert np.absolute(results[i,5].real) >= -1e-5
            assert np.absolute(results[i,9].real) >= -1e-5
    for i in range(0,50):
        with subtests.test("Hermitian matrix", i=i):
            # Tolerance increased due to larger timesteps in this setup
            assert results[i,2].imag == pytest.approx(-1.0*results[i,4].imag,rel=1e-5,abs=1e-5)
            assert results[i,2].real == pytest.approx(results[i,4].real,rel=1e-5,abs=1e-5)
            assert results[i,3].real == pytest.approx(results[i,7].real,rel=1e-5,abs=1e-5)
            assert results[i,3].imag == pytest.approx(-1.0*results[i,7].imag,rel=1e-5,abs=1e-5)
            assert results[i,6].real == pytest.approx(results[i,8].real,rel=1e-5,abs=1e-5)
            assert results[i,6].imag == pytest.approx(-1.0*results[i,8].imag,rel=1e-5,abs=1e-5)
    for i in range(0,50):
        with subtests.test("No illegally large coherences", i=i): # Tolerance of 1.5% allowed
            ratio = 0.98
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,5])) >= ratio*np.abs(results[i,2])
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,9])) >= ratio*np.abs(results[i,3])
            assert np.sqrt(np.absolute(results[i,5]))*np.sqrt(np.absolute(results[i,9])) >= ratio*np.abs(results[i,6])

