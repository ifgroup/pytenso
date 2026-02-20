""" Basic sanity checks on the bath generation functions.
"""
# coding: utf-8

import pytest
from tenso.prototypes.bath import gen_bcf


@pytest.mark.parametrize('re_d, width_d', [
    ([], []),
    ([200], [100]),
    ([200], [100]),
])
@pytest.mark.parametrize('freq_b, re_b, width_b', [
    ([], [], []),
    ([1000], [200], [100]),
    ([1000, 500], [200, 100], [10, 10]),
])
@pytest.mark.parametrize('freq_v, re_v', [
    ([], []),
    ([1000], [200]),
    ([1000, 500], [200, 100]),
])
@pytest.mark.parametrize('temperature, decomposition_method, n_ltc', [
    (300, 'Pade', 0),
    (77, 'Pade', 5),
])
def test_bcf_kmax(
             re_d, 
             width_d, 
             freq_b,
             re_b,
             width_b,             
             freq_v,
             re_v,
             temperature, 
             decomposition_method, 
             n_ltc):
    # Bath settings:
    bath = gen_bcf(
        re_d=re_d,
        width_d=width_d,
        freq_b=freq_b,
        re_b=re_b,
        width_b=width_b, 
        freq_v=freq_v,
        re_v=re_v,
        temperature=temperature,
        decomposition_method=decomposition_method,
        n_ltc=n_ltc,
    )

    if  len(re_d) > 0 or len(re_b) > 0: 
        kmax_target = len(re_d) + 2 * len(re_b) + 2 * len(re_v) + n_ltc 
    else:
        kmax_target = 2 * len(re_v)  
    assert bath.k_max == kmax_target


