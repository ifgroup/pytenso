#!/usr/bin/env python
# coding: utf-8
"""
Correlation function object for specific spectral density
"""
from __future__ import annotations
from typing import Literal

import numpy as np

from tenso.bath.correlation import Correlation
from tenso.bath.distribution import BoseEinstein
from tenso.bath.sd import CriticallyDampedBrownian, OverdampedBrownian, BrownianOscillator, SuperCriticalDamping
from numpy.typing import NDArray
from tenso.libs.quantity import Quantity as __
from tenso.prototypes.default_parameters import default_units, quantity

PI = np.pi


def brownian_oscillator_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    freq_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
    use_ht_function: bool = True,
) -> Correlation:
    """
    Factory function for a Brownian oscillator spectral density 
    with Ikeda's BCF basis:
        phi_1(t) = g/w1 * e^(-g t) * sin(w1 t) + e^(-g t) * cos(w1 t)
        phi_2(t) = -w0/w1 * e^(-g t) * sin(w1 t)
    """
    if re_b and width_b and freq_b:
        assert len(re_b) == len(width_b) == len(freq_b)
        sds = [
            BrownianOscillator(quantity(re, 'energy'),
                               quantity(width, 'energy'),
                               quantity(freq, 'energy'))
            for re, width, freq in zip(re_b, width_b, freq_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    for sd in sds:
        k_head = corr.k_max
        g = sd.gamma
        l = sd.lambda_
        w0 = sd.omega0
        w1 = sd.omega1
        print(f'k_head={k_head}, g={g}, l={l}, w0={w0}, w1={w1}', flush=True)
        # coefficient associated with phi_1(t):
        # c1 = 2.0 * l / beta
        # coefficient associated with e^(-g * t):
        # c2 = 1.0j * l * w0
        # Note: 2/beta approximates pole*cot(beta*pole/2) at high temperatures
        if not use_ht_function:
            raise NotImplementedError
            tmp = g / np.tan(beta * g / 2.0)
        else:
            tmp = 2.0 / beta
        c1 = 2.0 * l * tmp
        c2 = 1.0j * l * w0
        # coefficient associated with e^(-g * t)
        corr.coefficients.extend([c1, c2])
        corr.conj_coefficents.extend([c1.conjugate(), c2.conjugate()])
        corr.zeropoints.extend([1.0, 0.0])
        corr.derivatives[k_head, k_head + 1] = -w0
        corr.derivatives[k_head + 1, k_head] = w0
        corr.derivatives[k_head + 1, k_head + 1] = -2.0 * g

    corr._add_ltc(sds, distribution)  # type: ignore
    return corr


def critically_damped_brownian_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
    use_ht_function: bool = False,
) -> Correlation:
    """
    Factory function for a critically damped Brownian spectral density.

    """
    if re_b and width_b:
        assert len(re_b) == len(
            width_b), "re_b and width_b must have the same length."
        sds = [
            CriticallyDampedBrownian(quantity(re, 'energy'),
                                     quantity(width, 'energy'))
            for re, width in zip(re_b, width_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    for sd in sds:
        k_head = corr.k_max
        g = sd.gamma
        l = sd.lambda_
        # coefficient associated with (g t) e^(-g * t):
        # c2 = 2.0 * l / beta - 1.0j * l * g
        # coefficient associated with e^(-g * t):
        # c1 = 2.0 * l / beta
        # Note: 2/beta approximates g*cot(beta*g/2) at high temperatures
        if not use_ht_function:
            tmp = 0.5 * g / np.tan(beta * g / 2.0)
        else:
            tmp = 1.0 / beta
        c2 = 2.0 * l * tmp - 1.0j * l * g
        c1 = 2.0 * l * tmp
        # coefficient associated with e^(-g * t)
        corr.coefficients.extend([c1, c2])
        corr.conj_coefficents.extend([c1.conjugate(), c2.conjugate()])
        corr.zeropoints.extend([1.0, 0.0])
        print(f'k_head={k_head}, g={g}, l={l}', flush=True)
        corr.derivatives[k_head, k_head] = -g
        corr.derivatives[k_head + 1, k_head + 1] = -g
        corr.derivatives[k_head + 1, k_head] = g

    corr._add_ltc(sds, distribution)  # type: ignore
    return corr


def super_critical_damping_3rd_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
    use_ht_function: bool = False,
) -> Correlation:
    r"""
    Factory function for a Super Critically damped spectral density of order-3.
    At high temperatures, the BCF (t>0) reads:
        C(t) = (2 l)/(3 beta) ( (g t)^2 + 3 g t + 3 ) e^(-g t)
               - i (l g)/(3) ( (g t)^2 + g t ) e^(-g t)

    Let phi_3(t) = 0.5 * g^2 t^2 e^(-g t), phi_2(t) = -g t e^(-g t), phi_1(t) = e^(-g t),
    then
        d/dt phi_3(t) = -g phi_3(t) - g phi_2(t)
        d/dt phi_2(t) = -g phi_2(t) - g phi_1(t)
        d/dt phi_1(t) = -g phi_1(t)
    and
        C(t) = c1 phi_1(t) + c2 phi_2(t) + c3 phi_3(t)
    where
        c1 = (2 l)/(b) 
        c2 = - (2 l)/(b) + i (l g)/(3)
        c3 = (4 l)/(3 b) - i (2 l g)/(3)
    """
    if re_b and width_b:
        assert len(re_b) == len(
            width_b), "re_b and width_b must have the same length."
        sds = [
            SuperCriticalDamping(quantity(re, 'energy'),
                                 quantity(width, 'energy'),
                                 order=3) for re, width in zip(re_b, width_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    for sd in sds:
        k_head = corr.k_max
        g = sd.gamma
        l = sd.lambda_

        # at high temperatures:
        # c1 = (2 l)/(b)
        # c2 = - (2 l)/(b) + i (l g)/(3)
        # c3 = (4 l)/(3 b) - i (2 l g)/(3)
        # Note: 1 / beta approximates g*cot(beta*g/2)/2 at high temperatures
        if not use_ht_function:
            tmp = 0.5 * g / np.tan(beta * g / 2.0)
        else:
            tmp = 1.0 / beta
        c1 = 2.0 * l * tmp
        c2 = 2.0 * l * tmp - 1.0j * l * g / 3.0
        c3 = 4.0 * l * tmp / 3.0 - 2.0j * l * g / 3.0
        # coefficient associated with e^(-g * t)
        corr.coefficients.extend([c1, c2, c3])
        corr.conj_coefficents.extend(
            [c1.conjugate(), c2.conjugate(),
             c3.conjugate()])
        corr.zeropoints.extend([1.0, 0.0, 0.0])
        print(f'k_head={k_head}, g={g}, l={l}', flush=True)
        corr.derivatives[k_head, k_head] = -g
        corr.derivatives[k_head + 1, k_head + 1] = -g
        corr.derivatives[k_head + 2, k_head + 2] = -g
        corr.derivatives[k_head + 1, k_head] = g
        corr.derivatives[k_head + 2, k_head + 1] = g
    corr._add_ltc(sds, distribution)  # type: ignore
    return corr


def super_critical_damping_4th_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
    use_ht_function: bool = False,
) -> Correlation:
    r"""
    Factory function for a Super Critically damped spectral density of order-4.
    At high temperatures, the BCF (t>0) reads:
        C(t) =     \lambda \frac{2(gt)^3 + 6 (gt)^2 + 12 gt + 12}{15 \beta} e^{-gt}
               - i \lambda \frac{g (gt)^3}{15}} e^{-g t}

    Let phi_4(t) = -(1/6) * g^3 t^3 e^(-g t), phi_3(t) = (1/2) * g^2 t^2 e^(-g t), 
    phi_2(t) = -g t e^(-g t), phi_1(t) = e^(-g t),
    then
        d/dt phi_4(t) = -g phi_4(t) - g phi_3(t)
        d/dt phi_3(t) = -g phi_3(t) - g phi_2(t)
        d/dt phi_2(t) = -g phi_2(t) - g phi_1(t)
        d/dt phi_1(t) = -g phi_1(t)
    and
        C(t) = c1 phi_1(t) + c2 phi_2(t) + c3 phi_3(t) + c4 phi_4(t)
    where
        c4 = (4 l)/(5 b) - i (2 l g)/(5)
        c3 = (4 l)/(5 b)
        c2 = (4 l)/(5 b)
        c1 = (4 l)/(5 b)
    """
    if re_b and width_b:
        assert len(re_b) == len(
            width_b), "re_b and width_b must have the same length."
        sds = [
            SuperCriticalDamping(quantity(re, 'energy'),
                                 quantity(width, 'energy'),
                                 order=3) for re, width in zip(re_b, width_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    for sd in sds:
        k_head = corr.k_max
        g = sd.gamma
        l = sd.lambda_

        # at high temperatures:
        # c4 = (4 l)/(5 b) - i (2 l g)/(5)
        # c3 = (4 l)/(5 b)
        # c2 = (4 l)/(5 b)
        # c1 = (4 l)/(5 b)
        # Note: 1 / beta approximates g*cot(beta*g/2)/2 at high temperatures
        if not use_ht_function:
            tmp = 0.5 * g / np.tan(beta * g / 2.0)
        else:
            tmp = 1.0 / beta
        c1 = 4.0 * l * tmp / 5.0
        c2 = 4.0 * l * tmp / 5.0
        c3 = 4.0 * l * tmp / 5.0
        c4 = 4.0 * l * tmp / 5.0 - 2.0j * l * g / 5.0
        # coefficient associated with e^(-g * t)
        corr.coefficients.extend([c1, c2, c3, c4])
        corr.conj_coefficents.extend(
            [c1.conjugate(),
             c2.conjugate(),
             c3.conjugate(),
             c4.conjugate()])
        corr.zeropoints.extend([1.0, 0.0, 0.0, 0.0])
        print(f'k_head={k_head}, g={g}, l={l}', flush=True)
        corr.derivatives[k_head, k_head] = -g
        corr.derivatives[k_head + 1, k_head + 1] = -g
        corr.derivatives[k_head + 2, k_head + 2] = -g
        corr.derivatives[k_head + 3, k_head + 3] = -g
        corr.derivatives[k_head + 1, k_head] = g
        corr.derivatives[k_head + 2, k_head + 1] = g
        corr.derivatives[k_head + 3, k_head + 2] = g

    corr._add_ltc(sds, distribution)  # type: ignore
    return corr


def overdamped_brownian_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    freq_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
) -> Correlation:
    """
    Factory function for an overdamped Brownian spectral density.
    """
    if re_b and width_b and freq_b:
        assert len(re_b) == len(width_b) == len(
            freq_b), "re_b, width_b and freq_b must have the same length."
        sds = [
            OverdampedBrownian(
                quantity(re, 'energy'),
                quantity(freq, 'energy'),
                quantity(width, 'energy'),
            ) for re, width, freq in zip(re_b, width_b, freq_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    f = distribution.function
    for sd in sds:
        k_head = corr.k_max
        rs, ps = sd.get_residues_poles()
        # coefficient associated with e^(-g * t)
        c1 = complex(rs[0] * f(np.array(ps[0])))
        c2 = complex(rs[1] * f(np.array(ps[1])))
        corr.coefficients.extend([c1, c2])
        corr.conj_coefficents.extend([c1.conjugate(), c2.conjugate()])
        corr.zeropoints.extend([1.0, 1.0])
        corr.derivatives[k_head, k_head] = -1.0j * ps[0]
        corr.derivatives[k_head + 1, k_head + 1] = -1.0j * ps[1]

    corr._add_ltc(sds, distribution)  # type: ignore
    return corr


def jordan_critically_damped_brownian_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
    use_ht_function: bool = False,
) -> Correlation:
    """
    Factory function for a critically damped Brownian spectral density.

    """
    if re_b and width_b:
        assert len(re_b) == len(
            width_b), "re_b and width_b must have the same length."
        sds = [
            CriticallyDampedBrownian(quantity(re, 'energy'),
                                     quantity(width, 'energy'))
            for re, width in zip(re_b, width_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    for sd in sds:
        k_head = corr.k_max
        g = sd.gamma
        l = sd.lambda_
        # coefficient associated with t e^(-g * t):
        # c2 = l * g * 2.0 / beta - 1.0j * l * g**2
        # coefficient associated with e^(-g * t):
        # c1 = l * 2.0 / beta
        # Note: 2/beta approximates g*cot(beta*g/2) at high temperatures
        if not use_ht_function:
            tmp = g / np.tan(beta * g / 2.0)
        else:
            tmp = 2.0 / beta
        c2 = l * g * tmp - 1.0j * l * g**2
        c1 = l * tmp
        # coefficient associated with e^(-g * t)
        corr.coefficients.extend([c1, c2])
        corr.conj_coefficents.extend([c1.conjugate(), c2.conjugate()])
        corr.zeropoints.extend([1.0, 0.0])
        print(f'k_head={k_head}, g={g}, l={l}', flush=True)
        corr.derivatives[k_head, k_head] = -g
        corr.derivatives[k_head + 1, k_head + 1] = -g
        corr.derivatives[k_head + 1, k_head] = 1.0
        print(corr.derivatives)
    corr._add_ltc(sds, distribution)  # type: ignore
    return corr


def jordan_super_critical_damping_3rd_bcf(
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    temperature: float = 300.0,
    decomposition_method: Literal['Pade', 'Matsubara'] = 'Pade',
    n_ltc: int = 0,
    use_ht_function: bool = False,
) -> Correlation:
    r"""
    Factory function for a Super Critically damped spectral density of order-3 in Jordan form.
    At high temperatures, the BCF (t>0) reads:
        C(t) = (2 l)/(3 beta) ( (g t)^2 + 3 g t + 3 ) e^(-g t)
               - i (l g)/(3) ( (g t)^2 + g t ) e^(-g t)

    Let phi_3(t) = 0.5 * t^2 e^(-g t), phi_2(t) = t e^(-g t), phi_1(t) = e^(-g t),
    then
        d/dt phi_3(t) = -g phi_3(t) - g phi_2(t)
        d/dt phi_2(t) = -g phi_2(t) - g phi_1(t)
        d/dt phi_1(t) = -g phi_1(t)
    and
        C(t) = c1 phi_1(t) + c2 phi_2(t) + c3 phi_3(t)
    where
        c1 = (2 l)/(b) 
        c2 = (2 l g )/(b) - i (l g^2)/(3)
        c3 = (4 l g^2)/(3 b) - i (2 l g^3)/(3)
    """
    if re_b and width_b:
        assert len(re_b) == len(
            width_b), "re_b and width_b must have the same length."
        sds = [
            SuperCriticalDamping(quantity(re, 'energy'),
                                 quantity(width, 'energy'),
                                 order=3) for re, width in zip(re_b, width_b)
        ]
    else:
        sds = []

    corr = Correlation()
    beta = quantity(1.0 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distribution = BoseEinstein(
        n=n_ltc,
        beta=beta,
    )
    for sd in sds:
        k_head = corr.k_max
        g = sd.gamma
        l = sd.lambda_

        # at high temperatures:
        # c1 = (2 l)/(b)
        # c2 = - (2 l)/(b) + i (l g)/(3)
        # c3 = (4 l)/(3 b) - i (2 l g)/(3)
        # Note: 1 / beta approximates g*cot(beta*g/2)/2 at high temperatures
        if not use_ht_function:
            temp = 0.5 * g / np.tan(beta * g / 2.0)
        else:
            temp = 1.0 / beta
        c3 = 4.0 * l * g**2 * temp / 3.0 - 2.0j * l * g**3 / 3.0
        c2 = 2.0 * l * g * temp - 1.0j * l * g**2 / 3.0
        c1 = 2.0 * l * temp
        # coefficient associated with e^(-g * t)
        corr.coefficients.extend([c1, c2, c3])
        corr.conj_coefficents.extend(
            [c1.conjugate(), c2.conjugate(),
             c3.conjugate()])
        corr.zeropoints.extend([1.0, 0.0, 0.0])
        print(f'k_head={k_head}, g={g}, l={l}', flush=True)
        corr.derivatives[k_head, k_head] = -g
        corr.derivatives[k_head + 1, k_head + 1] = -g
        corr.derivatives[k_head + 2, k_head + 2] = -g
        corr.derivatives[k_head + 1, k_head] = 1
        corr.derivatives[k_head + 2, k_head + 1] = 1
    corr._add_ltc(sds, distribution)  # type: ignore
    return corr
