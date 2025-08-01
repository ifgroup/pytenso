#!/usr/bin/env python
# coding: utf-8
"""
Correlation function object
"""
from __future__ import annotations

import json
from typing import Callable, Optional
import numpy as np

from tenso.bath.distribution import BoseEinstein
from tenso.bath.sd import SpectralDensity
from numpy.typing import NDArray

PI = np.pi


class Correlation(object):

    def __init__(self) -> None:
        self.coefficients = list()  # type: list[complex]
        self.conj_coefficents = list()  # type: list[complex]
        self.zeropoints = list()  # type: list[complex]
        self.derivatives = dict()  # type: dict[tuple[int, int], complex]
        self.lindblad_rate = None  # type: Optional[float]
        return

    def dump(self, output_file: str) -> None:
        with open(output_file, 'w') as f:
            c = [(_c.real, _c.imag) for _c in self.coefficients]
            cc = [(_cc.real, _cc.imag) for _cc in self.conj_coefficents]
            z = [(_z.real, _z.imag) for _z in self.zeropoints]
            d = {
                f"{i},{j}": (_d.real, _d.imag)
                for (i, j), _d in self.derivatives.items()
            }
            kwargs = {
                'coefficients': c,
                'conj_coefficents': cc,
                'zeropoints': z,
                'derivatives': d,
                'lindblad_rate': self.lindblad_rate,
            }
            json.dump(kwargs, f, indent=4, sort_keys=True)
        return

    def remove_heom_terms(self) -> None:
        self.coefficients = list()
        self.conj_coefficents = list()
        self.zeropoints = list()
        self.derivatives = dict()
        return

    def load(self, input_file: str) -> None:
        with open(input_file, 'r') as f:
            kwargs = json.load(f)
            c = [complex(x, y) for x, y in kwargs['coefficients']]
            cc = [complex(x, y) for x, y in kwargs['conj_coefficents']]
            z = [complex(x, y) for x, y in kwargs['zeropoints']]
            dct = kwargs['derivatives']  # type: dict[str, tuple[float, float]]
            d = dict()  # type: dict[tuple[int, int], complex]
            for string, (x, y) in dct.items():
                idx = string.split(',')
                i = int(idx[0])
                j = int(idx[1])
                d[i, j] = complex(x, y)
            lr = kwargs['lindblad_rate']  # type: Optional[float]
            assert len(c) == len(cc) == len(z)
            self.coefficients = c
            self.conj_coefficents = cc
            self.zeropoints = z
            self.derivatives = d
            self.lindblad_rate = lr
        return

    @property
    def k_max(self):
        assert len(self.coefficients) == len(self.zeropoints)
        return len(self.coefficients)

    def add_discrete_vibration(self, frequency: float, coupling: float,
                               beta: Optional[float]) -> None:
        w0 = frequency
        g = coupling

        coth = 1.0 / np.tanh(beta * w0 / 2.0) if beta is not None else 1.0
        self.coefficients.extend(
            [g**2 / 2.0 * (coth + 1.0), g**2 / 2.0 * (coth - 1.0)])
        self.conj_coefficents.extend(
            [g**2 / 2.0 * (coth - 1.0), g**2 / 2.0 * (coth + 1.0)])
        self.zeropoints.extend([1.0, 1.0])
        k = len(self.derivatives)
        self.derivatives[k, k] = -1.0j * w0
        self.derivatives[k + 1, k + 1] = 1.0j * w0
        return

    def add_discrete_trigonometric(self, frequency: float, coupling: float,
                                   beta: Optional[float]) -> None:
        w0 = frequency
        g = coupling

        coth = 1.0 / np.tanh(beta * w0 / 2.0) if beta is not None else 1.0
        c1 = g**2 / 2.0 * (coth + 1.0)
        c2 = g**2 / 2.0 * (coth - 1.0)
        cp =complex(c2 + c1)
        cm = complex(c2 - c1) * 1.0j
        self.coefficients.extend([cp, cm])
        self.conj_coefficents.extend([cp.conjugate(), cm.conjugate()])
        self.zeropoints.extend([1.0, 0.0])  # cos * exp, sin * exp
        k = len(self.derivatives)
        self.derivatives[k, k + 1] = -w0
        self.derivatives[k + 1, k] = w0
        return

    def _add_ltc(self, sds: list[SpectralDensity], distribution: BoseEinstein):
        """Add LTC terms for spectral densities with poles.
        """
        rs, ps = distribution.get_residues_poles()
        if sds and rs and ps:
            for res, pole in zip(rs, ps):
                cs = [res * sd.function(pole) for sd in sds]
                c = np.sum(cs)
                self.coefficients.append(c)
                self.conj_coefficents.append(np.conj(c))
                self.zeropoints.append(1.0)
                k = len(self.derivatives)
                self.derivatives[k, k] = -1.0j * pole

        return

    def add_spectral_densities(self,
                               sds: list[SpectralDensity],
                               distribution: BoseEinstein,
                               zeropoint=1.0,
                               use_ht_function=False):
        f = distribution.function if not use_ht_function else distribution.ht_function
        for sd in sds:
            rs, ps = sd.get_residues_poles()
            if len(rs) == 1:
                c = complex(rs[0] * f(np.array(ps[0])))
                self.coefficients.append(c / zeropoint)
                self.conj_coefficents.append(c.conjugate() )
                self.zeropoints.append(zeropoint)
                k = len(self.derivatives)
                self.derivatives[k, k] = -1.0j * ps[0]
            elif len(rs) == 2:
                c1 = complex(rs[0] * f(np.array([ps[0]])) / zeropoint)
                c2 = complex(rs[1] * f(np.array([ps[1]])) / zeropoint)
                self.coefficients.extend([c1, c2])
                self.conj_coefficents.extend([c2.conjugate(), c1.conjugate()])
                self.zeropoints.extend([zeropoint, zeropoint])
                k = len(self.derivatives)
                self.derivatives[k, k] = -1.0j * ps[0]
                self.derivatives[k + 1, k + 1] = -1.0j * ps[1]
            else:
                raise RuntimeError(
                    'Poles must be symmetric along the imag axis.')

        self._add_ltc(sds, distribution)
        return

    def add_trigonometric(self, sds: list[SpectralDensity],
                          distribution: BoseEinstein):
        f = distribution.function
        for sd in sds:
            rs, ps = sd.get_residues_poles()
            if len(rs) == 2:
                # ps = [-1.0j * (g + 1.0j * w), -1.0j * (g - 1.0j * w)]
                g = (ps[0] + ps[1]) * 0.5j
                w = (ps[0] - ps[1]) * 0.5
                c1 = rs[0] * f(np.array(ps[0]))  # for term exp[(- iw - g) t]
                c2 = rs[1] * f(np.array(ps[1]))  # for term exp[(+ iw - g) t]
                cp = complex(c2 + c1)
                cm = complex(c2 - c1) * 1.0j
                self.coefficients.extend([cp, cm])
                self.conj_coefficents.extend([cp.conjugate(), cm.conjugate()])
                self.zeropoints.extend([1.0, 0.0])  # cos * exp, sin * exp
                k = len(self.derivatives)
                self.derivatives[k, k] = -g
                self.derivatives[k, k + 1] = -w
                self.derivatives[k + 1, k] = w
                self.derivatives[k + 1, k + 1] = -g
            elif len(rs) == 1:
                c = complex(rs[0] * f(np.array(ps[0])))
                self.coefficients.append(c)
                self.conj_coefficents.append(c.conjugate())
                self.zeropoints.append(1.0)
                k = len(self.derivatives)
                self.derivatives[k, k] = -1.0j * ps[0]
            else:
                raise RuntimeError(
                    'Poles must be symmetric along the imag axis.')

        self._add_ltc(sds, distribution)
        return

    def real_correlation_function(self, t):
        ans = np.zeros_like(t)
        for k, c in enumerate(self.coefficients):
            g = complex(self.derivatives[k, k])
            ans += c.real * np.exp(g.real * t) * np.cos(g.imag * t)
            ans -= c.imag * np.exp(g.real * t) * np.sin(g.imag * t)
        return ans

    def imag_correlation_function(self, t):
        ans = np.zeros_like(t)
        for k, c in enumerate(self.coefficients):
            g = complex(self.derivatives[k, k])
            ans += c.real * np.exp(g.real * t) * np.sin(g.imag * t)
            ans += c.imag * np.exp(g.real * t) * np.cos(g.imag * t)
        return ans

    def __str__(self) -> str:
        if self.k_max > 0:
            string = f"Correlation ( c | c* | z ) x{self.k_max} :"
            for c, cc, z in zip(self.coefficients, self.conj_coefficents,
                                self.zeropoints):
                string += f"\n{c.real:+.4e}{c.imag:+.4e}j | {cc.real:+.4e}{cc.imag:+.4e}j | {z.real:+.2e}{z.imag:+.2e}j"
            string += "\nDerivatives:"
            string += "".join([
                f"\n  [{i:d}, {j:d}] : {v.real:+.4e}{v.imag:+.4e}j"
                for (i, j), v in self.derivatives.items()
            ])
        else:
            string = 'No HEOM correlations.'
        if self.lindblad_rate is not None:
            string += f'\nLindblad rate: {self.lindblad_rate:.4e}'
        else:
            string += '\nNo Lindblad rate.'
        return string
