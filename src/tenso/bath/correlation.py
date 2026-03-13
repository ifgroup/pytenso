#!/usr/bin/env python
# coding: utf-8
"""
Object for handling the correlation function
"""
from __future__ import annotations

import json
from typing import Callable, Optional
import numpy as np

from tenso.bath.distribution import BoseEinstein
from tenso.bath.sd import SpectralDensity
from numpy.typing import NDArray
from tenso.prototypes.default_parameters import quantity

PI = np.pi


class Correlation(object):
    """ This class represents a correlation function for a single bath which
    may be coupled to the system which is stored in a decomposed form such that
    `C(t) = sum_k c_k e^{-gamma_k}`, with `c_k` and `gamma_k` being complex. 

    :param coefficients: The values of`c_k`
    :type coefficients: list, complex

    :param conj_coefficents: The complex conjucates of coefficients
    :type conj_coefficents: list, complex

    :param zeropoints: Initial states of the bexcitons
    :type zeropoints": list, real

    :param derivatives: The values of `gamma_k` in a dictionary where the keys are the pair of integers [k, k] and the value is `gamma_k.` Format as a dictionary is for historical reasons.
    :type derivatives: dictionary 
    
    :param lindblad_rate: Experimental parameter for use in more complicated equations of motion
    :type linblad_rate: list, optional

    """

    def __init__(self) -> None:
        """ Constructor of an empty correlation function.
        """
        self.coefficients = list()  # type: list[complex]
        self.conj_coefficents = list()  # type: list[complex]
        self.zeropoints = list()  # type: list[complex]
        self.derivatives = dict()  # type: dict[tuple[int, int], complex]
        self.lindblad_rate = None  # type: Optional[float]
        return

    def dump(self, output_file: str) -> None:
        """ Outputs internal state of the correlation function to the output file.

        :param output_file: File to write correlation function state to.
        :type output_file: string
        """
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
        """ Erases all coefficient, zeropoint, derivative and conjugate coefficient
        information from the correlation function. Note output is in TENSO's internal
        units.
        """
        self.coefficients = list()
        self.conj_coefficents = list()
        self.zeropoints = list()
        self.derivatives = dict()
        return

    def load(self, input_file: str) -> None:
        """ Imports information from a previously dumped correlation function file.
        Note that the imported file must be in TENSO's internal units.

        :param input_file: File formated as from :class: Correlation.dump
        :type input_file: string
        """
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


    def manual_corr_setup(self, c_ks: list[complex], gamma_ks: list[complex], unit_convert: bool = False):
        """ Method to initialize the correlation function object if the form of the
        correlation function in exponential breakdown is already known. This is for 
        an HEOM style bath, not a star boson style bath, and zeropoints will be set to 1.
        Any complex gamma_k values (within tolerance) will result in gamma_k^* being introduced as an 
        additional feature with a coefficient of 0 and a conjugate coefficient of c_k*.
        
        :param c_ks: list of c coefficients in the correlation function breakdown
        :type c_ks: list[complex]
        :param gamma_ks: list of gamma exponential coefficients in the correlation function breakdown
        :type gamma_ks: list[complex]
        :param unit_convert: whether to convert input to internal units
        :type unit_convert: Boolean
        """
        self.conj_coefficents = [] # Clear contents
        self.coefficients = []
        self.derivatives = {}
        assert (len(c_ks) == len(gamma_ks)), "Length of correlation coefficient lists must match."
        internal_gamma_ks = []
        internal_c_ks = []
        # Gammas are being provided in default external units
        if (unit_convert):
            # Gammas have energy units
            con_factor = quantity(1.0, 'energy')
            for gk in gamma_ks:
                internal_gamma_ks.append(gk*con_factor) 
            # c_ks have energy squared units. Convert to internal units of energy
            con_factor = con_factor*con_factor
            for c_k in c_ks:
                internal_c_ks.append(con_factor*c_k)
        else:
            internal_gamma_ks = gamma_ks
            internal_c_ks = c_ks
        kk = 0
        for c_k, gamma in zip(internal_c_ks, internal_gamma_ks):
            if (abs(gamma.imag) > 1e-8 and abs(gamma.imag/gamma.real) > 1e-5): # Complex gamma
                self.derivatives.update({(kk,kk): gamma})
                self.coefficients.append(c_k)
                self.conj_coefficents.append(0.0)
                kk = kk + 1
                self.derivatives.update({(kk,kk): gamma.conjugate()})
                self.coefficients.append(0.0)
                self.conj_coefficents.append(c_k.conjugate())
                kk = kk + 1
            else: # real gamma
                self.derivatives.update({(kk,kk): gamma.real})
                self.coefficients.append(c_k)
                self.conj_coefficents.append(c_k.conjugate())
                kk = kk + 1
        num_cs = len(self.coefficients) # Length added so far
        self.zeropoints = [complex(1.0)]*(num_cs)
        return




    @property
    def k_max(self):
        """ Returns the number of features in the correlation function.

        :returns: The number of features in the correlation function.
        :rtype: integer
        """
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
                c1 = complex(rs[0] * f(np.array([ps[0]]))[0] / zeropoint)
                c2 = complex(rs[1] * f(np.array([ps[1]]))[0] / zeropoint)
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
        """Get the real part of the correlation function at time 't'

        :param t: The time in internal units at which to evaluate the correlation function
        :type t: float

        :returns: The real part of the correlation function
        :rtype: float
        """
        ans = np.zeros_like(t)
        for k, c in enumerate(self.coefficients):
            g = complex(self.derivatives[k, k])
            ans += c.real * np.exp(g.real * t) * np.cos(g.imag * t)
            ans -= c.imag * np.exp(g.real * t) * np.sin(g.imag * t)
        return ans

    def imag_correlation_function(self, t):
        """Get the imaginary part of the correlation function at time 't'

        :param t: The time in internal units at which to evaluate the correlation function
        :type t: float

        :returns: The imarinary part of the correlation function
        :rtype: float
        """
        ans = np.zeros_like(t)
        for k, c in enumerate(self.coefficients):
            g = complex(self.derivatives[k, k])
            ans += c.real * np.exp(g.real * t) * np.sin(g.imag * t)
            ans += c.imag * np.exp(g.real * t) * np.cos(g.imag * t)
        return ans


    def __str__(self) -> str:
        """Get a string represenation of the correlation function

        :returns: string description
        :rval: string
        """
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
