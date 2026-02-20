#!/usr/bin/env python
# coding: utf-8
"""
Spectral density factory
"""
from __future__ import annotations
from typing import Optional

from tenso.bath.distribution import BoseEinstein

from tenso.libs.backend import PI
import numpy as np
from math import erf, exp, sqrt
from scipy.integrate import simpson, quad
from scipy.special import factorial, factorial2


class SpectralDensity:
    """
    Template for a spectral density.
    """
    FREQ_MIN = 1e-14
    FREQ_MAX = 1e3

    # print(FREQ_MAX, FREQ_MIN)

    def autocorrelation(self,
                        t: float,
                        beta: Optional[float] = None) -> complex:

        def _re(w):
            if beta is None:
                coth = 1.0
            else:
                coth = 1.0 / np.tanh(beta * w / 2.0)
                # coth = 2.0 / (beta * w)
            return self.function(w) * np.cos(w * t) * coth

        def _im(w):
            return -self.function(w) * np.sin(w * t)

        # w = np.logspace(np.log2(self.FREQ_MIN),
        #                 np.log2(self.FREQ_MAX),
        #                 num=100_000,
        #                 base=2)
        w = np.linspace(self.FREQ_MIN, self.FREQ_MAX, num=100_000)
        re = simpson(_re(w), w)
        im = simpson(_im(w), w)

        # re = quad(_re, self.FREQ_MIN, self.FREQ_MAX)[0]
        # im = quad(_im, self.FREQ_MIN, self.FREQ_MAX)[0]
        return complex(re + 1.0j * im)

    def function(self, w: complex) -> complex:
        raise NotImplementedError(
            'Spectral density function must be implemented in the subclass.')

    def get_residues_poles(self) -> tuple[list[complex], list[complex]]:
        """Get (-2 PI I * residues) and poles of the spectral density in the lower half-plane.

        Returns:
            tuple[list[complex], list[complex]]: List of (-2 PI I * residues) and poles.
        """
        raise NotImplementedError(
            'Spectral density residues and poles must be implemented in the subclass.'
        )


class Drude(SpectralDensity):

    def __init__(self, reorganization_energy: float,
                 relaxation: float) -> None:
        self.l = reorganization_energy
        self.g = relaxation
        return

    def function(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return (2.0 / PI) * l * g * w / (w**2 + g**2)

    def get_residues_poles(self) -> tuple[list[complex], list[complex]]:
        return [-2.0j * self.l * self.g], [-1.0j * self.g]


class UnderdampedBrownian(SpectralDensity):

    def __init__(self, reorganization_energy: float, frequency: float,
                 relaxation: float) -> None:
        self.omega = frequency
        self.gamma = relaxation
        self.lambda_ = reorganization_energy
        return

    def function(self, w: complex) -> complex:
        l = self.lambda_
        g = self.gamma
        w1 = self.omega
        return (4.0 / PI) * l * g * (w1**2 + g**2) * w / (
            (w + w1)**2 + g**2) / ((w - w1)**2 + g**2)

    def get_residues_poles(self) -> tuple[list[complex], list[complex]]:
        l = self.lambda_
        g = self.gamma
        w = self.omega

        a = complex(l * (w**2 + g**2) / w)

        residues = [a, -a]
        # [-g - 1j * w, -g + 1j * w]
        poles = [-1.0j * (g + 1.0j * w), -1.0j * (g - 1.0j * w)]
        return residues, poles

class OhmicExp(SpectralDensity):

    def __init__(self, reorganization_energy: float, cutoff: float) -> None:
        self.l = reorganization_energy
        self.g = cutoff
        return

    def function(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return l / g * w * np.exp(-w / g)

    def get_residues_poles(self):
        raise RuntimeError('Infinite order of pole for OhmicExp.')




