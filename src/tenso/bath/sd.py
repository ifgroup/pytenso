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


class OhmicTruncated(SpectralDensity):

    def __init__(self, reorganization_energy: float, cutoff: float) -> None:
        self.l = reorganization_energy
        self.g = cutoff
        return

    def function(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return l / g * w * np.where(w < g, 1.0, 0.0)


class OhmicSemicircular(SpectralDensity):

    def __init__(self, reorganization_energy: float, cutoff: float) -> None:
        self.h = 2.0 / np.pi * (
            np.sqrt(reorganization_energy * PI + cutoff**2) - cutoff)
        self.g = cutoff
        return

    def function(self, w: complex) -> complex:
        h = self.h
        g = self.g
        window = np.where(
            np.abs(w) < g,
            h,
            np.where(
                np.abs(w) < g + h, h * np.sqrt(1 - (w / h - g / h)**2), 0.0),
        )

        return h * w * window


class UnderdampedGaussian(SpectralDensity):

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
        return l / sqrt(2.0 * PI) / g * w * (np.exp(-0.5 * (
            (w - w1) / g)**2) + np.exp(-0.5 * ((w + w1) / g)**2))


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


class OverdampedBrownian(SpectralDensity):

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
        print(w1, g)
        w0 = sqrt(g**2 - w1**2)
        return (4.0 / PI) * l * g * w0**2 * w / (
            (w**2 - w0**2)**2 + 4.0 * g**2 * w**2)

    def get_residues_poles(self) -> tuple[list[complex], list[complex]]:
        l = self.lambda_
        g = self.gamma
        w = self.omega

        a = complex(-1.0j * l * (g**2 - w**2) / w)

        residues = [a, -a]
        # [-g - 1j * w, -g + 1j * w]
        poles = [-1.0j * (g - w), -1.0j * (g + w)]
        return residues, poles



class CriticallyDampedBrownian(SpectralDensity):

    def __init__(self, reorganization_energy: float,
                 relaxation: float) -> None:
        self.gamma = relaxation
        self.lambda_ = reorganization_energy
        return

    def function(self, w: complex) -> complex:
        l = self.lambda_
        g = self.gamma
        return (4.0 / PI) * l * g**3 * w / (w**2 + g**2)**2

    def get_residues_poles(self):
        raise RuntimeError('2nd order pole needs special treatment.')




class BrownianOscillator(SpectralDensity):

    def __init__(self, reorganization_energy: float,
                 relaxation: float,
                 intrinsic_frequency: float
                 ) -> None:
        self.gamma = relaxation
        self.lambda_ = reorganization_energy
        self.omega0 = intrinsic_frequency
        self.omega1 = np.sqrt(self.omega0**2 - self.gamma**2, dtype=complex)
        return

    def function(self, w: complex) -> complex:
        l = self.lambda_
        g = self.gamma
        w0 = self.omega0
        return (4.0 / PI) * l * g * w0**2 * w / (
            (w**2 - w0**2)**2 + 4.0 * g**2 * w**2)

    def get_residues_poles(self):
        raise RuntimeError('Possible 2nd order pole needs special treatment.')


class SuperCriticalDamping(SpectralDensity):

    def __init__(self,
                 reorganization_energy: float,
                 relaxation: float,
                 order=3) -> None:
        self.gamma = relaxation
        self.lambda_ = reorganization_energy
        self.order = order
        # Renormalization constant: 2^N (N-1)! / (2N-3)!! / PI
        self.norm = 2.0 / PI
        for k in range(1, order):
            self.norm *= 2.0 * k / (2 * k - 1)
        return

    def function(self, w: complex) -> complex:
        l = self.lambda_
        g = self.gamma
        order = self.order
        norm = self.norm
        return norm * l * (w/g) / ((w/g)**2 + 1)**order

    def get_residues_poles(self):
        raise RuntimeError('N-th order pole needs special treatment.')




if __name__ == '__main__':
    from tenso.libs.quantity import Quantity as __
    from matplotlib import pyplot as plt
    unit = __(1000, '/cm').au
    # print('Pseudo g:', sd.lambda_**2 / sd.omega)
    beta = __(1 / 300, '/K').au * unit
    sd = UnderdampedBrownian(0.2, 0.5, 0.01)
    be = BoseEinstein(beta=beta)

    fig, ax = plt.subplots(1, 2, tight_layout=True)
    time_max = 100
    time_max_fs = __(time_max / unit).convert_to('fs').value
    time_space = np.linspace(0, time_max)
    time_space_fs = np.linspace(0, time_max_fs)
    ct_ref = np.array(
        [sd.autocorrelation(t, beta=be.beta) for t in time_space])
    ax[0].plot(time_space_fs, ct_ref.real, 'r:', label='Ref.', lw=3)
    ax[1].plot(time_space_fs, ct_ref.imag, 'r:', label='Ref.', lw=3)
    ax[1].legend()
    plt.show()
