#!/usr/bin/env python
# coding: utf-8
"""
Diagonal Correlation function object.
"""
from __future__ import annotations
import json

from typing import Literal, Optional
import numpy as np

from tenso.bath.distribution import BoseEinstein
from tenso.bath.sd import SpectralDensity
from tenso.libs.backend import NDArray
from scipy.special import jv, chebyt, j0, j1
from scipy.integrate import simpson, quad
from scipy.optimize import root_scalar
from functools import partial
from scipy import linalg

quad = partial(quad, epsabs=1.0e-14, epsrel=1.0e-14, limit=100)

from tenso.prototypes.default_parameters import quantity, value

EPSILON = 1.0e-21
ABSORB_RATE = 0.1
N_INT = 100_000  # Number of points for numerical integration


class StarBosons:

    def __init__(self) -> None:
        self.couplings = list()  # type: list[float | complex]
        self.conj_couplings = list()  # type: list[float | complex]
        self.frequencies = dict()  # type: dict[tuple[int, int], complex]
        self.base_function = None
        return
    
    def get_reorganization_energy(self) -> float:
        assert self.is_diagonalized()
        re = 0.0
        for k in range(self.k_max):
            c = self.couplings[k]
            w = self.frequencies[k, k]
            re += abs(c)**2 * w.real
        return re

    def filter(self, underflow: float) -> None:
        keep_idx = set()
        for n, (z, cz) in enumerate(zip(self.couplings, self.conj_couplings)):
            if abs(z) > underflow or abs(cz) > underflow:
                keep_idx.add(n)
        keep_ij = set()
        for (i, j), v in self.frequencies.items():
            if i in keep_idx or j in keep_idx:
                if abs(v) > underflow:
                    keep_ij.add((i, j))
                    keep_idx.add(i)
                    keep_idx.add(j)
        idx_map = {i: n for n, i in enumerate(keep_idx)}
        # print('[Debug]', idx_map, flush=True)
        self.couplings = [
            c for n, c in enumerate(self.couplings) if n in keep_idx
        ]
        self.conj_couplings = [
            c for n, c in enumerate(self.conj_couplings) if n in keep_idx
        ]
        self.frequencies = {
            (idx_map[i], idx_map[j]): d
            for (i, j), d in self.frequencies.items() if (i, j) in keep_ij
        }
        return
    
    def degenerate(self, multiplicity: int):
        assert self.is_diagonalized()
        assert multiplicity >= 1

        new_couplings = []  
        new_conj_couplings = []
        new_frequencies = dict()

        mode_idx = 0
        for k in range(self.k_max):
            c = self.couplings[k] / np.sqrt(multiplicity)
            cc = self.conj_couplings[k] / np.sqrt(multiplicity)
            freq = self.frequencies.get((k, k), 0.0)
            if abs(freq) > EPSILON: 
                for _ in range(multiplicity):
                    new_couplings.append(c)
                    new_conj_couplings.append(cc)
                    new_frequencies[mode_idx, mode_idx] = freq
                    mode_idx += 1
        
        self.couplings = new_couplings
        self.conj_couplings = new_conj_couplings
        self.frequencies = new_frequencies
        self.base_function = None
        return


    def dump(self, output_file: str) -> None:
        with open(output_file, 'w') as f:
            z = [(_z.real, _z.imag) for _z in self.couplings]
            cz = [(_z.real, _z.imag) for _z in self.conj_couplings]
            d = {
                f"{i},{j}": (_d.real, _d.imag)
                for (i, j), _d in self.frequencies.items()
            }
            kwargs = {
                'couplings': z,
                'conj_couplings': cz,
                'frequencies': d,
            }
            json.dump(kwargs, f, indent=4, sort_keys=True)
        return

    def load(self, input_file: str) -> None:
        with open(input_file, 'r') as f:
            kwargs = json.load(f)
            z = [complex(x, y) for x, y in kwargs['couplings']]
            cz = [complex(x, y) for x, y in kwargs['conj_couplings']]
            dct = kwargs['frequencies']  # type: dict[str, tuple[float, float]]
            d = dict()
            for string, (x, y) in dct.items():
                idx = string.split(',')
                i = int(idx[0])
                j = int(idx[1])
                d[i, j] = complex(x, y)
            self.couplings = z
            self.conj_couplings = cz
            self.frequencies = d
        return

    @property
    def k_max(self):
        return len(self.couplings)

    def add_discrete_vibrations(self, ph_parameters: list[tuple[float, float]],
                                beta: Optional[float]) -> None:
        """
        ph_parameters: list[frequency, coupling]
        """
        distr = BoseEinstein(beta=beta)
        for w, g in ph_parameters:
            k0 = self.k_max
            if beta is None:
                self.couplings.append(g)
                self.conj_couplings.append(g)
                self.frequencies[k0, k0] = -1.0j * w
            else:
                g_p = g * np.sqrt(distr.function(w))
                g_m = g * np.sqrt(-distr.function(-w))
                self.couplings.extend([g_p, g_m])
                self.conj_couplings.extend([g_p, g_m])
                self.frequencies[k0, k0] = -1.0j * w
                self.frequencies[k0 + 1, k0 + 1] = 1.0j * w
        return
    
    def is_diagonalized(self) -> bool:
        ans = True
        for (i, j), v in self.frequencies.items():
            if i != j and abs(v) > EPSILON:
                ans = False
                break
        return ans

    def diagonalize(self) -> None:
        freq_mat = np.zeros((self.k_max, self.k_max), dtype=complex)
        for (i, j), v in self.frequencies.items():
            freq_mat[i, j] = v
        e, s = np.linalg.eig(freq_mat)
        inv_s = np.linalg.inv(s)
        new_frequencies = dict()
        for n, ei in enumerate(e):
            new_frequencies[n, n] = ei
        z = np.array(self.couplings)
        cz = np.array(self.conj_couplings)

        z = z @ inv_s
        cz = s @ cz
        self.couplings = list(z)
        self.conj_couplings = list(cz)
        self.frequencies = new_frequencies
        self.base_function = None
        return
    
    def total_reorganization_energy(self) -> float:
        re = 0.0
        for k in range(self.k_max):
            c = self.couplings[k]
            w = self.frequencies[k, k]
            if (w.imag) > EPSILON:
                re += abs(c)**2 / w.imag
            # re += abs(c)**2 / abs(w.imag)
        return re

    def real_correlation_function(self, t):
        ans = np.zeros_like(t)
        for k, (c, cc) in enumerate(zip(self.couplings, self.conj_couplings)):
            g = complex(self.frequencies[k, k])
            coeff = c * cc
            ans += coeff.real * np.exp(g.real * t) * np.cos(g.imag * t)
            ans -= coeff.imag * np.exp(g.real * t) * np.sin(g.imag * t)
        return ans

    def imag_correlation_function(self, t):
        ans = np.zeros_like(t)
        for k, (c, cc) in enumerate(zip(self.couplings, self.conj_couplings)):
            g = complex(self.frequencies[k, k])
            coeff = c * cc
            ans += coeff.real * np.exp(g.real * t) * np.sin(g.imag * t)
            ans += coeff.imag * np.exp(g.real * t) * np.cos(g.imag * t)
        return ans

    def add_spectral_densities(self,
                               sds: list[SpectralDensity],
                               beta: None | float,
                               n: int,
                               method='Fourier',
                               cutoff: float | None = None,
                               n_int: int = N_INT,
                               shift_frequency: bool = True,
                               log_base: int = 10,
                               log_minimum_frequency: float | None = None,
                               absorb_rate: float | None = ABSORB_RATE,
                               int_grid_size: float | None = None, 
                               int_lower_bound: float | None = None,
                               ) -> None:
        """
        Add spectral densities to the diagonal correlation function.
        
        Parameters
        ----------
        sds : list[SpectralDensity]
            The spectral densities.
        beta : float
            The inverse temperature.
        n : int
            The number of discrete modes to compute.
        method : str
            The method to compute the diagonal correlation function.
            Currently, 'Fourier', 'LogFourier', 'EqualReorganizationEnergy' and 'Chebyshev' are supported.
        cutoff : float
            The cutoff frequency. (for the Chebyshev method)
        n_int : int
            The number of points for numerical integration. (for the Chebyshev method)
        shift_frequency : bool
            Shift the frequency space to avoid the zero-frequency component. (for the Fourier method)
        absorb_rate : float
            The absorption rate. (for the Chebyshev method)
        """

        w = np.linspace(0, cutoff, num=n_int)
        j = np.zeros_like(w)
        for b in sds:
            j += b.function(w)
        print('Method:', method, flush=True)
        if method == 'Fourier':
            c = Fourier(sds,
                        BoseEinstein(n=0, beta=beta),
                        cutoff,
                        n,
                        shift=shift_frequency)
        elif method == 'LogFourier':
            c = LogFourier(sds,
                           BoseEinstein(n=0, beta=beta),
                           cutoff,
                           n,
                           minimum_frequency=log_minimum_frequency,
                           base=log_base)
        elif method == 'EqualReorganizationEnergy':
            if cutoff is not None:
                EqualReorganizationEnergy.upper_bound = cutoff
            if int_lower_bound is not None:
                EqualReorganizationEnergy.lower_bound = int_lower_bound
            c = EqualReorganizationEnergy(sds,
                                          BoseEinstein(n=0, beta=beta),
                                          n,
                                          int_grid_size=int_grid_size)
        elif method == 'Chebyshev':
            c = Chebyshev(w, j, beta, n, cutoff, absorb_rate=absorb_rate)
            # raise NotImplementedError
        # elif method == 'LiftedChebyshev':
        #     c = LiftedChebyshev(sds, beta, n, cutoff)
        elif method == 'TEDOPA':
            # print(w[0], w[-1], flush=True)
            # print(j[0], j[-1], flush=True)
            c = Tedopa(w, j, beta, n)
        else:
            raise NotImplementedError

        k0 = self.k_max
        z, d = c.get_star_parameters()
        self.singlar_values = z
        # print(z, d, flush=True)
        if hasattr(c, 'base_function'):
            self.base_function = c.base_function
            z = np.array([
                zk * self.base_function(k, 0.0).conj()
                for k, zk in enumerate(z)
            ],
                         dtype=complex)
        else:
            self.base_function = None
        self.couplings.extend(z)
        self.conj_couplings.extend(z.conjugate())
        d = -1.0j * d  # Convert to the derivative matrix.
        n_pts = len(z)
        for i in range(n_pts):
            for j in range(n_pts):
                dij = d[i, j]
                if abs(dij) > EPSILON:
                    print(f'[Debug] {i}, {j} : {dij}', flush=True)
                    self.frequencies[k0 + i, k0 + j] = dij

        return

    def __str__(self) -> None:
        # print('[Debug]', self.zeropoints, self.derivatives, flush=True)
        if self.k_max > 0:
            string = f"Correlation {self.k_max}x (z): " + ', '.join(
                f"{z:.4e}" for z in self.couplings) + '\n'
            string += f"Correlation {self.k_max}x (z*): " + ', '.join(
                f"{z:.4e}" for z in self.conj_couplings) + '\n'
            string += "Derivatives:"
            string += "".join([
                f"\n    [{i:d}, {j:d}] : {v.real:+.4e}{v.imag:+.4e}j"
                for (i, j), v in self.frequencies.items()
            ])
        else:
            string = 'Empty Diagonal Correlation object'
        return string

    def autocorrelation(self, t: float, t0: float = 0.0) -> complex:
        if self.base_function is None:
            # Assuming the base function is the exponential function.
            base_function = lambda k, t: np.exp(-1.0j * self.frequencies[k, k]
                                                * t)
        else:
            base_function = self.base_function
        ans = np.zeros_like(t, dtype=complex)
        for k in range(self.k_max):
            ans += np.abs(self.singlar_values[k])**2 * base_function(
                k, (t + t0)) * base_function(k, t0).conj()
        return ans


class DiscretizationSolver:
    underflow = 1.0e-14

    def __init__(self, w: NDArray, j: NDArray, beta: float | None, n_max: int):

        w = np.array(w)
        j = np.array(j)
        # Remove the zero-frequency component.
        if abs(w[0]) < self.underflow:
            w = w[1:]
            j = j[1:]

        self.frequency_space = np.concatenate((-w[::-1], w))
        if beta is None:
            self.sd_space = np.concatenate((-j[::-1], j))
            self.tsd_space = np.concatenate((np.zeros_like(j), j))
        else:
            self.sd_space = np.concatenate((-j[::-1], j))
            self.tsd_space = self.be_function(
                beta * self.frequency_space) * self.sd_space
        self.beta = beta
        self.n_max = n_max

        return

    @staticmethod
    def be_function(x: NDArray) -> NDArray:
        return 0.5 + 0.5 / np.tanh(0.5 * x)

    def __str__(self) -> str:
        string = (
            f"Discretization: n = {self.n_max}, beta = {self.beta}, " +
            f"omega in ({self.frequency_space.min()}, {self.frequency_space.max})"
        )
        return string

    def get_star_parameters(self) -> NotImplementedError:
        raise NotImplementedError


class Tedopa(DiscretizationSolver):

    def __init__(self, w: NDArray, j: NDArray, beta: float | None, n_max: int):
        """
        The chain map coefficients used in the (T-)TEDOPA algorithm.

        Parameters
        ----------
        sd : Callable[[NDArray], NDArray]
            The spectral density function.
        beta : float
            The inverse temperature.
        n_max : int
            The maximum number of chain map coefficients to compute.
        ret
        """
        super().__init__(w, j, beta, n_max)

        # Coefficents for the orthogonal polynomials.
        # ip[n] = <p_n, p_n>
        self._ip = np.zeros(n_max)
        # ipx[n] = <x * p_n, p_n>
        self._ipx = np.zeros(n_max)
        # alpha[n] = <x * p_n, p_n> / <p_n, p_n>
        self._alpha = np.zeros(n_max)
        # beta[n] = <p_n, p_n> / <p_{n-1}, p_{n-1}>
        self._beta = np.zeros(n_max)

        self._generate_polynomials()
        e, v = np.linalg.eigh(self._get_chain_matrix())
        mat = np.diag(e)
        self._zeropoints = self._get_c0() * v[:, 0]
        self._freq_mat = mat
        return

    def get_star_parameters(self) -> tuple[NDArray, NDArray]:
        return self._zeropoints, self._freq_mat

    def base_function(self, k: int, t: NDArray) -> NDArray:
        return np.exp(-1.0j * self._freq_mat[k, k] * t)

    def _get_c0(self) -> float:
        """
        Compute the zeroth-order coefficient.
        """
        return np.sqrt(self._ip[0])

    def _get_chain_frequency(self) -> NDArray:
        """
        Compute the chain frequencies.
        """
        return self._alpha

    def _get_chain_coupling(self) -> NDArray:
        """
        Compute the chain couplings.
        """
        return np.sqrt(self._beta)

    def _get_chain_matrix(self) -> NDArray:
        """
        Compute the chain matrix.
        """
        return np.diag(self._get_chain_coupling()[1:], k=1) + \
            np.diag(self._get_chain_coupling()[1:], k=-1) + \
            np.diag(self._get_chain_frequency())

    def _integrate(self, f: NDArray) -> float:
        """
        Integrate a function with respect to the spectral density.
        """
        return simpson(f * self.tsd_space, x=self.frequency_space)

    def _generate_polynomials(self) -> None:
        """
        Generate the orthogonal polynomials.
        """
        one = np.ones_like(self.frequency_space)
        # base case
        p_n2 = np.zeros_like(self.frequency_space)
        p_n1 = np.ones_like(self.frequency_space)
        self._ip[0] = self._integrate(p_n1 * p_n1)
        self._ipx[0] = self._integrate(self.frequency_space * p_n1 * p_n1)
        self._alpha[0] = self._ipx[0] / self._ip[0]
        self._beta[0] = 0.0
        for n in range(1, self.n_max):
            p_n = (self.frequency_space - self._alpha[n - 1] * one) * p_n1 - \
                self._beta[n - 1] * p_n2
            self._ip[n] = self._integrate(p_n * p_n)
            self._ipx[n] = self._integrate(self.frequency_space * p_n * p_n)
            self._alpha[n] = self._ipx[n] / self._ip[n]
            self._beta[n] = self._ip[n] / self._ip[n - 1]
            p_n2 = p_n1
            p_n1 = p_n
        return


class EqualReorganizationEnergy(DiscretizationSolver):
    """
    Discretize the spectral density into finite modes with equal reorganization energy.
    Ref: https://doi.org/10.1002/jcc.24527
    """
    lower_bound = quantity(1.0e-5, 'energy')
    upper_bound = quantity(1.0e5, 'energy')

    def __init__(
        self,
        sds: list[SpectralDensity],
        distr: BoseEinstein,
        n: int,
        int_grid_size: float | None = None,
    ) -> None:
        if int_grid_size is None:
            self.int_grid_size = quantity(1e-2, 'energy')
        else:
            self.int_grid_size = int_grid_size
        print(f'[debug] int_grid_size={self.int_grid_size}', flush=True)
        print(f'[debug] lower_bound={value(self.lower_bound, "energy")}', flush=True)
        print(f'[debug] upper_bound={value(self.upper_bound, "energy")}', flush=True)


        self.distr = distr
        self.n = n
        self.sds = sds
        freqencies = []
        zeropoints = []

        zeropoints, freqencies = self._iterative_positive_solver()
        if distr.beta is not None:
            zeropoints_neg, freqencies_neg = self._iterative_negative_solver()
            zeropoints += zeropoints_neg
            freqencies += freqencies_neg

        self._freqencies = np.array(freqencies)
        self._zeropoints = np.array(zeropoints)

        return

    def _iterative_positive_solver(self) -> tuple[list[float], list[float]]:
        n = self.n
        print(f'[debug] n={n}', flush=True) 
        int_grid_size = self.int_grid_size
        f = self.lambda_w
        re = quad(f, self.lower_bound, self.upper_bound)[0]

        print(f'[debug] re_pos={re}', flush=True)
        re_j = re / n
        target_cumm_lambda = np.array([(j + 0.5) * re_j for j in range(n)])
        actual_cumm_lambda = np.zeros(n)
        # print(sum(target_cumm_lambda))
        freq_space = np.zeros(n)

        j = 0
        lower_bound = self.lower_bound
        upper_bound = lower_bound + int_grid_size
        while (j < n):
            if upper_bound > self.upper_bound:
                break
            print(f'[debug] j={j} for [{value(lower_bound, 'energy')}, {value(upper_bound, 'energy')}]', end='\r')
            interval_re = quad(f, lower_bound, upper_bound)[0]
            if interval_re > target_cumm_lambda[j]:
                freq_space[j] = upper_bound 
                actual_cumm_lambda[j] = interval_re
                upper_bound += int_grid_size
                j += 1
                print()
            else:
                upper_bound += int_grid_size

        print()
        lambda_space = np.array([1.0] * n) * re_j
        return list(np.sqrt(lambda_space)), list(freq_space)

    def _iterative_negative_solver(self) -> tuple[list[float], list[float]]:
        n = self.n
        int_grid_size = self.int_grid_size
        f = self.lambda_w
        re = quad(f, -self.upper_bound, -self.lower_bound)[0]
        print(f'[debug] re_neg={re}', flush=True)
        re_j = re / n
        target_cumm_lambda = np.array([(j + 0.5) * re_j for j in range(n)])
        actual_cumm_lambda = np.zeros(n)
        # print(sum(target_cumm_lambda)) 
        freq_space = np.zeros(n)

        j = 0
        lower_bound = self.lower_bound
        upper_bound = lower_bound + int_grid_size
        while (j < n):
            if upper_bound > self.upper_bound:
                break
            print(f'[debug] j={j} for [{value(-lower_bound, 'energy')}, {value(-upper_bound, 'energy')}]', end='\r')
            interval_re = quad(f, -upper_bound, -lower_bound)[0] 
            if interval_re > target_cumm_lambda[j]:
                freq_space[j] = -upper_bound 
                actual_cumm_lambda[j] = interval_re
                upper_bound += int_grid_size
                j += 1
                print()
            else:
                upper_bound += int_grid_size
        print()
        lambda_space = np.array([1.0] * n) * re_j
        return list(np.sqrt(lambda_space)), list(freq_space)

    def lambda_w(self, w: float) -> float:
        return self.jw(w) * self.distr.function(w)

    def jw(self, w):
        jw = np.zeros_like(w, dtype=float)
        for b in self.sds:
            jw += np.where(w >= 0., b.function(w), -b.function(-w))
        return jw

    def get_star_parameters(self) -> tuple[NDArray, NDArray]:
        return self._zeropoints, np.diag(self._freqencies)

    def base_function(self, k: int, t: NDArray) -> NDArray:
        return np.exp(-1.0j * self._freqencies[k] * t)




class LogFourier(DiscretizationSolver):

    def __init__(
        self,
        sds: list[SpectralDensity],
        distr: BoseEinstein,
        cutoff_frequency: float,
        n: int,
        base: float = 10.0,
        minimum_frequency: float | None = None,
        _vibrations: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        self.distr = distr
        self.omega = cutoff_frequency
        self.minimum_frequency = minimum_frequency
        self.n = n
        self.base = base
        self.sds = sds
        if _vibrations is None:
            _vibrations = []

        frequencies = []
        zeropoints = []

        if sds and self.n > 0:
            freq_space = self._log_space()
            diff_w = freq_space[1:] - freq_space[:-1]
            diff_w = np.concatenate(
                ([freq_space[0]], diff_w, [self.omega - freq_space[-1]]))
            dw = (diff_w[1:] + diff_w[:-1]) / 2.0
            print('[debug] freq_space', [value(f, 'energy') for f in freq_space], flush=True)
            print('[debug] Delta w', [value(_d, 'energy') for _d in dw], flush=True)
            frequencies += list(freq_space)
            zeropoints += list(
                np.sqrt(self.jw(freq_space) * distr.function(freq_space) * dw))

        for _w, _g in _vibrations:
            frequencies.append(_w)
            zeropoints.append(_g * np.sqrt(distr.function(_w)))

        if distr.beta is not None:
            if sds and self.n > 0:
                neg_freq = -freq_space[::-1]
                rev_dw = dw[::-1]
                derivatives_neg = list(neg_freq)
                zeropoints_neg = list(
                    np.sqrt(
                        self.jw(neg_freq) * distr.function(neg_freq) * rev_dw))
                frequencies += derivatives_neg
                zeropoints += zeropoints_neg

            for _w, _g in _vibrations:
                frequencies.append(-_w)
                zeropoints.append(-_g * np.sqrt(distr.function(-_w)))

        self._freqencies = np.array(frequencies)
        self._zeropoints = np.array(zeropoints)

        return

    def get_star_parameters(self) -> tuple[NDArray, NDArray]:
        return self._zeropoints, np.diag(self._freqencies)

    def jw(self, w):
        jw = np.zeros_like(w, dtype=float)
        for b in self.sds:
            jw += np.where(w >= 0., b.function(w), -b.function(-w))
        return jw

    def _log_space(self):
        if self.minimum_frequency is None:
            minimum_frequency = self.omega / 1e4  # Use 1/10000 of the cutoff frequency to avoid high stiffness.
        else:
            minimum_frequency = self.minimum_frequency
        freq_space = np.logspace(np.log(minimum_frequency) / np.log(self.base),
                                 np.log(self.omega) / np.log(self.base),
                                 num=self.n,
                                 base=self.base,
                                 endpoint=False)
        return freq_space

    def base_function(self, k: int, t: NDArray) -> NDArray:
        return np.exp(-1.0j * self._freqencies[k] * t)


class Fourier(DiscretizationSolver):

    def __init__(
        self,
        sds: list[SpectralDensity],
        distr: BoseEinstein,
        cutoff_frequency: float,
        n: int,
        shift: bool = True,
        zero_derivative: Optional[float] = None,
        _vibrations: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        self.distr = distr
        self.omega = cutoff_frequency
        self.n = n
        self.sds = sds
        if _vibrations is None:
            _vibrations = []

        frequencies = []
        zeropoints = []

        if sds and self.n > 0:
            if shift:
                freq_space = self._shift_freq_space()
            else:
                freq_space = self._freq_space()
            dw = self.omega / self.n
            frequencies += list(freq_space)
            zeropoints += list(
                np.sqrt(self.jw(freq_space) * distr.function(freq_space) * dw))

        for _w, _g in _vibrations:
            frequencies.append(_w)
            zeropoints.append(_g * np.sqrt(distr.function(_w)))

        if distr.beta is not None:
            if sds and self.n > 0:
                neg_freq = -freq_space[::-1]
                derivatives_neg = list(neg_freq)
                zeropoints_neg = list(
                    np.sqrt(self.jw(neg_freq) * distr.function(neg_freq) * dw))
                frequencies += derivatives_neg
                zeropoints += zeropoints_neg

            for _w, _g in _vibrations:
                frequencies.append(-_w)
                zeropoints.append(-_g * np.sqrt(distr.function(-_w)))

            if not shift and zero_derivative is not None:
                d_z = np.array([[0.0]])
                z_z = np.sqrt(2.0 * zero_derivative * dw / distr.beta) / 2.0
                frequencies = [d_z] + frequencies
                zeropoints = [z_z] + zeropoints

        self._freqencies = np.array(frequencies)
        self._zeropoints = np.array(zeropoints)

        return

    def get_star_parameters(self) -> tuple[NDArray, NDArray]:
        return self._zeropoints, np.diag(self._freqencies)

    def jw(self, w):
        jw = np.zeros_like(w, dtype=float)
        for b in self.sds:
            jw += np.where(w >= 0., b.function(w), -b.function(-w))
        return jw

    def _freq_space(self):
        return np.linspace(0, self.omega, num=(self.n + 2),
                           endpoint=True)[1:-1]

    def _shift_freq_space(self):
        half_dw = self.omega / (2 * self.n)
        freq_space = np.linspace(half_dw,
                                 self.omega - half_dw,
                                 num=self.n,
                                 endpoint=True)
        return freq_space

    def base_function(self, k: int, t: NDArray) -> NDArray:
        return np.exp(-1.0j * self._freqencies[k] * t)


class Chebyshev(DiscretizationSolver):

    def __init__(
        self,
        w: NDArray,
        j: NDArray,
        beta: float | None,
        n_max: int,
        cutoff_frequency: float | None,
        absorb_rate: float | None = None,
    ) -> None:

        super().__init__(w, j, beta, n_max)
        if absorb_rate is not None:
            assert absorb_rate > 0.0
        self.absorb_rate = absorb_rate
        if cutoff_frequency is None:
            cutoff_frequency = w[-1]
        self.cutoff_frequency = cutoff_frequency

        d_mat = self._gen_d_mat()
        eigenvals, u_mat = linalg.eigh(d_mat)
        self.singular_values = np.sqrt(np.abs(eigenvals.real))
        # keep for the base function
        self.u_mat = u_mat
        return

    def get_star_parameters(self) -> tuple[NDArray, NDArray]:
        #print(self.derivatives.shape, flush=True)
        u_mat = self.u_mat
        c_mat = self._gen_c_mat()  # type: NDArray
        c_mat_v = u_mat.T.conj() @ c_mat @ u_mat
        return self.singular_values, 1.0j * c_mat_v.T

    def base_function(self, k: int, t: NDArray) -> NDArray:
        ut = np.array([self._time_basis(i)(t) for i in range(self.n_max)])
        return ut @ self.u_mat[:, k].conj()

    def _freq_basis(self, k: int):
        if k == 0:

            def _eta0(x: NDArray):
                return np.ones_like(x)

            return _eta0
        else:

            def _etak(x: NDArray):
                return chebyt(k)(x)

            return _etak

    def _time_basis(self, k: int):
        if k == 0:

            def _u0(t: NDArray):
                return j0(self.cutoff_frequency * t)

            return _u0
        else:

            def _uk(t: NDArray):
                return jv(k, self.cutoff_frequency * t) * 2.0 * (-1.0j)**k

            return _uk

    def _gen_d_mat(self):
        """Integrals of J_beta(Omega x) eta_i(x) eta_j(x).
        """
        omega = self.cutoff_frequency
        x_space = self.frequency_space / omega
        # print(x_space)
        eta_k_space = [self._freq_basis(k)(x_space) for k in range(self.n_max)]
        # import matplotlib.pyplot as plt
        # plt.close()
        # for i in range(self.n_max):
        #     plt.plot(x_space, np.abs(eta_k_space[i]))
        # plt.show()

        ans = np.zeros((self.n_max, self.n_max), dtype=np.complex128)
        for i in range(self.n_max):
            for j in range(i + 1):
                d_ij = omega * simpson(
                    self.tsd_space * eta_k_space[i] * eta_k_space[j].conj(),
                    x=x_space,
                )
                ans[i, j] = d_ij
                if i != j:
                    ans[j, i] = d_ij.conj()
        return ans

    def _gen_c_mat(self):
        """Derivatives connetions bewteen u_k(t).
        """
        n = max(2, self.n_max)
        ans = np.zeros((n, n), dtype=np.complex128)
        for i in range(n - 1):
            ans[i, i + 1] = 0.5
            ans[i + 1, i] = -0.5
        ans[1, 0] = -1
        c_mat_j = ans[:self.n_max, :self.n_max]
        if self.n_max >= 2 and self.absorb_rate is not None:
            c_mat_j[-2, -1] = 0.0
            c_mat_j[-1, -1] = -self.absorb_rate

        coeff = np.array([2.0 * (-1.0j)**i for i in range(n)])
        coeff[0] = 1.0
        coeff_diag = np.diag(coeff)
        inv_coeff_diag = np.diag(1.0 / coeff)
        c_mat_u = coeff_diag @ c_mat_j @ inv_coeff_diag
        # print('Starting block:\n', c_mat_u[:3, :3] / 1j * 2, flush=True)
        # print('Terminating block:\n', c_mat_u[-3:, -3:] / 1j * 2, flush=True)
        return c_mat_u * self.cutoff_frequency


'''
class LiftedChebyshev(DiscretizationSolver):

    def __init__(self,
                 sds: list[SpectralDensity],
                 beta: None,
                 n: int,
                 max_frequency: float,
                 min_frequency: float = 0.0,
                 _vibrations=None) -> None:
        self.beta = beta
        self.omega = max_frequency - min_frequency
        self.mean_omega = (max_frequency + min_frequency) / 2.0
        self.n = n
        self.sds = sds
        if _vibrations is None:
            _vibrations = []
        self._vibrations = _vibrations  # type: list[tuple[float, float]]

        self.eigenvals, self.u_mat = self.eigen_pair()

        # u0 = np.array([self._u(k)(0) for k in range(self.n)])
        # print(u0)
        self.v0 = self.u_mat[0]
        # print(self.v0)
        # print(self.eigenvals)
        self.derivatives = self.derivative_mat
        self.zeropoints = self.v0 * np.sqrt(abs(self.eigenvals))

        return

    def get_star_parameters(self) -> tuple[NDArray, NDArray]:
        #print(self.derivatives.shape, flush=True)
        return self.zeropoints, self.derivatives

    def autocorrelation(self, t: float) -> complex:
        # ans = np.zeros_like(t, dtype=complex)
        ut = np.array([self._u(i)(t) for i in range(self.n)])
        ans = self.v0.conj() * self.eigenvals * (self.u_mat @ ut)
        return (np.sum(ans))

    def _sign(self, v: float, underflow) -> Literal[1, -1, 0]:
        return 1 if v > underflow else -1 if v < -underflow else 0

    def _d_mat(self):
        ans = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(i + 1):
                d_ij = self._d_ij(i, j)
                ans[i, j] = d_ij
                if i != j:
                    ans[j, i] = d_ij
        return ans

    def jw(self, w):
        jw = np.zeros_like(w, dtype=float)
        for b in self.sds:
            jw += np.where(w >= 0., b.function(w), -b.function(-w))
        return jw

    def _d_ij(self, i: int, j: int) -> float:

        def _int(w):
            ans = self.jw(w) * self.be_function(
                self.beta, w) * self._eta(i)(w) * self._eta(j)(w)
            return ans

        # Calculate real part
        ans = 0.0
        # w_space = np.logspace
        if self.sds:
            eps = 1e-7
            num = 10000

            # w_p = np.logspace(np.log2(eps),
            #                   np.log2(self.omega),
            #                   num=num,
            #                   base=2,
            #                   endpoint=True)
            w_p = np.linspace(eps, self.omega, num=num, endpoint=True)
            if _INCLUDE_POSITIVE:
                ans += simpson(_int(w_p), x=w_p)
                # ans += quad(_int, 0, self.omega)[0]
            if _INCLUDE_NEGATIVE:
                w_n = -w_p[::-1]
                ans += simpson(_int(w_n), x=w_n)
                # ans += quad(_int, -self.omega, 0)[0]
        for _w, _g in self._vibrations:
            assert abs(_w) < self.omega
            if _INCLUDE_POSITIVE:
                ans += _g**2 * self.be_function(_w) * self._eta(i)(
                    _w) * self._eta(j)(_w)
            if _INCLUDE_NEGATIVE:
                ans += -_g**2 * self.be_function(-_w) * self._eta(i)(
                    -_w) * self._eta(j)(-_w)

        return ans

    def _eta(self, k: int):

        def _eta_k(x: NDArray):
            return chebyt(k)(x)

        return _eta_k

    def _u(self, k: int):
        if k == 0:

            def _u0(t: NDArray):
                return j0(self.omega * t) * np.exp(-1.0j * self.mean_omega * t)

            return _u0
        elif k == 1:

            def _u1(t: NDArray):
                return -2.0j * j1(self.omega * t) * np.exp(
                    -1.0j * self.mean_omega * t)

            return _u1
        else:

            def _uk(t: NDArray):
                return 2.0 * (-1.0j)**k * jv(k, self.omega * t) * np.exp(
                    -1.0j * self.mean_omega * t)

            return _uk

    @property
    def c_mat(self):
        """(1j *) Derivatives connetions bewteen u_k(t)."""
        ans = np.zeros((self.n, self.n), dtype=float)
        o = self.omega / 2.0
        for i in range(self.n - 1):
            ans[i, i + 1] = o
        for i in range(1, self.n):
            ans[i, i - 1] = o
        ans[1, 0] = self.omega
        return ans

    @property
    def derivative_mat(self):
        """(1j *) Derivatives connetions bewteen v_k(t)."""
        c = self.c_mat
        u = self.u_mat
        return u.T.conj() @ c @ u

    def eigen_pair(self):
        d = self._d_mat()
        w, u = linalg.eigh(d)
        lst = [(w[i], u[:, i]) for i in range(self.n)]

        def key(item):
            return abs(item[1][0]**2 * item[0])

        lst = sorted(lst, key=key, reverse=True)
        # lst = sorted(lst, key=lambda item: abs(item[1][0]), reverse=True)
        # lst = sorted(lst, key=lambda item: abs(item[0]), reverse=True)

        sorted_w = np.zeros_like(w)
        sorted_u = np.zeros_like(u)
        for i, (wi, ui) in enumerate(lst):
            sorted_w[i] = wi
            sorted_u[:, i] = ui if ui[0] > 0 else -ui
        assert np.allclose(np.diag(sorted_w), sorted_u.T.conj() @ d @ sorted_u)
        return sorted_w, sorted_u

    def reduce(self, underflow):
        signs = [self._sign(v, underflow) for v in self.eigenvals]
        for nn, s in enumerate(signs):
            if s == 0:
                break
        self.derivatives = self.derivatives[:nn, :nn]
        self.zeropoints = self.zeropoints[:nn]
        return
'''

if __name__ == '__main__':
    from tenso.bath.sd import OhmicExp, Drude
    import matplotlib.pyplot as plt
    from tenso.libs.quantity import Quantity as __
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
    print(__(1 / 300, '/K').au * __(1000, '/cm').au)
    beta = __(1 / 300, '/K').au * __(1000, '/cm').au
    #spd = OhmicExp(0.2, 0.1)
    spd = Drude(1, 0.1)
    t_fs = np.linspace(0, 500, num=1000)
    t_unit = __(1, 'fs').au * __(1000, '/cm').au
    t_space = t_fs * t_unit
    # Exact solution
    c_space = np.array([spd.autocorrelation(t, beta) for t in t_space])
    plt.plot(t_fs, c_space.real, 'r-', label='Re. (Exact)', lw=1)
    plt.plot(t_fs, c_space.imag, 'b-', label='Im. (Exact)', lw=1)

    # ----------------
    sb = StarBosons()
    n_chev = 31
    sb.add_spectral_densities(
        [spd],
        beta=beta,
        cutoff=1.4,
        n=n_chev,
        # method='Fourier',
        method='Chebyshev',
        absorb_rate=None)
    d = sb.frequencies
    print(sb.k_max)
    dmat = np.zeros((sb.k_max, sb.k_max), dtype=complex)
    for (i, j), v in d.items():
        dmat[i, j] = v
    c_space = np.array([sb.autocorrelation(t) for t in t_space])
    plt.plot(t_fs,
             c_space.real,
             'r-.',
             label=f'Re. (Chebyshev (n={n_chev}))',
             lw=1)
    plt.plot(t_fs,
             c_space.imag,
             'b-.',
             label=f'Im. (Chebyshev (n={n_chev}))',
             lw=1)

    plt.legend()
    plt.show()

    plt.close()
    sb.diagonalize()
    sb.filter(1.0e-7)
    diag_freq = np.array(
        [sb.frequencies.get((i, i), 0.0) for i in range(sb.k_max)])
    sorted_idx = sorted(range(sb.k_max), key=lambda i: (diag_freq[i].imag))
    diag_freq = np.array([diag_freq[i] for i in sorted_idx])
    positive_freq = diag_freq[diag_freq.imag > 1e-7]
    positive_idx = np.where(diag_freq.imag > 1e-7)[0]
    negative_freq = diag_freq[diag_freq.imag < -1e-7]
    decay_idx = np.where(diag_freq.real < -1e-7)[0]
    decay_freq = diag_freq[diag_freq.real < -1e-7]
    negative_idx = np.where(diag_freq.imag < -1e-7)[0]
    coupling = np.array([sb.couplings[i] for i in sorted_idx])
    ccoupling = np.array([sb.conj_couplings[i] for i in sorted_idx])
    plt.plot(coupling.real, 'r>', fillstyle='none', label='Couplings')
    plt.plot(ccoupling.real, 'b<', fillstyle='none', label='Conj Couplings')
    plt.plot(decay_idx,
             decay_freq.real,
             'kx',
             fillstyle='none',
             label='Decay frequencies')
    plt.plot(positive_idx,
             abs(positive_freq.imag),
             'ro',
             fillstyle='none',
             label='Forward frequencies')
    plt.plot(negative_idx,
             -abs(negative_freq.imag),
             'bo',
             fillstyle='none',
             label='Backward frequencies')
    # print(diag_freq)
    for k, v in enumerate(diag_freq):
        print(k, ': ', v)
    plt.legend()
    plt.show()
    # ----------------
    # sb = StarBosons()
    # sb.add_spectral_densities(
    #     [spd],
    #     beta=beta,
    #     cutoff=0.6,
    #     n=15,
    #     method='Fourier',
    #     shift_frequency=True,
    # )
    # c_space = np.array([sb.autocorrelation(t) for t in t_space])
    # plt.plot(t_fs, (c_space).real, 'r-.', label='Re. (Fourier (n=30))', lw=1)
    # plt.plot(t_fs, c_space.imag, 'b-.', label='Im. (Fourier (n=30))', lw=1)
    # plt.legend()
    # plt.xlabel('Time [fs]')
    # plt.ylabel('$C(t)$')
    # plt.show()

    #print(sb)
