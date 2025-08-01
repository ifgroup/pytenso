"""Computing the chain map coefficients used in the (T-)TEDOPA algorithm.

Note: This code does not pass the tests yet, and is not ready for production.
"""


import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable
from tenso.libs.quantity import Quantity as __

from scipy.integrate import simpson, quad


class Tedopa:
    underflow = 1.0e-14

    @staticmethod
    def be_function(beta: float, w: NDArray) -> NDArray:
        return 0.5 + 0.5 / np.tanh(0.5 * beta * w)

    def __init__(self, w: NDArray, j: NDArray, beta: Optional[None], n_max: int):
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
        w = np.array(w)
        j = np.array(j)
        # Remove the zero-frequency component.
        if abs(w[0]) < self.underflow:
            w = w[1:]
            j = j[1:]

        if beta is None:
            self.frequency = w
            self.sd = j
            self.tsd = self.sd
        else:
            self.frequency = np.concatenate((-w[::-1], w))
            self.sd = np.concatenate((-j[::-1], j))
            self.tsd = self.be_function(beta, self.frequency) * self.sd
        self.beta = beta
        self.n_max = n_max

        # Coefficents for the orthogonal polynomials.
        # ip[n] = <p_n, p_n>
        self.ip = np.zeros(n_max)
        # ipx[n] = <x * p_n, p_n>
        self.ipx = np.zeros(n_max)
        # alpha[n] = <x * p_n, p_n> / <p_n, p_n>
        self.alpha = np.zeros(n_max)
        # beta[n] = <p_n, p_n> / <p_{n-1}, p_{n-1}>
        self.beta = np.zeros(n_max)
        self.generate_polynomials()
        return

    def c0(self) -> float:
        """
        Compute the zeroth-order coefficient.
        """
        return np.sqrt(self.ip[0])

    def chain_frequency(self) -> NDArray:
        """
        Compute the chain frequencies.
        """
        return self.alpha

    def chain_coupling(self) -> NDArray:
        """
        Compute the chain couplings.
        """
        return np.sqrt(self.beta)

    def chain_matrix(self) -> NDArray:
        """
        Compute the chain matrix.
        """
        return np.diag(self.chain_coupling()[1:], k=1) + \
            np.diag(self.chain_coupling()[1:], k=-1) + \
            np.diag(self.chain_frequency())

    def star_parameters(self) -> NDArray:
        e, v = np.linalg.eigh(self.chain_matrix())
        return e, self.c0() * v[:, 0]

    def int(self, f: NDArray) -> float:
        """
        Integrate a function with respect to the spectral density.
        """
        return simpson(f * self.tsd, self.frequency)

    def generate_polynomials(self) -> None:
        """
        Generate the orthogonal polynomials.
        """
        one = np.ones_like(self.frequency)
        # base case
        p_n2 = np.zeros_like(self.frequency)
        p_n1 = np.ones_like(self.frequency)
        self.ip[0] = self.int(p_n1 * p_n1)
        self.ipx[0] = self.int(self.frequency * p_n1 * p_n1)
        self.alpha[0] = self.ipx[0] / self.ip[0]
        self.beta[0] = 0.0
        for n in range(1, self.n_max):
            p_n = (self.frequency - self.alpha[n - 1] * one) * p_n1 - \
                self.beta[n - 1] * p_n2
            self.ip[n] = self.int(p_n * p_n)
            self.ipx[n] = self.int(self.frequency * p_n * p_n)
            self.alpha[n] = self.ipx[n] / self.ip[n]
            self.beta[n] = self.ip[n] / self.ip[n - 1]
            p_n2 = p_n1
            p_n1 = p_n
        return


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    energy_unit = __(1000, '/cm').au
    freq_max = 45
    n_tedopa = 100
    beta = __(1 / 300.0, '/K').au * energy_unit
    print(beta)
    # beta = None
    re = 0.2
    gamma = 0.01
    print(gamma, re)

    print('Δ[J(w)/w]')
    for freq_max in [500]:
        n_space = int(freq_max * 10000)
        freq_space = (np.linspace(0, freq_max, n_space))[1:]
        # Drude-Lorentz spectral density
        # j = re / (2.0 * np.pi) * freq_space * \
        #     gamma / (freq_space**2 + gamma**2)
        # Ohmic spectral density
        j = re * 2 / np.sqrt(gamma * np.pi) * freq_space * \
            np.exp(-freq_space / gamma)
        tedopa_solver = Tedopa(freq_space, j, beta, n_tedopa)

        ww, gg = tedopa_solver.star_parameters()
        lc = simpson(tedopa_solver.tsd / tedopa_solver.frequency,
                     tedopa_solver.frequency)
        ld = sum(gg**2 / ww)
        d = lc - ld
        print(f'N={n_tedopa} | Fmax={freq_max} | {lc:.4} | {d:.4}')
        print(simpson(j/freq_space, freq_space))
        plt.plot(tedopa_solver.frequency,
                 tedopa_solver.sd, '-')
        plt.plot(ww, gg**2, 'x')
        for ni, (wi, gi) in enumerate(zip(ww, gg)):
            plt.text(wi, gi**2,
                     str(ni), color="black", fontsize=12)
        # plt.xlim(-3, 3)
        plt.show()
