# coding: utf-8
r"""A simple implementation of a discrete variable representation integrator
based on the implementation found in the Heidelberg MCTDH documentation. 

References
----------
.. [1] http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
"""

from typing import Callable, Optional

from tenso.libs.backend import NDArray
import numpy as np


class DiscreteVariationalRepresentation:
    """Abstract class for any discrete variable representation basis.
    """
    def __init__(self, num: int) -> None:
        """General constructor for a DVR basis.
        """
        self.n = num
        self.grid_points = [None] * num
        self.dvr2fbr_mat = None  # type: Optional[NDArray]
        return

    @property
    def q_mat(self) -> NDArray:
        """Get the position operator in DVR basis.

        :returns: Position operator, diagonal in the grid points
        :rtype: numpy array
        """
        return np.diag(self.grid_points)

    @property
    def dq_mat(self) -> NDArray:
        pass

    @property
    def dq2_mat(self) -> NDArray:
        pass

    @property
    def creation_mat(self) -> NDArray:
        """Get the creation operator in the DVR basis

        :returns: the creation operator
        :rtype: numpy array
        """
        return (self.q_mat - self.dq_mat) / np.sqrt(2.0)

    @property
    def annihilation_mat(self) -> NDArray:
        """Get the annihilation operator in the DVR basis

        :returns: the annihilation operator
        :rtype: numpy array
        """
        return (self.q_mat + self.dq_mat) / np.sqrt(2.0)

    @property
    def fock2dvr_mat(self) -> NDArray:
        _, u = np.linalg.eigh(self.numberer_mat)
        ans = np.array(u, dtype=complex)
        return ans

    def eigen2dvr_mat(self, ham) -> NDArray:
        _, u = np.linalg.eigh(ham)
        ans = np.array(u, dtype=complex)
        return ans

    def fbr_func(self, i: int) -> Callable[[NDArray], NDArray]:
        """`i`-th FBR basis function."""
        pass

    @property
    def numberer_mat(self) -> NDArray:
        q, dq = self.q_mat, self.dq_mat
        q2 = q**2
        if self.dq2_mat is not None:
            dq2 = self.dq2_mat
        else:
            dq2 = dq @ dq
        eye = np.identity(self.n)
        return 0.5 * (q2 - dq2 - eye)


class SincDVR(DiscreteVariationalRepresentation):
    """ Class for performing the universal Sinc DVR basis of Colbert and Miller (1982)
    """

    def __init__(self, start: float, stop: float, num: int) -> None:
        """Constructor for the Sinc DVR

        :param start: One end of the basis extent
        :type start: float

        :param stop: Other end of the basis extent
        :type stop: float

        :param num: Number of grid points for the basis
        :type num: int
        """
        self.length = abs(stop - start)
        self.grid_points = np.array(
            [start + i * self.length / (num + 1) for i in range(1, num + 1)],
            dtype=complex)
        self.n = num
        self.length = abs(start - stop)
        self.delta = self.length / (num + 1)
        self.grid_points = np.array(
            [start + i * self.delta for i in range(1, num + 1)], dtype=complex)

    @property
    def dq_mat(self) -> NDArray:
        """Get the volume element `d/dq` matrix, alternately -i times momentum, in the DVR basis

        :returns: The volume element `d/dq` matrix `(<alpha| d/dq |beta>)`
        :rtype: array[complex]"""
        _i = np.arange(1, self.n + 1, dtype=complex)[:, np.newaxis]
        _j = np.arange(1, self.n + 1, dtype=complex)[np.newaxis, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ans = np.where(_i == _j, 0.0,
                           (-1.0)**(_i - _j) / (_i - _j) / self.delta)
        return ans

    @property
    def dq2_mat(self) -> NDArray:
        """Get the square volume element `d2/dq2` matrix, alternately -momentum^2, in the DVR basis

        :returns: The squared volume element `dq2` matrix `(<alpha| d2/dq2 |beta>)`
        :rtype: array[complex]"""
        _i = np.arange(1, self.n + 1, dtype=complex)[:, np.newaxis]
        _j = np.arange(1, self.n + 1, dtype=complex)[np.newaxis, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ans = np.where(
                _i == _j, -np.pi**2 / 3.0 / self.delta**2,
                -2.0 * (-1.0)**(_i - _j) / (_i - _j)**2 / self.delta**2)
        return ans


class SineDVR(DiscreteVariationalRepresentation):
    """ Class for the Sine-DVR
    """
    def __init__(self, start: float, stop: float, num: int) -> None:
        """Constructor for the Sine DVR

        :param start: One end of the basis extent
        :type start: float

        :param stop: Other end of the basis extent
        :type stop: float

        :param num: Number of grid points for the basis
        :type num: int
        """
        self.grid_points = np.linspace(start, stop, num)
        self.n = num
        self.length = abs(start - stop)
        self.grid_points = np.array(
            [start + i * self.length / (num + 1) for i in range(1, num + 1)],
            dtype=complex)
        _i = np.arange(1, self.n + 1)[:, np.newaxis]
        _j = np.arange(1, self.n + 1)[np.newaxis, :]
        dvr2fbr_mat = np.array(np.sqrt(2 / (self.n + 1)) *
                               np.sin(_i * _j * np.pi / (self.n + 1)),
                               dtype=complex)  # type: NDArray
        self.dvr2fbr_mat = dvr2fbr_mat
        return

    @property
    def q_mat(self) -> NDArray:
        """Get the position operator, q in the DVR basis."""
        return np.diag(self.grid_points)

    @property
    def abs_q_mat(self) -> NDArray:
        """Get the absolute value of the position operator, q in the DVR basis."""
        return np.diag(np.abs(self.grid_points))

    @property
    def abs_dq_mat(self) -> NDArray:
        """q in DVR basis."""
        j = np.arange(1, self.n + 1)
        fbr_mat = np.diag(j * np.pi / self.length)
        u = self.dvr2fbr_mat
        return u.T @ fbr_mat @ u

    @property
    def dq_mat(self) -> NDArray:
        """Get the volume element `d/dq` matrix, alternately -i times momentum, in the DVR basis

        :returns: The volume element `d/dq` matrix `(<alpha| d/dq |beta>)`
        :rtype: array[complex]"""
         # fbr_mat = np.zeros((self.n, self.n), dtype=complex)
        l = self.length
        n = self.n
        _i = np.arange(1, n + 1, dtype=int)[:, np.newaxis]
        _j = np.arange(1, n + 1, dtype=int)[np.newaxis, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            fbr_mat = np.where(
                _i == _j, 0.0,
                2.0 * (1.0 - (-1)**(_i + _j)) * _i * _j / (_i**2 - _j**2) / l)
        u = self.dvr2fbr_mat
        return u.T @ fbr_mat @ u

    @property
    def dq2_mat(self) -> NDArray:
        """Get the square volume element `d2/dq2` matrix, alternately `-p^2`, in the DVR basis

        :returns: The squared volume element `dq2` matrix `(<alpha| d2/dq2 |beta>)`
        :rtype: array[complex]"""
        j = np.arange(1, self.n + 1)
        fbr_mat = np.diag(-(j * np.pi / self.length)**2)
        u = self.dvr2fbr_mat
        return u.T @ fbr_mat @ u

    @property
    def t_mat(self):
        """Return the kinetic energy matrix in the DVR basis in internal energy units.
        Returns
        -------
        `(n, n)` np.ndarray
            A 2-d matrix.
        """
        return -0.5 * self.dq2_mat

    @property
    def fock2dvr_mat(self) -> NDArray:
        _, u = np.linalg.eigh(self.numberer_mat)
        ans = np.array(u, dtype=complex)
        # correct the direction according to annihilation_mat
        # subdiag = np.diagonal(u.conj().T @ self.annihilation_mat @ u, offset=1)
        # counter = 0
        # for _n, _d in enumerate(subdiag, start=1):
        #     if _d < 0:
        #         counter += 1
        #     ans[:, _n] = (-1)**(counter) * ans[:, _n]

        # if abs(ans[:, 0].sum()) < 0:
        #     ans = -ans

        # print(abs(ans[:, 0].sum()))
        return ans

    @property
    def height(self):
        l = self.length
        n = self.n
        x = self.grid_points + l / 2
        ans = 0.5 / np.sqrt(l * (n + 1)) * (
            2 * n + 1 - np.sin(np.pi *
                               (2 * n + 1) * x / l) / np.sin(np.pi * x / l))
        return ans

    def fbr_func(self, i: int) -> Callable[[NDArray], NDArray]:
        """`i`-th FBR basis function."""
        l = self.length
        x0 = self.grid_points[0]

        def _func(_x: NDArray) -> NDArray:
            with np.errstate(divide='ignore', invalid='ignore'):
                ans = np.where(
                    np.logical_and(x0 < _x, _x < x0 + l),
                    np.sqrt(2.0 / l) * np.sin((i + 1) * np.pi * (_x - x0) / l),
                    0.0)
            return ans

        return _func

    def fbr2cont(self, vec):
        r"""Transform a vector from FBR to the spatial function.
        """
        assert len(vec) == self.n

        def _psi(x):
            fbr_s = [self.fbr_func(j)(x) * vec[j] for j in range(self.n)]
            return sum(fbr_s)

        return _psi

    def dvr2cont(self, vec):
        vec = self.dvr2fbr_mat @ vec
        psi = self.fbr2cont(vec)
        return psi


