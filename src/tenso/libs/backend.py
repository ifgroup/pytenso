# coding: utf-8
r"""Backend for accelerated array-operations.
"""

import math
from typing import Callable, Iterable
from numpy.typing import ArrayLike, NDArray

import torch as _opt
from torch import linalg as opt_linalg
from torch import sqrt as opt_sqrt
from torch import maximum as opt_maximum
from torch import minimum as opt_minimum
from torch import abs as opt_abs
from torch import zeros_like as opt_zeros_like
from torch import ones_like as opt_ones_like
from torch import matrix_exp as opt_matrix_exp
from torch import save as opt_save
from torch import load as opt_load
# import multiprocessing as mp

import torchdiffeq
try:
    from scikits.odes.odeint import odeint as sundials_odeint
except ImportError:
    sundials_odeint = None

# N_PROCESS = mp.cpu_count()
# import opt_einsum as oe
MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10
PI = math.pi

# PyTorch package settings
_opt.set_grad_enabled(False)  # disable autograd by defalt
DOUBLE_PRECISION = True
FORCE_CPU = True
ON_DEVICE_EIGEN_SOLVER = False
if FORCE_CPU:
    opt_device = 'cpu'
elif _opt.cuda.is_available():
    DOUBLE_PRECISION = False
    opt_device = 'cuda'
elif _opt.backends.mps.is_available():
    DOUBLE_PRECISION = False
    opt_device = 'mps'
else:
    opt_device = 'cpu'

# GPU settings
if DOUBLE_PRECISION:
    opt_dtype = _opt.complex128
else:
    opt_dtype = _opt.complex64
OptArray = _opt.Tensor


def opt_to_numpy(array: OptArray) -> NDArray:
    return array.cpu().numpy()


def opt_array(array: ArrayLike) -> OptArray:
    ans = _opt.tensor(array, dtype=opt_dtype, device=opt_device)
    return ans


def opt_zeros(shape: list[int]) -> OptArray:
    return _opt.zeros(shape, dtype=opt_dtype, device=opt_device)


def opt_cat(tensors: list[OptArray]) -> OptArray:
    return _opt.cat(tensors)


def opt_stack(tensors: list[OptArray] | tuple[OptArray, ...]) -> OptArray:
    return _opt.stack(tensors, dim=0)


def opt_split(tensors: OptArray, size_list: list[int]) -> list[OptArray]:
    return list(_opt.split(tensors, size_list))


def opt_einsum(*args) -> OptArray:
    """Currently wrapper for torch.einsum without optimizing contraction order."""
    return _opt.einsum(*args)


def opt_sum(array: OptArray, dim: int) -> OptArray:
    return _opt.sum(array, dim=dim)


def opt_tensordot(a: OptArray, b: OptArray,
                  axes: tuple[list[int], list[int]]) -> OptArray:
    return _opt.tensordot(a, b, dims=axes)


def opt_svd(a: OptArray) -> tuple[OptArray, OptArray, OptArray]:
    """
    Perform singular value decomposition (SVD) on the input array without full matrices.

    Args:
        a (OptArray): The input array.

    Returns:
        tuple[OptArray, OptArray, OptArray]: A tuple containing the left singular vectors,
        singular values, and right singular vectors.
        Note that the singular values are of the real type.
    """
    if (a != a).any():
        raise ValueError('NaN detected in the input array.')

    if not ON_DEVICE_EIGEN_SOLVER:
        a = a.cpu()

    u, s, vh = _opt.linalg.svd(a, full_matrices=False)

    if not ON_DEVICE_EIGEN_SOLVER:
        u = u.to(device=opt_device)
        s = s.to(device=opt_device)
        vh = vh.to(device=opt_device)

    return u, s, vh


def opt_odeint(func: Callable[[float, OptArray], OptArray],
               t0: float,
               y0: OptArray,
               dt: float,
               atol: float,
               rtol: float,
               method: str = 'dopri5') -> OptArray:
    """Avaliable method:
    - Home-made integrators:
        - `iterX` Taylor series up to `X`-th order.
        - `rk4` Fourth-order Runge-Kutta with 3/8 rule.
    - Adaptive-step from `torchdiffeq`:
        - `dopri8` Runge-Kutta 7(8) of Dormand-Prince-Shampine
        - `dopri5` Runge-Kutta 4(5) of Dormand-Prince.
        - `bosh3` Runge-Kutta 2(3) of Bogacki-Shampine
        - `adaptive_heun` Runge-Kutta 1(2)
    - Fixed-step `torchdiffeq`:
        - `euler` Euler method.
        - `midpoint` Midpoint method.
        - `explicit_adams` Explicit Adams.
        - `implicit_adams` Implicit Adams.
    - Scikit.odes/SUNDIALS compatable method (using numpy.array): 
        - 'cvode' CVODE
        - 'bdf' Backward Differentiation Formula
        - 'admo' Adams-Moulton
        - 'rk8' Runge-Kutta 7(8)
        - 'rk5' Runge-Kutta 4(5)
    """

    if method == 'rk4':
        # Fourth-order Runge-Kutta with 3/8 rule
        k1 = func(t0, y0) * dt
        k2 = func(t0 + dt / 3.0, y0 + k1 / 3.0) * dt
        k3 = func(t0 + dt * 2.0 / 3.0, y0 - k1 / 3.0 + k2) * dt
        k4 = func(t0 + dt, y0 + k1 - k2 + k3) * dt
        y1 = y0 + (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0
    elif method.startswith('iter'):
        # Taylor series up to Xth order as in `iterX` for linear ODE
        iter_n = int(method[4:])
        cumm = y0
        yn = y0
        for n in range(1, iter_n + 1):
            yn = func(t0, yn) * dt / n
            cumm += yn
        y1 = cumm
    elif method in [
            'dopri8', 'dopri5', 'bosh3', 'adaptive_heun', 'euler', 'midpoint',
            'explicit_adams', 'implicit_adams'
    ]:
        t = opt_array([t0, t0 + dt]).real
        solution = torchdiffeq.odeint(func,
                                      y0,
                                      t,
                                      method=method,
                                      rtol=rtol,
                                      atol=atol)
        y1 = solution[1]
    elif method in ['bdf', 'admo', 'rk5', 'rk8', 'cvode']:
        if sundials_odeint is None:
            raise RuntimeError(
                f'Unable to import `scikits.odes` to use SUNDIALS method `{method}`.'
            )
        shape = y0.shape

        def rhseqn(t, _y, _ydot):
            # print(_y)
            tdot = func(t, opt_array(_y).reshape(shape)).flatten()
            for i, ti in enumerate(tdot):
                _ydot[i] = ti
            return

        _y0 = y0.flatten()
        _tout = [t0, t0 + dt]
        output = sundials_odeint(rhseqn, _tout, _y0, method=method)
        _yout = output.values.y
        if _yout.shape[0] != 2:
            raise RuntimeError(
                f'SUNDIALS failed to integrate with method `{method}`.')
        _y1 = _yout[1, :]
        y1 = opt_array(_y1.reshape(shape))
    else:
        raise NotImplementedError(f'Unsupported method `{method}`.')
    return y1


def opt_pinv(a: OptArray, atol) -> OptArray:
    return _opt.linalg.pinv(a, atol=atol)


def opt_inv(a: OptArray) -> OptArray:
    return _opt.linalg.inv(a)


# @_opt.compile
def opt_transform(op: OptArray, tensor: OptArray, op_ax: int, tensor_ax: int):
    dotted = opt_tensordot(tensor, op, axes=([tensor_ax], [op_ax]))
    return dotted.movedim(-1, tensor_ax)


# @_opt.compile
def opt_multitransform(op_dict: dict[int, OptArray],
                       tensor: OptArray) -> OptArray:
    # ax_list = list(sorted(op_dict.keys(),
    #                       key=(lambda ax: tensor.shape[ax])))
    # mat_list = [op_dict[ax] for ax in ax_list]
    # ans = tensor
    # for ax, mat in zip(ax_list, mat_list):
    #     ans = opt_transform(mat, ans, 1, ax)

    ans = tensor
    for ax, mat in op_dict.items():
        ans = opt_transform(mat, ans, 1, ax)

    return ans


def opt_eye(dim1: int, dim2: int | None = None) -> OptArray:
    if dim2 is None:
        dim2 = dim1
    return _opt.eye(dim1, dim2, dtype=opt_dtype, device=opt_device)


# @_opt.compile
def opt_trace(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
    """Complex conjugate not included
    """
    dim1 = tensor1.shape[ax]
    dim2 = tensor2.shape[ax]

    left = tensor1.moveaxis(ax, 0).reshape((dim1, -1))
    right = tensor2.moveaxis(ax, -1).reshape((-1, dim2))
    return left @ right


# @_opt.compile
def opt_inner_product(tensor1: OptArray, tensor2: OptArray) -> complex:
    left = tensor1.flatten()
    right = tensor2.flatten()
    return (left @ right).item()


# def opt_unfold(tensor: OptArray, ax: int) -> OptArray:
#     dim = tensor.shape[ax]
#     ans = tensor.moveaxis(ax, 0).reshape((dim, -1))
#     return ans

# def opt_fold(vectors: OptArray, shape: list[int], ax: int):
#     dim = shape[ax]
#     _shape = [dim] + [n for i, n in enumerate(shape) if i != ax]
#     assert dim == vectors.shape[0]
#     ans = vectors.reshape(_shape).moveaxis(0, ax)
#     return ans

# def opt_transform(tensor: OptArray, ax: int, op: OptArray) -> OptArray:
#     """Tensor-matrix contraction that keeps the indices convension of tensor.
#     """
#     shape = list(tensor.shape)
#     ans_vectors = op @ opt_unfold(tensor, ax)
#     return opt_fold(ans_vectors, shape, ax)

# def opt_trace(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
#     assert tensor1.shape == tensor2.shape
#     assert 0 <= ax < tensor1.ndim
#     vectors1 = opt_unfold(tensor1, ax)
#     vectors2 = opt_unfold(tensor2, ax).transpose()
#     return vectors1 @ vectors2
