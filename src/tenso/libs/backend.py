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
    """Transform a pytorch tensor to a numpy array

    :param OptArray: Input numpy array
    :type OptArray: OptArray

    :returns: Equivalent numpy array
    :rtype: array
    """
    return array.cpu().numpy()


def opt_array(array: ArrayLike) -> OptArray:
    """Transform an array like to a pytorch tensor

    :param array: Array to transform
    :type array: ArrayLike

    :returns: Equivalent pytorch tensor
    :rtype: :class:`OptArray`
    """
    ans = _opt.tensor(array, dtype=opt_dtype, device=opt_device)
    return ans


def opt_zeros(shape: list[int]) -> OptArray:
    """Get a pytorch tensor with zeros in the given shape (wraps torch.zeros)

    :param shape: List of dimensions of the tensor
    :type shape: list[int]

    :returns: Tensor of zeros in pytorch form
    :rtype: :class:`OptArray`
    """
    return _opt.zeros(shape, dtype=opt_dtype, device=opt_device)


def opt_cat(tensors: list[OptArray]) -> OptArray:
    """Stack the listed tensors along the zeroth dimension leading to a tensor with the same number of dimensions as the input tensors, all of which must be the same size except along the dimension to concatenate (wraps torch.cat)

    :param tensors: Input pytorch tensors to concatenate
    :type tensors: :class:`OptArray`

    :returns: Concatenated tensors
    :rtype: :class:`OptArray`
    """
    return _opt.cat(tensors)


def opt_stack(tensors: list[OptArray] | tuple[OptArray, ...]) -> OptArray:
    """Stack the listed tensors along a new dimension leading to a tensor with one additional dimensions than the input tensors, all of which must be the same size except along the dimension to concatenate (wraps torch.stack)

    :param tensors: Input pytorch tensors to concatenate
    :type tensors: list[:class:`OptArray`] or tuple[:class:`OptArray`]

    :returns: Stacked tensors, with the new dimension being the first dimension
    :rtype: :class: `OptArray`
    """
    return _opt.stack(tensors, dim=0)


def opt_split(tensors: OptArray, size_list: list[int]) -> list[OptArray]:
    """Split a tensor into pieces along dimension zero (wraps torch.split)

    :param tensors: Input pytorch tensor to break up
    :type tensors: :class:`OptArray`

    :returns: A list of the broken up tensors
    :rtype: list[:class:`OptArray`]
    """
    return list(_opt.split(tensors, size_list))


def opt_einsum(*args) -> OptArray:
    """Perform an Einstein summation over the tensors provided (wraps torch.einsum, see documentation for torch.einsum)

    :param args: The required string specification for how to perform the eigensum followed by the tensors to operator on
    :type args: varies

    :returns; Result of the requested tensor contraction
    :rtype: :class:`OptArray`
    """
    return _opt.einsum(*args)


def opt_sum(array: OptArray, dim: int) -> OptArray:
    """Sum over all elements along the given dimension of the tensor (wraps torch.sum)

    :param array: Tensor with elements to sum
    :type array: :class:`OptArray`
    
    :returns: Tensor after summation along requested dimension
    :rtye: :class:`OptArray`
    """
    return _opt.sum(array, dim=dim)


def opt_tensordot(a: OptArray, b: OptArray,
                  axes: tuple[list[int], list[int]]) -> OptArray:
    """Perform a tensor contraction over tensors a and b along the dimensions
    specified by axes

    :param a: first tensor to contract
    :type a: :class:`OptArray`

    :param b: second tensor to contract
    :type b: :class:`OptArray`

    :param axes: Tuple of lists specifying which dimensions of a and b to perform a contraction over
    :type axes: tuble[list[int]]

    :returns: The requested tensor after contraction
    :rtype: :class:`OptArray`
    """
    return _opt.tensordot(a, b, dims=axes)


def opt_svd(a: OptArray) -> tuple[OptArray, OptArray, OptArray]:
    """ Perform singular value decomposition (SVD) on the input array without full matrices.

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
    """Selection of method and parameters for the ordinary differential equation solver
    Avaliable method:
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
    - Scikit.odes/SUNDIALS compatable method (using numpy.array) slow but may handle stiff equations better: 
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
    """Perofrm Moore-Penrose pseudoinverse of the tensor

    :param a: Tensor to invert
    :type a: :class:`OptArray`

    :returns: Tensor pseudoinverse
    :rtype: :class:`OptArray`
    """
    return _opt.linalg.pinv(a, atol=atol)


def opt_inv(a: OptArray) -> OptArray:
    """Invert the given tensor or throws an error (wraps torch.linalg.inv)

    :param a: Tensor to invert
    :type a: :class:`OptArray`

    :returns: Inverted tensor
    :rtype: :class:`OptArray`
    """
    return _opt.linalg.inv(a)


# @_opt.compile
def opt_transform(op: OptArray, tensor: OptArray, op_ax: int, tensor_ax: int):
    """Perform a tensor contraction over the specified axes of input tensors then rearrange the dimensions to place the last dimension at the location of contraction

    :param op: Second tensor in contraction
    :type op: :class:`OptArray`
    
    :param tensor: First tensor in contraction
    :type tensor: :class:`OptArray`

    :param op_ax: Contraction dimension of op
    :type op_ax: integer

    :param tensor_ax: Contraction dimension of tensor
    :type tensor_ax: integer

    :returns: The contracted and rearranged tensor
    :rtype: :class:`OptArray`
    """
    dotted = opt_tensordot(tensor, op, axes=([tensor_ax], [op_ax]))
    return dotted.movedim(-1, tensor_ax)


# @_opt.compile
def opt_multitransform(op_dict: dict[int, OptArray],
                       tensor: OptArray) -> OptArray:
    """Perform a series of tensor contractions and rearrangements by repeated calls to opt_transform referencing a dictionary of dimensions to contract and tensors to contract with

    :param op_dict: Dictionary associating a dimension with a tensor for transformation
    :type op_dict: dictionary[integer, :class:`OptArray`]

    :param tensor: Tensor on which to carry out transformations
    :type tensor: :class:`OptArray`

    :returns: Result of performing the series of transformations
    :rtype: :class:`OptArray`
    """

    ans = tensor
    for ax, mat in op_dict.items():
        ans = opt_transform(mat, ans, 1, ax)

    return ans


def opt_eye(dim1: int, dim2: int | None = None) -> OptArray:
    """Obtain a two dimensional identity tensor (wraps torch.eye)

    :param dim1: Number of rows
    :type dim1: integer
    :param dim2: Number of columns
    :type dim2: integer

    :returns: Two dimensional identity tensor
    :rtype: :class:`OptArray`
    """
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

