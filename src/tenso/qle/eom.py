# coding: utf-8
"""Generating the derivative of the pseudo mode rho in SoP formalism.

The density operator is represented as a tensor network with a "locally entangled" structure.
That is, for each node attaced to the leaf ends, the tensor of 3 degrees, where the first links
towards the root node, the second link is one ket end, and the third link the corresponding bra end.
In this way, the trace of the density operator can be computed by (1) for the leaf nodes A_{ijk}:
    T_i = A_{ijj}
and (2) for the nodes are not leaf nodes B_{aij}, the trace is computed by contracting
    T_a = B_{aij} T_i T_j  
and suppose the root node is C_{mna} with m and n are the ket and bra ends for the system, then
    rho_{mn} = C_{mna} T_a
gives the reduced density operator for the system.
"""

from itertools import chain
from math import prod
from typing import Literal
import numpy as np

from tenso.basis.dvr import DiscreteVariationalRepresentation as Dvr
from tenso.bath.correlation import Correlation
from tenso.libs.backend import (MAX_EINSUM_AXES, OptArray, ArrayLike,
                                opt_array, opt_einsum, opt_array,
                                opt_transform)
from tenso.libs.utils import Optional, huffman_tree
from tenso.state.pureframe import Frame, Node, End
from tenso.state.puremodel import Model, eye_model

EPSILON = 1.0e-14


def local_trace(tensor: OptArray, bra_ax: int, ket_ax: int) -> OptArray:
    """Computing the local trace of a tensor network.

    Parameters
    ----------
    tensor : OptArray
        The tensor to compute the trace.
    bra_ax : int
        The axis index for the bra end.
    ket_ax : int
        The axis index for the ket end.

    Returns
    -------
    OptArray
        The tensor with the bra and ket axes traced out.
    """
    order = tensor.ndim
    assert order >= 2 and bra_ax < order and ket_ax < order
    input_axes = list(range(order))
    input_axes[bra_ax] = input_axes[ket_ax]  # set bra and ket axes to the same
    output_axes = [ax for ax in input_axes if ax != ket_ax]
    assert len(output_axes) == order - 2
    ans = opt_einsum(tensor, input_axes, output_axes)
    return ans


def terminate(tensor: OptArray, term_dict: dict[int, OptArray]):
    order = tensor.ndim
    n = len(term_dict)
    assert order + n - 1 < MAX_EINSUM_AXES

    ax_list = list(sorted(term_dict.keys(), key=(lambda ax: tensor.shape[ax])))
    vec_list = [term_dict[ax] for ax in ax_list]

    args = [tensor, list(range(order))]
    for _v, _ax in zip(vec_list, ax_list):
        args += [_v, [_ax]]
    args.append([ax for ax in range(order) if ax not in ax_list])
    ans = opt_einsum(*args)
    return ans


class _BathOp:

    def __init__(self, dim: int) -> None:
        self.up = opt_array(np.diag(np.sqrt(np.arange(1, dim)), k=-1))
        self.down = opt_array(np.diag(np.sqrt(np.arange(1, dim)), k=1))
        self.number = opt_array(np.diag(np.arange(dim)))
        return


class _DVRBathOp(_BathOp):

    def __init__(self, dvr: Dvr) -> None:
        self.up = opt_array(dvr.creation_mat)
        self.down = opt_array(dvr.annihilation_mat)
        self.number = opt_array(dvr.numberer_mat)
        return


class Hierachy:

    def __init__(self,
                 frame: Frame,
                 root: Node,
                 sys_ket_end: End,
                 sys_bra_end: End,
                 bath_ket_ends: list[End],
                 bath_bra_ends: list[End],
                 sys_dim: int,
                 bath_dims: list[int],
                 bases: Optional[dict[int, Dvr]] = None) -> None:
        self.sys_ket_end = sys_ket_end  # i
        self.sys_bra_end = sys_bra_end  # j
        self.bath_ket_ends = bath_ket_ends  # k_is
        self.bath_bra_ends = bath_bra_ends  # k_js
        all_dims = [sys_dim, sys_dim] + bath_dims + bath_dims
        all_ends = [sys_ket_end, sys_bra_end] + bath_ket_ends + bath_bra_ends
        assert root in frame and set(all_ends) == frame.ends

        dims = {
            end: d
            for end, d in zip(all_ends, all_dims)
        }  # type: dict[End, int]
        self.dims = dims  # type: dict[End, int]
        self.frame = frame
        self.root = root

        # Terminators and basis settings
        self._bases = bases  # type: dict[int, Dvr] | None
        bathops = {}  # type: dict[int, _BathOp]
        if bases is None:
            bases = dict()
        for k, end in enumerate(bath_ket_ends):
            b = bases.get(k, None)
            dim = dims[end]
            assert dim == dims[bath_bra_ends[k]]
            if b is None:
                bathops[k] = _BathOp(dim)
            else:
                assert b.n == dim
                bathops[k] = _DVRBathOp(b)

        self._bathops = bathops
        self._terminate_visitor = list(
            reversed(frame.node_visitor(root, method='BFS')[1:]))
        self._node_axes = frame.get_node_axes(root)
        return

    def lvn_list(self,
                 sys_hamiltonian: ArrayLike) -> list[dict[End, OptArray]]:
        i_end = self.sys_ket_end
        j_end = self.sys_bra_end
        print('generating lvn_list for the system Hamiltonian', flush=True)
        sys_hamiltonian = opt_array(sys_hamiltonian)
        return [{
            i_end: -1.0j * sys_hamiltonian
        }, {
            j_end: 1.0j * sys_hamiltonian.conj()
        }]

    @staticmethod
    def _uhp_sqrt(c: complex) -> tuple[float, float]:
        """Get the square root of a complex number in the upper half plane.
        The square root is defined as:
            sqrt(c) = g - i * epsilon,
        where g is real and epsilon >= 0.

        Returns:
            g: float
            epsilon: float
        """
        sqrt = np.sqrt(c, dtype=complex)
        if sqrt.imag > 0:
            g = -sqrt.real
            epsilon = sqrt.imag
        else:
            g = sqrt.real
            epsilon = -sqrt.imag
        return g, epsilon

    @staticmethod
    def _boolean(z: complex) -> bool:
        if abs(z - 1.0) < EPSILON:
            return True
        elif abs(z) < EPSILON:
            return False
        else:
            raise ValueError(f'Cannot convert {z} to boolean.')

    def quasi_lindblad_list(
            self,
            sys_op: ArrayLike,
            correlation: Correlation,
            hermite: bool = False) -> list[dict[End, OptArray]]:
        """Quasi-Lindblad part of the EOM.
        """
        sys_op = opt_array(sys_op)
        k_max = correlation.k_max
        derivatives = correlation.derivatives
        assert k_max == len(self.bath_ket_ends) == len(self.bath_bra_ends)
        ans = []
        print('generating quasi_lindblad_list for the system operator',
              flush=True)
        if hermite:
            _ql_list_k = self._qlh_list_k
        else:
            _ql_list_k = self._ql_list_k
        for k in range(k_max):
            d_k = derivatives.get((k, k), 0.0)
            w_k = -d_k.imag
            eta_k = -d_k.real
            z_k = self._boolean(correlation.zeropoints[k])
            g_k, epsilon_k = self._uhp_sqrt(correlation.coefficients[k])
            print(
                f'k={k}, z_k={z_k}: '
                f'w_k={w_k}, eta_k={eta_k}, '
                f'g_k={g_k}, epsilon_k={epsilon_k}',
                flush=True)
            ans += _ql_list_k(sys_op, k, w_k, eta_k, g_k, epsilon_k, z_k=z_k)
        for k, j in derivatives.keys():
            if k == j:
                continue
            d_kj = derivatives[k, j]
            g_k, epsilon_k = self._uhp_sqrt(correlation.coefficients[k])
            g_j, epsilon_j = self._uhp_sqrt(correlation.coefficients[j])
            rate = -d_kj * (g_j - 1.0j * epsilon_j) / (g_k - 1.0j * epsilon_k)
            w_kj = rate.imag
            eta_kj = rate.real
            print(f'k={k}, j={j}, w_kj={w_kj}, eta_kj={eta_kj}', flush=True)
            ans += self._ql_list_kj(k, j, w_kj, eta_kj)
        return ans

    def _ql_list_kj(self, k, j, w_kj, eta_kj) -> list[dict[End, OptArray]]:
        """Quasi-Lindblad part between the k-th and j-th bath modes.
        """
        ans = []  # type: list[dict[End, OptArray]]
        k_ket = self.bath_ket_ends[k]
        k_bra = self.bath_bra_ends[k]
        j_ket = self.bath_ket_ends[j]
        j_bra = self.bath_bra_ends[j]
        bathop_k = self._bathops[k]
        bathop_j = self._bathops[j]
        if abs(eta_kj) > EPSILON:
            ans += [
                {
                    j_ket: 2.0 * eta_kj * bathop_j.down,
                    k_bra: bathop_k.down.conj(),
                },
            ]
        if abs(eta_kj) > EPSILON or abs(w_kj) > EPSILON:
            # if eta_k is not zero, we have the Lindblad part
            # if w_k is not zero, we have the Lamb shift part
            ans += [
                {
                    k_ket: (-eta_kj - 1.0j * w_kj) * (bathop_k.up),
                    j_ket: bathop_j.down
                },
                {
                    k_bra: (-eta_kj + 1.0j * w_kj) * (bathop_k.down.conj()),
                    j_bra: bathop_j.up.conj()
                },
            ]
        print(f'k={k}, j={j}, len={len(ans)}', flush=True)
        return ans

    def _ql_list_k(
        self,
        sys_op: OptArray,
        k: int,
        w_k: float,
        eta_k: float,
        g_k: float,
        epsilon_k: float,
        z_k: bool = True,
    ) -> list[dict[End, OptArray]]:
        """Quasi-Lindblad part for the k-th bath mode.
        """
        s_ket = self.sys_ket_end
        s_bra = self.sys_bra_end
        k_ket = self.bath_ket_ends[k]
        k_bra = self.bath_bra_ends[k]
        bathop = self._bathops[k]
        ans = []  # type: list[dict[End, OptArray]]
        if abs(eta_k) > EPSILON:
            # if eta_k is not zero, we have the Lindblad part
            ans += [
                {
                    k_ket: 2.0 * eta_k * bathop.down,
                    k_bra: bathop.down.conj(),
                },
            ]
        if abs(eta_k) > EPSILON or abs(w_k) > EPSILON:
            # if w_k is not zero, we have the Lamb shift part
            ans += [
                {
                    k_ket: (-eta_k - 1.0j * w_k) * (bathop.number),
                },
                {
                    k_bra: (-eta_k + 1.0j * w_k) * (bathop.number.conj()),
                },
            ]

        _sigma = g_k - 1.0j * epsilon_k
        _sigma_c = g_k + 1.0j * epsilon_k
        if abs(g_k) > EPSILON or abs(epsilon_k) > EPSILON:
            ans += [
                {
                    s_ket: -1.0j * sys_op.T.conj(),
                    k_ket: _sigma * bathop.down,
                },
                {
                    s_bra: 1.0j * sys_op.T,
                    k_bra: _sigma_c * bathop.down.conj(),
                },
                {
                    s_ket: -1.0j * sys_op,
                    k_bra: (_sigma_c - z_k * _sigma) * bathop.down.conj(),
                },
                {
                    s_bra: 1.0j * sys_op.conj(),
                    k_ket: (_sigma - z_k * _sigma_c) * bathop.down,
                },
            ]
            if z_k:
                ans += [
                    {
                        s_ket: -1.0j * sys_op,
                        k_ket: _sigma * bathop.up,
                    },
                    {
                        s_bra: 1.0j * sys_op.conj(),
                        k_bra: _sigma_c * bathop.up.conj(),
                    },
                ]
        print(f'k={k}, z_k={z_k}, len={len(ans)}', flush=True)
        return ans

    def _qlh_list_k(
        self,
        sys_op: OptArray,
        k: int,
        w_k: float,
        eta_k: float,
        g_k: float,
        epsilon_k: float,
        z_k: bool = True,
    ) -> list[dict[End, OptArray]]:
        """Quasi-Lindblad part for the k-th bath mode.
        Assume sys_op is Hermitian.
        """
        s_ket = self.sys_ket_end
        s_bra = self.sys_bra_end
        k_ket = self.bath_ket_ends[k]
        k_bra = self.bath_bra_ends[k]
        bathop = self._bathops[k]
        ans = []  # type: list[dict[End, OptArray]]
        if eta_k > EPSILON:
            # if eta_k is not zero, we have the Lindblad part
            ans += [
                {
                    k_ket: 2.0 * eta_k * bathop.down,
                    k_bra: bathop.down.conj(),
                },
            ]
        if eta_k > EPSILON or abs(w_k) > EPSILON:
            # if w_k is not zero, we have the Lamb shift part
            ans += [
                {
                    k_ket: (-eta_k - 1.0j * w_k) * (bathop.number),
                },
                {
                    k_bra: (-eta_k + 1.0j * w_k) * (bathop.number.conj()),
                },
            ]

        _sigma = g_k - 1.0j * epsilon_k
        _sigma_c = g_k + 1.0j * epsilon_k
        if abs(g_k) > EPSILON or abs(epsilon_k) > EPSILON:
            ans += [
                {
                    s_ket: -1.0j * sys_op,
                    k_bra: (_sigma_c - z_k * _sigma) * bathop.down.conj(),
                },
                {
                    s_bra: 1.0j * sys_op.conj(),
                    k_ket: (_sigma - z_k * _sigma_c) * bathop.down,
                },
                {
                    s_ket: -1.0j * sys_op,
                    k_ket: _sigma * (bathop.down + z_k * bathop.up),
                },
                {
                    s_bra: 1.0j * sys_op.conj(),
                    k_bra: _sigma_c * (bathop.down + z_k * bathop.up).conj(),
                },
            ]
        print(f'(H) k={k}, z_k={z_k}, len={len(ans)}', flush=True)
        return ans

    def initialize_state(self, rdo: ArrayLike, rank: int) -> Model:
        """
        Assume Ends sys_i and sys_j are attached to the root node axes 0 and 1.
        """
        rdo = np.array(rdo, dtype=complex)
        rdo /= np.trace(rdo)
        root = self.root
        shapes = dict()  # type: dict[Node, list[int]]
        for _n in self.frame.nodes:
            shapes[_n] = [
                rank if isinstance(p, Node) else self.dims[p]
                for p in self.frame.near_points(_n)
            ]
        model = eye_model(self.frame, root, shapes=shapes)
        ext_shape = [k for i, k in enumerate(shapes[root]) if i > 1]
        ext = np.zeros([prod(ext_shape)])
        ext[0] = 1.0
        root_array = np.tensordot(rdo, ext, axes=0).reshape(shapes[root])
        model.update({root: opt_array(root_array)})
        return model

    def _get_local_trace(self, model: Model) -> dict[Node, OptArray]:
        """Initialize the number basis for the bath ends.
        """
        terminators = dict()  # type: dict[Node, OptArray]
        for ki_end, kj_end in zip(self.bath_ket_ends, self.bath_bra_ends):
            dim = self.dims[ki_end]
            assert dim == self.dims[kj_end]
            leaf_node, i = self.frame.dual(ki_end, None)
            leaf_node2, j = self.frame.dual(kj_end, None)
            # assert that the ki and kj ends are attached to the same leaf node
            assert leaf_node == leaf_node2
            assert isinstance(leaf_node, Node)
            assert i is not None and j is not None

            t = local_trace(model[leaf_node], i, j)
            terminators[leaf_node] = t
        return terminators

    def get_rdo(self, edo: Model) -> OptArray:
        axes = self._node_axes
        root = self.root
        near = self.frame.near_points

        terminators = self._get_local_trace(
            edo)  # initialize the terminators for the leaf nodes
        for p in self._terminate_visitor:
            if p in terminators:
                continue
            term_dict = {
                i: terminators[q]
                for i, q in enumerate(near(p)) if i != axes[p]
            }
            terminators[p] = terminate(edo[p], term_dict)

        # root node: i and j and n_left
        term_dict = {
            i: terminators[q]
            for i, q in enumerate(near(root)) if i >= 2
        }
        rdo = terminate(edo[root], term_dict)
        return rdo


class FrameFactory:
    prefix = '[H]'

    def __init__(self, bath_dof: int) -> None:
        self.bath_dof = bath_dof  # type: int
        self.sys_ket_end = End(self.prefix + '>')  # type: End
        self.sys_bra_end = End(self.prefix + '<')  # type: End
        self.bath_ket_ends = [
            End(self.prefix + str(i) + '>') for i in range(bath_dof)
        ]  # type: list[End]
        self.bath_bra_ends = [
            End(self.prefix + str(i) + "<") for i in range(bath_dof)
        ]  # type: list[End]
        self._node_counter = 0  # type: int
        return

    def _new_node(self) -> Node:
        n = Node(self.prefix + str(self._node_counter))
        assert isinstance(n, Node)
        self._node_counter += 1
        return n

    def naive(self) -> tuple[Frame, Node]:
        frame = Frame()
        bath_nodes = [self._new_node() for _ in range(self.bath_dof)]
        for k, n in enumerate(bath_nodes):
            frame.add_link(n, self.bath_ket_ends[k])
            frame.add_link(n, self.bath_bra_ends[k])
        root = self._new_node()
        for e in chain([self.sys_ket_end, self.sys_bra_end], bath_nodes):
            frame.add_link(root, e)
        return frame, root

    def tree(self,
             bath_importances: None | list[int] = None,
             n_ary: int = 2) -> tuple[Frame, Node]:
        if bath_importances is None:
            bath_importances = [1] * self.bath_dof
        frame = Frame()
        bath_nodes = [self._new_node() for _ in range(self.bath_dof)]
        for k, n in enumerate(bath_nodes):
            frame.add_link(n, self.bath_ket_ends[k])
            frame.add_link(n, self.bath_bra_ends[k])

        root = self._new_node()
        frame.add_link(root, self.sys_ket_end)
        frame.add_link(root, self.sys_bra_end)
        graph, b_root = huffman_tree(bath_nodes,
                                     self._new_node,
                                     importances=bath_importances,
                                     n_ary=n_ary)
        frame.add_link(root, b_root)
        for n, children in graph.items():
            for child in children:
                frame.add_link(n, child)
        return frame, root

    def train(self) -> tuple[Frame, Node]:
        k_max = self.bath_dof
        frame = Frame()
        bath_nodes = [self._new_node() for _ in range(self.bath_dof)]
        for k, n in enumerate(bath_nodes):
            frame.add_link(n, self.bath_ket_ends[k])
            frame.add_link(n, self.bath_bra_ends[k])

        train_nodes = [self._new_node() for _ in range(k_max)]
        root = train_nodes[0]
        frame.add_link(root, self.sys_ket_end)
        frame.add_link(root, self.sys_bra_end)
        for i in range(1, k_max):
            frame.add_link(train_nodes[i - 1], train_nodes[i])
            frame.add_link(train_nodes[i], bath_nodes[i - 1])
        frame.add_link(train_nodes[-1], bath_nodes[-1])
        return frame, root
