# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
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
                 bath_ends: list[list[End]],
                 sys_dim: int,
                 bath_dims: list[list[int]],
                 bases: Optional[dict[End, Dvr]] = None) -> None:
        self.sys_ket_end = sys_ket_end  # i
        self.sys_bra_end = sys_bra_end  # j
        self.bath_ends = bath_ends  # ks
        all_dims = [sys_dim, sys_dim] + list(chain(*bath_dims))
        all_ends = [sys_ket_end, sys_bra_end] + list(chain(*bath_ends))
        assert root in frame and set(all_ends) == frame.ends

        dims = {
            end: d
            for end, d in zip(all_ends, all_dims)
        }  # type: dict[End, int]
        self.dims = dims  # type: dict[End, int]
        self.frame = frame
        self.root = root

        # Terminators and basis settings
        self._bases = bases  # type: dict[End, Dvr]
        bathops = {}  # type: dict[End, _BathOp]
        if bases is None:
            bases = dict()
        for end in chain(*bath_ends):
            b = bases.get(end, None)
            if b is None:
                bathops[end] = _BathOp(dims[end])
            else:
                assert b.n == dims[end]
                bathops[end] = _DVRBathOp(b)
        self._bathops = bathops
        self._terminators = {}  # type: dict[End, OptArray]
        self._terminate_visitor = list(
            reversed(frame.node_visitor(root, method='BFS')[1:]))
        self._node_axes = frame.get_node_axes(root)
        return

    def lvn_list(self,
                 sys_hamiltonian: ArrayLike) -> list[dict[End, OptArray]]:
        i_end = self.sys_ket_end
        j_end = self.sys_bra_end
        return [{
            i_end: -1.0j * opt_array(sys_hamiltonian)
        }, {
            j_end: 1.0j * opt_array(sys_hamiltonian.conj())
        }]

    def lindblad_list(
            self, sys_op: ArrayLike,
            lindblad_rate: float | None) -> list[dict[End, OptArray]]:
        if lindblad_rate is None:
            return []

        i_end = self.sys_ket_end
        j_end = self.sys_bra_end
        sys_op = opt_array(sys_op)
        _lamb = 0.5 * lindblad_rate * sys_op @ sys_op

        return [{
            i_end: -_lamb,
        }, {
            j_end: -_lamb.conj(),
        }, {
            i_end: lindblad_rate * sys_op,
            j_end: sys_op.conj(),
        }]

    def heom_list(
        self,
        n_bath: int,
        sys_op: ArrayLike,
        correlation: Correlation,
        metric: Literal['re', 'abs'] | complex = 're'
    ) -> list[dict[End, OptArray]]:
        k_max = correlation.k_max
        coefficients = correlation.coefficients
        conj_coefficents = correlation.conj_coefficents
        zeropoints = correlation.zeropoints
        derivatives = correlation.derivatives
        k_ends = self.bath_ends[n_bath]
        assert len(k_ends) == k_max
        adm_factors = [
            self._adm_factor(k, correlation, metric=metric)
            for k in range(k_max)
        ]

        i_end = self.sys_ket_end
        i_op = -1.0j * opt_array(sys_op)
        j_end = self.sys_bra_end
        j_op = i_op.conj()

        ups = []  # type: list[OptArray]
        downs = []  # type: list[OptArray]
        nums = []  # type: list[OptArray]
        for k_end in k_ends:
            k_op = self._bathops[k_end]
            ups.append(k_op.up)
            downs.append(k_op.down)
            nums.append(k_op.number)

        ans = []
        for _k in range(k_max):
            zk = zeropoints[_k]
            fk = adm_factors[_k]
            up = ups[_k] / fk
            down = downs[_k] * fk
            k_end = k_ends[_k]
            ans += [{
                i_end: i_op,
                k_end: zk * coefficients[_k] * up + down
            }, {
                j_end: j_op,
                k_end: zk * conj_coefficents[_k] * up + down
            }]

        for (_k1, _k2), dk in derivatives.items():
            if _k1 == _k2:
                ans.append({k_ends[_k1]: dk * nums[_k1]})
            else:
                sqrt_dk = np.sqrt(dk, dtype=complex)
                ans.append({
                    k_ends[_k1]: sqrt_dk / adm_factors[_k1] * ups[_k1],
                    k_ends[_k2]: sqrt_dk * adm_factors[_k2] * downs[_k2]
                })
        return ans

    @staticmethod
    def _adm_factor(k: int,
                    c: Correlation,
                    metric: Literal['re', 'abs'] | complex = 're'):
        c_k = c.coefficients[k]
        cc_k = c.conj_coefficents[k]
        if metric == 're':
            f_k = np.sqrt(abs(c_k.real + cc_k.real) / 2.0)
            if (c_k.imag + cc_k.imag) > EPSILON:
                f_k *= -1.0
        elif metric == 'abs':
            f_k = np.sqrt((abs(c_k) + abs(cc_k)) / 2.0)
        else:
            f_k = complex(metric)
        if __debug__:
            print(
                f'For k={k}: '
                f's:{(c_k.real + cc_k.real) / 2:.8f} | '
                f'e:{(c_k.real - cc_k.real) / 2:.8f} | '
                f'a:{c_k.imag:.8f} | '
                f'f:{f_k:.8f} | '
                f'f^2:{f_k**2:.8f}',
                flush=True,
            )
        return f_k

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
        for k_end in chain.from_iterable(self.bath_ends):
            if k_end in self._bases:
                self._init_dvr_basis(model, k_end, self._bases[k_end])
            else:
                self._init_number_basis(k_end)
        return model

    def _init_number_basis(self, k_end: End):
        dim = self.dims[k_end]
        ground_vec = np.zeros([dim]).real
        ground_vec[0] = 1.0
        self._terminators[k_end] = opt_array(ground_vec)
        return

    def _init_dvr_basis(self, state: Model, k_end: End, basis: Dvr):
        tfmat = opt_array(basis.fock2dvr_mat)
        if tfmat[:, 0].real.sum() < 0:
            tfmat = -tfmat
        ground_vec = (tfmat.mH)[0]
        if ground_vec.real.sum() < 0:
            ground_vec = -ground_vec

        _n, _i = self.frame.dual(k_end, None)
        state.update({_n: opt_transform(tfmat, state[_n], 1, _i)})
        self._terminators[k_end] = ground_vec
        return

    def get_rdo(self, edo: Model) -> OptArray:
        axes = self._node_axes
        root = self.root
        near = self.frame.near_points

        terminators = {e: vec for e, vec in self._terminators.items()}
        for p in self._terminate_visitor:
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

    def __init__(self, bath_dofs: list[int]) -> None:
        self.bath_dofs = bath_dofs  # type: int
        self.bath_dof = sum(bath_dofs)
        self.sys_ket_end = End(self.prefix + 'i')  # type: End
        self.sys_bra_end = End(self.prefix + 'j')  # type: End
        self.chained_bath_ends = [
            End(self.prefix + f'{n}-{k}') for (n, dof) in enumerate(bath_dofs)
            for k in range(dof)
        ]  # type: list[End]
        self.bath_ends = [[
            self.chained_bath_ends[n * dof + k] for k in range(dof)
        ] for n, dof in enumerate(bath_dofs)]

        self._node_counter = 0  # type: int
        return

    def _new_node(self) -> Node:
        n = Node(self.prefix + str(self._node_counter))
        assert isinstance(n, Node)
        self._node_counter += 1
        return n

    def naive(self) -> tuple[Frame, Node]:
        frame = Frame()
        root = self._new_node()
        for e in chain([self.sys_ket_end, self.sys_bra_end],
                       self.chained_bath_ends):
            frame.add_link(root, e)
        return frame, root

    def tree(self,
             bath_importances: None | list[int] = None,
             n_ary: int = 2) -> tuple[Frame, Node]:
        if bath_importances is None:
            bath_importances = [1] * self.bath_dof
        frame = Frame()
        root = self._new_node()
        frame.add_link(root, self.sys_ket_end)
        frame.add_link(root, self.sys_bra_end)
        graph, b_root = huffman_tree(self.chained_bath_ends,
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
        train_nodes = [self._new_node() for _ in range(k_max)]
        root = train_nodes[0]
        frame.add_link(root, self.sys_ket_end)
        frame.add_link(root, self.sys_bra_end)
        for i in range(1, k_max):
            frame.add_link(train_nodes[i - 1], train_nodes[i])
            frame.add_link(train_nodes[i], self.chained_bath_ends[i - 1])
        frame.add_link(train_nodes[-1], self.chained_bath_ends[-1])
        return frame, root
