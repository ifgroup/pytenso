# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from itertools import chain
from math import prod
from typing import Literal
import numpy as np

from tenso.basis.dvr import DiscreteVariationalRepresentation as Dvr
from tenso.bath.correlation import Correlation
from tenso.libs.backend import (MAX_EINSUM_AXES, OptArray, ArrayLike, opt_array,
                                  opt_einsum, opt_array, opt_transform)
from tenso.libs.utils import Optional, huffman_tree
from tenso.state.pureframe import Frame, Node, End
from tenso.state.puremodel import Model, eye_model
from tenso.heom.eom import _BathOp, _DVRBathOp, terminate

EPSILON = 1.0e-14

 


class Hierachy:
    def __init__(self,
                 frame: Frame,
                 root: Node,
                 sys_ket_ends: list[End],
                 sys_bra_ends: list[End],
                 bath_ends: list[list[End]],
                 sys_dims: list[int],
                 bath_dims: list[list[int]],
                 bases: Optional[dict[End, Dvr]] = None) -> None:
        self.sys_ket_ends = sys_ket_ends  # is
        self.sys_bra_ends = sys_bra_ends  # js
        self.bath_ends = bath_ends  # kss
        all_dims = sys_dims + sys_dims + list(chain(*bath_dims))
        all_ends = sys_ket_ends + sys_bra_ends + list(chain(*bath_ends))
        assert root in frame and set(all_ends) == frame.ends
        assert len(all_dims) == len(all_ends)

        dims = {
            end: d
            for end, d in zip(all_ends, all_dims)
        }  # type: dict[End, int]
        self.dims = dims  # type: dict[End, int]
        self.frame = frame
        self.root = root

        # Terminators and basis settings
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
        self._bases = bases  # type: dict[End, Dvr]
        self._terminators = {}  # type: dict[End, OptArray]
        self._terminate_visitor = list(
            reversed(frame.node_visitor(root, method='BFS')))
        self._node_axes = frame.get_node_axes(root)
        return

    def lvn_list(
        self, sys_hamiltonians: list[None | ArrayLike],
        sys_couplings: list[dict[int,
                                 ArrayLike]]) -> list[dict[End, OptArray]]:
        i_ends = self.sys_ket_ends
        j_ends = self.sys_bra_ends
        ans = list()
        for _s, h in enumerate(sys_hamiltonians):
            if h is not None:
                op = opt_array(h)
                ans += [{
                    i_ends[_s]: -1.0j * op
                }, {
                    j_ends[_s]: 1.0j * op.conj()
                }]
        for op_dict in sys_couplings:
            phase = np.exp(-0.5j * np.pi / len(op_dict))
            idxs = sorted(op_dict.keys())
            ops = [opt_array(phase * op_dict[_i]) for _i in idxs]
            # # Put the phase factor with DOF-0.
            # ops[0] = -1.0j * ops[0]
            ans += [
                {
                    i_ends[idx]: op
                    for idx, op in zip(idxs, ops)
                },
                {
                    j_ends[idx]: op.conj()
                    for idx, op in zip(idxs, ops)
                },
            ]
        return ans

    def lindblad_list(
            self, sys_ops: list[dict[int, ArrayLike]],
            lindblad_rates: list[float | None]) -> list[dict[End, OptArray]]:

        assert len(sys_ops) == len(lindblad_rates)
        ans = list()
        for s_dict, lr in zip(sys_ops, lindblad_rates):
            if lr is not None:
                s_idxs = sorted(s_dict.keys())
                s_mats = [s_dict[_i] for _i in s_idxs]
                ans += self._single_lindblad_list(s_idxs, s_mats, lr)
        return ans

    def _single_lindblad_list(
            self, sys_idxs: list[int], sys_ops: list[ArrayLike],
            lindblad_rate: float) -> list[dict[End, OptArray]]:
        opt_ops = [opt_array(op) for op in sys_ops]
        lamb_shift_ops = [0.5 * lindblad_rate * op @ op for op in opt_ops]
        i_op = {
            self.sys_ket_ends[_i]: -op
            for _i, op in zip(sys_idxs, lamb_shift_ops)
        }
        j_op = {
            self.sys_bra_ends[_j]: -op.conj()
            for _j, op in zip(sys_idxs, lamb_shift_ops)
        }
        ij_op = {
            self.sys_ket_ends[_i]: lindblad_rate * op
            for _i, op in zip(sys_idxs, opt_ops)
        }
        ij_op.update({
            self.sys_bra_ends[_j]: op.conj()
            for _j, op in zip(sys_idxs, opt_ops)
        })
        return [i_op, j_op, ij_op]

    def heom_list(
        self,
        sys_ops: list[dict[int, ArrayLike]],
        correlations: list[Correlation],
        metric: Literal['re', 'abs'] | complex = 're'
    ) -> list[dict[End, OptArray]]:
        assert len(sys_ops) == len(correlations)
        ans = list()
        for n_bath, (s_dict, c) in enumerate(zip(sys_ops, correlations)):
            s_idxs = sorted(s_dict.keys())
            s_mats = [s_dict[_i] for _i in s_idxs]
            ans += self._single_heom_list(s_idxs,
                                          s_mats,
                                          n_bath,
                                          c,
                                          metric=metric)
        return ans

    def _single_heom_list(
            self, sys_idxs: list[int], sys_ops: list[ArrayLike], bath_idx: int,
            correlation: Correlation, metric: Literal['re', 'abs'] | complex
    ) -> list[dict[End, OptArray]]:
        k_max = correlation.k_max
        coefficients = correlation.coefficients
        conj_coefficents = correlation.conj_coefficents
        zeropoints = correlation.zeropoints
        derivatives = correlation.derivatives
        k_ends = self.bath_ends[bath_idx]
        assert len(k_ends) == k_max

        phase = np.exp(-0.5j * np.pi / len(sys_idxs))
        opt_ops = [opt_array(phase * op) for op in sys_ops]
        # opt_ops = [opt_array(op) for op in sys_ops]
        # opt_ops[0] = -1.0j * opt_ops[0]
        i_op = [(self.sys_ket_ends[_i], op)
                for _i, op in zip(sys_idxs, opt_ops)]
        j_op = [(self.sys_bra_ends[_j], op.conj())
                for _j, op in zip(sys_idxs, opt_ops)]
        adm_factors = [
            self._adm_factor(k, correlation, metric=metric)
            for k in range(k_max)
        ]

        ups = list()  # type: list[OptArray]
        downs = list()  # type: list[OptArray]
        nums = list()  # type: list[OptArray]
        for fk, k_end in zip(adm_factors, k_ends):
            k_op = self._bathops[k_end]
            ups.append(k_op.up / fk)
            downs.append(k_op.down * fk)
            nums.append(k_op.number)

        ans = list()
        for k in range(k_max):
            zk = zeropoints[k]
            up = ups[k]
            down = downs[k]
            k_end = k_ends[k]
            fw = dict(i_op)
            fw[k_end] = zk * coefficients[k] * up + down
            bw = dict(j_op)
            bw[k_end] = zk * conj_coefficents[k] * up + down
            ans += [fw, bw]
        for (k1, k2), dk in derivatives.items():
            if k1 == k2:
                ans.append({k_ends[k1]: dk * nums[k]})
            else:
                ans.append({
                    k_ends[k1]: dk / adm_factors[k1] * ups[k1],
                    k_ends[k2]: adm_factors[k2] * downs[k2]
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

    def initialize_pure_state(
            self,
            local_wfns: list[ArrayLike],
            rank: int,
            local_hs: None | list[None | ArrayLike] = None) -> Model:
        # Initialize the system part.
        shapes = dict()  # type: dict[Node, list[int]]
        for _n in self.frame.nodes:
            shapes[_n] = [
                rank if isinstance(p, Node) else self.dims[p]
                for p in self.frame.near_points(_n)
            ]
        model = eye_model(self.frame, self.root, shapes=shapes)
        # Note that the eye_model represents the lowest level.
        # Perform a unitary transformation to the dual dimension of End such that
        # the local_wfns become the lowest level. The rest of the unitary matrix
        # is constructed from QR decomposition of [wfn] + [eigenvectors of h].
        if local_hs is None:
            local_hs = [None] * len(local_wfns)
        for _i, wfn in enumerate(local_wfns):
            new_valuation = dict()  # type: dict[Node, OptArray]
            _i_end = self.sys_ket_ends[_i]
            _j_end = self.sys_bra_ends[_i]
            dim = self.dims[_i_end]
            fst = np.array(wfn)
            # Normalize the first vector.
            fst /= np.linalg.norm(fst)
            fst = fst.reshape((dim, 1))
            h_i = local_hs[_i]
            if h_i is None:
                h_i = np.identity(dim)
            columnspace = np.hstack((fst, h_i))
            q, _ = np.linalg.qr(columnspace)
            opt_q = opt_array(q)
            p_node, i = self.frame.dual(_i_end, None)
            q_node, j = self.frame.dual(_j_end, None)
            if p_node is q_node:
                tmp = opt_transform(opt_q, model[p_node], 1, i)
                a = opt_transform(opt_q.conj(), tmp, 1, j)
                new_valuation[p_node] = a
            else:
                new_valuation[p_node] = opt_transform(opt_q, model[p_node], 1,
                                                      i)
                new_valuation[q_node] = opt_transform(opt_q.conj(),
                                                      model[q_node], 1, j)
            model.update(new_valuation)

        # Initialize the bath part.
        for k_end in chain(*self.bath_ends):
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

    def get_rdo_element(self, edo: Model, sys_is: list[None | int],
                        sys_js: list[None | int]) -> complex:
        assert len(sys_is) == len(self.sys_ket_ends)
        assert len(sys_js) == len(self.sys_bra_ends)
        axes = self._node_axes
        root = self.root
        near = self.frame.near_points

        ket_terminators = {}
        for i_end, _i in zip(self.sys_ket_ends, sys_is):
            if _i is not None:
                vec = np.zeros([self.dims[i_end]])
                vec[_i] = 1.0
                ket_terminators[i_end] = opt_array(vec)
        bra_terminators = {}
        for j_end, _j in zip(self.sys_bra_ends, sys_js):
            if _j is not None:
                vec = np.zeros([self.dims[j_end]])
                vec[_j] = 1.0
                bra_terminators[j_end] = opt_array(vec)

        terminators = {
            e: vec
            for e, vec in chain(ket_terminators.items(), bra_terminators.items(
            ), self._terminators.items())
        }
        for p in self._terminate_visitor:
            term_dict = {
                i: terminators[q]
                for i, q in enumerate(near(p)) if i != axes[p]
            }
            terminators[p] = terminate(edo[p], term_dict)

        return terminators[root].item()

    def get_rdo(self, edo: Model) -> OptArray:
        raise NotImplementedError
        axes = self._node_axes
        root = self.root
        near = self.frame.near_points

        terminators = {e: self._terminators[e] for e in self._terminators}
        #print(terminators)
        # print(terminators)
        for p in self._terminate_visitor:
            term_dict = {
                i: _t
                for i, q in enumerate(near(p)) if i != axes[p] and (
                    _t := terminators.get(q, None)) is not None
            }
            terminators[p] = terminate(edo[p], term_dict)

        for p in self.frame.point_visitor(root, method='BFS'):
            if p not in terminators:
                print(f'{p} is not in terminators.')
            else:
                print(f'{p}: {terminators[p].shape}')
        print(f'{root}: {terminators[root].shape}')
        return terminators[root]


class FrameFactory:
    prefix = '[MH]'

    def __init__(self, sys_dof: int, bath_dofs: list[int]) -> None:
        self.n_bath = len(bath_dofs)
        self.sys_dof = sys_dof  # type: int
        self.bath_dofs = bath_dofs  # type: list[int]
        self.dof = 2 * sys_dof + sum(bath_dofs)
        self.sys_ket_ends = [End(self.prefix + f"i-{_i}")
                             for _i in range(sys_dof)]  # type: list[End]
        self.sys_bra_ends = [End(self.prefix + f"j-{_i}")
                             for _i in range(sys_dof)]  # type: list[End]
        self.bath_ends = [[End(self.prefix + f"{n}-{k}") for k in range(dof)]
                          for n, dof in enumerate(bath_dofs)]  # type: list[list[End]]
        self.ends = list(
            chain(reversed(self.sys_bra_ends), *self.bath_ends, self.sys_ket_ends))

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
        for e in self.ends:
            frame.add_link(root, e)
        return frame, root

    def tree(self,
             importances: None | dict[End, int] = None,
             n_ary: int = 2) -> tuple[Frame, Node]:
        if importances is None:
            im_list = [1] * self.dof
        else:
            im_list = [importances.get(e, 1) for e in self.ends]
        graph, root = huffman_tree(self.ends,
                                   self._new_node,
                                   importances=im_list,
                                   n_ary=n_ary)
        frame = Frame()
        for n, children in graph.items():
            for child in children:
                frame.add_link(n, child)
        return frame, root

    def train(self, end_order: list[End]) -> tuple[Frame, Node]:
        assert set(end_order) == set(self.ends)
        assert self.dof >= 3
        train_nodes = [self._new_node() for _ in range(self.dof - 2)]
        frame = Frame()
        frame.add_link(end_order[0], train_nodes[0])
        for i, end in enumerate(end_order[1:-1]):
            frame.add_link(train_nodes[i], end)
            if i > 0:
                frame.add_link(train_nodes[i - 1], train_nodes[i])
        frame.add_link(train_nodes[-1], end_order[-1])
        root = train_nodes[0]
        return frame, root
