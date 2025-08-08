# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from itertools import chain
import numpy as np

from tenso.basis.dvr import DiscreteVariationalRepresentation as Dvr
from tenso.bath.star import StarBosons
from tenso.libs.backend import (OptArray, ArrayLike, opt_array, opt_array,
                                  opt_transform)
from tenso.libs.utils import huffman_tree
from tenso.state.pureframe import Frame, Node, End, Point
from tenso.state.puremodel import Model, eye_model

EPSILON = 1.0e-14
PREFIX = '[HS]'
PREFIX_SYS = '[S]'


def trace(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
    """Complex conjugate not included
    """
    dim1 = tensor1.shape[ax]
    dim2 = tensor2.shape[ax]

    left = tensor1.moveaxis(ax, 0).reshape((dim1, -1))
    right = tensor2.moveaxis(ax, -1).reshape((-1, dim2))
    return left @ right


class _BathOp:

    def __init__(self, dim: int) -> None:
        self.up = opt_array(np.diag(np.sqrt(np.arange(1, dim)), k=-1))
        self.down = opt_array(np.diag(np.sqrt(np.arange(1, dim)), k=1))
        self.number = opt_array(np.diag(np.arange(dim)))
        return


class Hierachy:
    def __init__(self, frame: Frame, root: Node, sys_ends: list[End],
                 bath_ends: list[list[End]], sys_dims: list[int],
                 bath_dims: list[list[int]]) -> None:
        self.sys_ends = sys_ends  # system
        self.bath_ends = bath_ends  # baths
        all_dims = sys_dims + list(chain(*bath_dims))
        all_ends = sys_ends + list(chain(*bath_ends))
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
        bathops = {
            end: _BathOp(dims[end])
            for end in chain(*bath_ends)
        }  # type: dict[End, _BathOp]
        self._bathops = bathops
        self._terminators = {}  # type: dict[End, OptArray]
        self._point_visitor = frame.point_visitor(start=root, method='DFS')
        self._node_axes = frame.get_node_axes(root)
        return

    def get_densities(self, state: Model) -> dict[Point, OptArray]:
        axes = self._node_axes
        dual = self.frame.dual
        densities = dict()  # type: dict[Point, OptArray]

        _point_visitor = self.frame.point_visitor(start=self.root,
                                                  method='DFS')
        # From root to leaves
        for p in _point_visitor[1:]:
            i = axes.get(p, None)
            q, j = dual(p, i)
            ax = axes[q]
            a_q = state[q]
            a_qc = a_q.conj()
            if ax is not None:
                den_q = densities[q]
                a_q = opt_transform(den_q, a_q, 0, ax)
                ans = trace(a_q, a_qc, j)
            else:
                ans = trace(a_q, a_qc, j)
            densities[p] = ans
        return densities

    def tdse_list(
        self, sys_hamiltonians: list[None | ArrayLike],
        sys_couplings: list[dict[int,
                                 ArrayLike]]) -> list[dict[End, OptArray]]:
        i_ends = self.sys_ends
        ans = list()
        for _s, h in enumerate(sys_hamiltonians):
            if h is not None:
                op = opt_array(h)
                ans += [{i_ends[_s]: -1.0j * op}]
        for op_dict in sys_couplings:
            idxs = sorted(op_dict.keys())
            # phase = np.exp(-0.5j * np.pi / len(op_dict))
            # ops = [opt_array(phase * op_dict[_i]) for _i in idxs]
            # Put the phase factor with DOF-0.
            ops = [opt_array(op_dict[_i]) for _i in idxs]
            ops[0] = -1.0j * ops[0]
            ans += [{i_ends[idx]: op for idx, op in zip(idxs, ops)}]
        return ans

    def heom_list(
        self,
        sys_ops: list[dict[int, ArrayLike]],
        correlations: list[StarBosons],
    ) -> list[dict[End, OptArray]]:
        assert len(sys_ops) == len(correlations)
        ans = list()
        for n_bath, (s_dict, c) in enumerate(zip(sys_ops, correlations)):
            s_idxs = sorted(s_dict.keys())
            s_mats = [s_dict[_i] for _i in s_idxs]
            ans += self._single_heom_list(s_idxs, s_mats, n_bath, c)
        return ans

    def bath_q_list(
            self, correlations: list[StarBosons]) -> list[dict[End, OptArray]]:
        ans = list()
        for n_bath, c in enumerate(correlations):
            k_ends = self.bath_ends[n_bath]
            couplings = c.couplings
            conj_couplings = c.conj_couplings
            ups = list()  # type: list[OptArray]
            downs = list()  # type: list[OptArray]
            for k_end in k_ends:
                k_op = self._bathops[k_end]
                ups.append(k_op.up)
                downs.append(k_op.down)
            for k in range(c.k_max):
                term = dict()
                x_k = conj_couplings[k] * ups[k] + couplings[k] * downs[k]
                term[k_ends[k]] = x_k
                ans.append(term)
        return ans

    def bath_q2_list(
            self, correlations: list[StarBosons]) -> list[dict[End, OptArray]]:
        ans = list()
        for n_bath, c in enumerate(correlations):
            couplings = c.couplings
            conj_couplings = c.conj_couplings
            k_ends = self.bath_ends[n_bath]
            ups = list()  # type: list[OptArray]
            downs = list()  # type: list[OptArray]
            for k_end in k_ends:
                k_op = self._bathops[k_end]
                ups.append(k_op.up)
                downs.append(k_op.down)
            for k in range(c.k_max):
                term = dict()
                x_k = conj_couplings[k] * ups[k] + couplings[k] * downs[k]
                term[k_ends[k]] = x_k.conj() @ x_k
                ans.append(term)
        return ans

    def _single_heom_list(
            self, sys_idxs: list[int], sys_ops: list[ArrayLike], bath_idx: int,
            correlation: StarBosons) -> list[dict[End, OptArray]]:
        k_max = correlation.k_max
        couplings = correlation.couplings
        conj_couplings = correlation.conj_couplings
        frequecies = correlation.frequencies
        k_ends = self.bath_ends[bath_idx]
        assert len(k_ends) == k_max

        phase = np.exp(-0.5j * np.pi / len(sys_idxs))
        opt_ops = [opt_array(phase * op) for op in sys_ops]
        # opt_ops = [opt_array(op) for op in sys_ops]
        # opt_ops[0] = -1.0j * opt_ops[0]
        i_op = [(self.sys_ends[_i], op) for _i, op in zip(sys_idxs, opt_ops)]

        ups = list()  # type: list[OptArray]
        downs = list()  # type: list[OptArray]
        nums = list()  # type: list[OptArray]
        for k_end in k_ends:
            k_op = self._bathops[k_end]
            ups.append(k_op.up)
            downs.append(k_op.down)
            nums.append(k_op.number)

        ans = list()
        for k in range(k_max):
            fw = dict(i_op)
            fw[k_ends[k]] = conj_couplings[k] * ups[k] + \
                            couplings[k] * downs[k]
            ans.append(fw)
        for (k1, k2), dk in frequecies.items():
            if k1 == k2:
                ans.append({k_ends[k1]: dk * nums[k]})
            else:
                ans.append({k_ends[k1]: dk * ups[k1], k_ends[k2]: downs[k2]})
        return ans

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
            _i_end = self.sys_ends[_i]
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
            new_valuation[p_node] = opt_transform(opt_q, model[p_node], 1, i)
            model.update(new_valuation)

        # Initialize the bath part.
        for k_end in chain(*self.bath_ends):
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


class FrameFactory:
    prefix = '[HS]'

    def __init__(self, sys_dof: int, bath_dofs: list[int]) -> None:
        self.n_bath = len(bath_dofs)
        self.sys_dof = sys_dof  # type: int
        self.bath_dofs = bath_dofs  # type: list[int]
        self.dof = sys_dof + sum(bath_dofs)
        self.sys_ends = [
            End(self.prefix + f"i-{_i}") for _i in range(sys_dof)
        ]  # type: list[End]
        self.bath_ends = [[End(self.prefix + f"{n}-{k}") for k in range(dof)]
                          for n, dof in enumerate(bath_dofs)
                          ]  # type: list[list[End]]
        self.ends = list(chain(self.sys_ends,
                               *self.bath_ends))  # type: list[End]

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
