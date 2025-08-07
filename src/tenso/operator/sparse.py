# coding: utf-8
from __future__ import annotations
from math import ceil, prod
import threading
from typing import Callable, Generator, Literal
from itertools import combinations

from tenso.libs.backend import (OptArray, opt_dtype, opt_odeint, opt_linalg,
                                  opt_multitransform, opt_trace, opt_ones_like,
                                  opt_zeros_like, opt_transform, opt_tensordot,
                                  opt_svd, opt_cat, opt_split, opt_stack,
                                  opt_eye, opt_maximum, opt_inner_product)
from tenso.state.pureframe import End, Node, Frame, Point
from tenso.state.puremodel import Model

# For debugging
import numpy as np
import torch


def _stack_orth(tensor1: OptArray,
                tensor2: OptArray,
                axis: int,
                _dummy=True) -> OptArray:
    """Stack the tensor1 and tensor2 along axis."""
    # assume tensor1 is already orthogonalized
    return tensor1
    shape1 = list(tensor1.shape)
    shape2 = list(tensor2.shape)
    shape = list(shape1)
    shape.pop(axis)
    mat1 = tensor1.moveaxis(axis, -1).reshape((-1, shape1[axis]))
    mat2 = tensor2.moveaxis(axis, -1).reshape((-1, shape2[axis]))
    # print(mat1.shape, mat2.shape)
    new_mat = torch.hstack((mat1, mat2))
    orth_mat = torch.linalg.qr(new_mat)[0]
    ans = orth_mat.reshape(shape + [-1]).moveaxis(-1, axis)
    return ans


def _truncate(tensor: OptArray, rank: int, axis: int):
    """Truncate the tensor along axis."""
    shape = list(tensor.shape)
    dim = shape.pop(axis)
    dim_left = prod(shape)
    mat = tensor.moveaxis(axis, -1).reshape((dim_left, dim))[:, :rank]
    return mat.reshape(shape + [rank]).moveaxis(-1, axis)


def _find_truncate_index(s: OptArray, atol: float) -> int:
    # # This is the original cummulatve method
    # total_error = 0.0
    # for _k, s_k in reversed(list(enumerate(s))):
    #     total_error += s_k
    #     if total_error > atol:
    #         break
    # return _k + 1
    for _k, s_k in reversed(list(enumerate(s))):
        if s_k > atol:
            break
    return _k + 1


def _one_site_split(array: OptArray, i: int) -> tuple[OptArray, OptArray]:
    shape = list(array.shape)
    l_shape = shape[:i] + shape[i + 1:]
    dim = shape[i]

    mat_p = array.moveaxis(i, -1).reshape((-1, dim))
    u, s, vh = opt_svd(mat_p)
    edge_array = s.to(opt_dtype)[:, None] * vh
    p_tensor = u.reshape(l_shape + [-1]).moveaxis(-1, i)

    return p_tensor, edge_array


def _one_site_merge(array: OptArray, j: int, from_: OptArray) -> OptArray:
    return opt_transform(from_, array, 1, j)


def _two_site_merge(state: Model, p: Node, i: int, q: Node,
                    j: int) -> OptArray:
    return opt_tensordot(state[p], state[q], axes=([i], [j]))



def _partitions(lst: list[int], order: int) -> list[tuple[list[int], list[int]]]:
    p1 = [list(c) for c in combinations(lst, order)]
    p2 = [list(p for p in p1i if not p in lst) for p1i in p1]
    ans = list(zip(p1, p2))
    return ans


def _adaptive_two_site_split(
        state: Model,
        p: Node,
        i: int,
        q: Node,
        j: int,
        from_: OptArray,
        target_rank: None | int = None,
        atol: None | float = None,
    ) -> tuple[OptArray, list[int], OptArray, list[int]]:
    p_shape = state.shape(p) 
    p_ord = state.order(p)
    q_shape = state.shape(q)
    q_ord = state.order(q) 
    full_shape = p_shape[:i] + p_shape[i + 1:] + q_shape[:j] + q_shape[j + 1:]
    full_axis_list = list(range(len(full_shape)))
    assert len(full_axis_list) == p_ord + q_ord - 2
    partitions =  _partitions(full_axis_list, p_ord)
    final_entropy = None
    final_l_shape = None
    final_r_shape = None
    final_p_axes = None
    final_q_axes = None
    for l_axes, r_axes in partitions: 
        l_shape = [full_shape[ax] for ax in l_axes]
        r_shape = [full_shape[ax] for ax in l_axes]
        mat = from_.permute(l_axes + r_axes).reshape(prod(l_shape), prod(r_shape))
        u, s, vh = opt_svd(mat)
        entropy_i = (s**4).sum().item() / (s**2).sum().item()
        if final_entropy is None or entropy_i < final_entropy:
            final_entropy = entropy_i
            final_p_axes = l_axes
            final_q_axes = r_axes
            final_l_shape = l_shape
            final_r_shape = r_shape
            l_mat = u.reshape(l_shape + [-1]).moveaxis(-1, i)
            r_mat = s.to(opt_dtype)[:, None] * vh.reshape([-1] + r_shape).moveaxis(0, j)
        
        rank_s = len(s)
        # Compress the rank according to atol and enlarge to target_rank
        # but no larger than rank_s
        rank_atol = rank_s if atol is None else _find_truncate_index(s, atol) 
        rank = min(
            rank_s,
            rank_atol if target_rank is None else max(target_rank, rank_atol))
        l_mat = u[:, :rank]
        r_mat = s.to(opt_dtype)[:rank, None] * vh[:rank, :]

        p_array = l_mat.reshape(final_l_shape + [rank]).moveaxis(-1, i)
        q_array = r_mat.reshape([rank] + final_r_shape).moveaxis(0, j)
        final_p_axes.insert(i, len(final_p_axes))
        final_q_axes.insert(j, len(final_q_axes))
    return p_array, final_p_axes, q_array, final_q_axes


def _modify_frame(frame: Frame, p, p_axes, q, q_axes) -> Frame:
    """
    Modify the frame to include the new axes for p and q.
    """
    p_neighbors = frame._neighbor[p]
    q_neighbors = frame._neighbor[q]
    all_neighbors = p_neighbors + q_neighbors
    new_p_neighbors = [all_neighbors[i] for i in p_axes]
    new_q_neighbors = [all_neighbors[i] for i in q_axes]

    new_frame = frame.copy()
    new_frame._neighbor[p] = new_p_neighbors
    new_frame._neighbor[q] = new_q_neighbors
    return new_frame


def _two_site_split(state: Model,
                    p: Node,
                    i: int,
                    q: Node,
                    j: int,
                    from_: OptArray,
                    target_rank: None | int = None,
                    atol: None | float = None,
                    ratio: None | float = None) -> tuple[OptArray, OptArray]:
    p_shape = state.shape(p)
    l_shape = p_shape[:i] + p_shape[i + 1:]
    q_shape = state.shape(q)
    r_shape = q_shape[:j] + q_shape[j + 1:]

    u, s, vh = opt_svd(from_.reshape(prod(l_shape), prod(r_shape)))
    rank_s = len(s)
    # Compress the rank according to atol and enlarge to target_rank
    # but no larger than rank_s
    rank_atol = rank_s if atol is None else _find_truncate_index(s, atol)
    if ratio is not None:
        rank_atol = ceil(rank_atol * ratio)
    rank = min(
        rank_s,
        rank_atol if target_rank is None else max(target_rank, rank_atol))
    l_mat = u[:, :rank]
    r_mat = s.to(opt_dtype)[:rank, None] * vh[:rank, :]

    p_array = l_mat.reshape(l_shape + [rank]).moveaxis(-1, i)
    q_array = r_mat.reshape([rank] + r_shape).moveaxis(0, j)
    return p_array, q_array


class SparseSPO:

    def __init__(self,
                 op_list: list[dict[End, OptArray]],
                 f_list: None
                 | Callable[[float], list[dict[End, OptArray]]] = None,
                 initial_time: float = 0.0) -> None:

        dims = dict()
        # Check the consistency of the dimensions.
        n_ti = len(op_list)
        for term in op_list:
            for e, a in term.items():
                assert a.ndim == 2 and a.shape[0] == a.shape[1]
                if e in dims:
                    assert dims[e] == a.shape[0]
                else:
                    dims[e] = a.shape[0]
        if f_list is None:
            n_td = 0
        else:
            td_list = f_list(initial_time)
            n_td = len(td_list)
            for td_term in td_list:
                for e, a in td_term.items():
                    assert a.ndim == 2 and a.shape[0] == a.shape[1]
                    if e in dims:
                        assert dims[e] == a.shape[0]
                    else:
                        dims[e] = a.shape[0]

        self.n_ti = n_ti
        self.op_list = op_list
        self.n_td = n_td
        self.f_list = f_list
        self.dims = dims  # type: dict[End, int]
        return

    def __add__(self, other: SparseSPO) -> SparseSPO:

        def f_list(t):
            return self.f_list(t) + other.f_list(t)

        return SparseSPO(self.op_list + other.op_list, f_list)

    def get_ti_terms(self) -> list[dict[End, OptArray]]:
        return self.op_list

    def get_td_terms(self, t: float) -> list[dict[End, OptArray]]:
        if self.f_list is None:
            return []
        else:
            return self.f_list(t)

    @property
    def ends(self) -> set[End]:
        return set(self.dims.keys())


class SPOKet:
    """
    Intermediate class for the sparse operations in the model.
    """

    def __init__(self,
                 op: SparseSPO,
                 state: Model,
                 frame: Frame,
                 root: Node,
                 time=0.0):
        self.frame = frame
        self.root = root
        self._node_link_visitor = frame.node_link_visitor(root)

        self.op_list = op.get_ti_terms() + op.get_td_terms(time)
        self.is_time_dependent = (op.n_td > 0)
        # Initialize the new valuation
        self.original_state = state
        self.state_list = self._operate()
        return

    def _operate(self) -> list[Model]:
        frame = self.frame
        state = self.original_state
        if not self.op_list:
            ans = [state]
        else:
            state_list = []
            # each term as one element in the state_list
            for term in self.op_list:
                tmp_model = state.copy()
                for e, a in term.items():
                    dual_node, dual_i = frame.dual(e, None)
                    new_a = opt_transform(a, tmp_model[dual_node], 1, dual_i)
                    tmp_model.update({dual_node: new_a})
                state_list.append(tmp_model)
            ans = state_list
        return ans

    def canonicalize(self) -> None:
        """Canonicalize the state."""
        _move = self._move
        for p, i, q, j in self._node_link_visitor:
            for _s in self.state_list:
                _move(_s, p, i, q, j)
        return

    def _move(self, state: Model, p: Node, i: int, q: Node, j: int) -> None:
        # Split the edge_array and update p
        l_tensor, edge_array = _one_site_split(state[p], i)
        r_tensor = state[q]

        # Merge and update q
        new_root = _one_site_merge(r_tensor, j, edge_array)
        state.update({p: l_tensor, q: new_root})
        self.root = q
        return

    def close_with_bra(self, bra: Model | None = None) -> complex:
        """Calculate the inner product."""
        if bra is None:
            bra = self.original_state.conjugate()

        inner_product = 0.0
        for op_ket in self.state_list:
            sip = SparseSandwich(self.frame, self.root, op_ket, bra)
            inner_product += sip.forward()
        return inner_product

    def close_with_conj(self, bras: SPOKet) -> complex:
        """Calculate the inner product."""
        assert len(self.state_list) == len(bras.state_list)
        inner_product = 0.0
        for op_ket, op_bra in zip(self.state_list, bras.state_list):
            sip = SparseSandwich(self.frame, self.root, op_ket,
                                 op_bra.conjugate())
            inner_product += sip.forward()
        return inner_product


class ListModelInnerProduct:

    def __init__(self,
                 frame: Frame,
                 root: Node,
                 ket_states: list[Model],
                 bra_states: list[Model] | None = None):

        self.frame = frame
        self.root = root
        self.ket_states = ket_states
        if bra_states is None:
            self.bra_states = [ket for ket in ket_states]
        else:
            self.bra_states = bra_states
        return

    def forward(self) -> complex:
        """Actuall calculate the inner product."""
        inner_product = 0.0
        for ket, bra in zip(self.ket_states, self.bra_states):
            sip = SparseSandwich(self.frame, self.root, ket, bra)
            inner_product += sip.forward()
        return inner_product


class SparseSandwich:

    def __init__(self,
                 frame: Frame,
                 root: Node,
                 ket_state: Model,
                 bra_state: Model | None = None,
                 op: SparseSPO | None = None,
                 time=0.0):
        if op is None:
            op = SparseSPO([{}])
        if bra_state is None:
            bra_state = ket_state
        self.op = op
        self.frame = frame
        self.root = root
        self.ket = ket_state
        self.bra = bra_state.conjugate()
        # for n in frame.nodes:
        #     print(n,
        #           self.ket[n].flatten()[0],
        #           self.bra[n].flatten()[0],
        #           flush=True)

        # Construct pools for the each term
        n_ti, n_td = op.n_ti, op.n_td
        self.n_terms = n_ti + n_td
        if n_ti > 0:
            self._ti_eoms = [
                _ComplexSingleTerm(frame, root, self.bra,
                                   op.get_ti_terms()[n], self.ket)
                for n in range(n_ti)
            ]  # type: list[_ComplexSingleTerm]
        else:
            self._ti_eoms = [
                _ComplexSingleTerm(frame, root, self.bra, dict(), self.ket)
            ]  # type: list[_ComplexSingleTerm]
        self._td_eoms = [
            _ComplexSingleTerm(frame, root, self.bra,
                               op.get_td_terms(time)[n], self.ket)
            for n in range(n_td)
        ]  # type: list[_ComplexSingleTerm]

        return

    def forward(self) -> complex:
        """Calculate the inner product."""
        eoms = self._ti_eoms + self._td_eoms
        for eom in eoms:
            eom.update_mean_fields()
        inner_product = 0.0
        for eom in eoms:
            # print(eom.expectation_value, flush=True)
            inner_product += eom.expectation_value
        return inner_product


class _ComplexSingleTerm:
    r"""Calculate the matrix element value of a single term.
       < bra_state | [op] | ket_state >
    """

    def __init__(
        self,
        frame: Frame,
        root: Node,
        bra_state: Model,
        op: dict[End, OptArray],
        ket_state: Model,
    ) -> None:
        """
        Initialize the SparseOperator class.

        Args:
            op (dict[End, OptArray]): The operator dictionary.
            frame (Frame): The frame of the model.
            root (Node): The root node of the model.
            ket_state (Model): The ket state of the model.
            bra_state (Model): The bra state of the model. (complex conjugate included)

        Returns:
            None
        """
        self.frame = frame
        self.root = root
        self.ket_state = ket_state
        self.bra_state = bra_state

        # Cache for mean fields
        self.mean_fields = dict(
        )  # type: dict[tuple[Node, int], None | OptArray]
        self.densities = dict()  # type: dict[End, OptArray]
        self.expectation_value = None
        self.node_axes = frame.get_node_axes(
            root)  # type: dict[Node, None | int]

        self._dual = frame.dual
        self._node_visitor = frame.node_visitor(
            start=root, method='BFS')  # type: list[Node]
        self._point_visitor = self.frame.point_visitor(
            start=root, method='DFS')  # type: list[Point]

        self.update_primitive_mean_fields(op)
        return

    def update_primitive_mean_fields(self, op: dict[End, OptArray]) -> None:
        """
        Update the mean fields of the primitive operator.

        Args:
            op (dict[End, OptArray]): The operator dictionary containing the mean fields.

        Returns:
            None
        """
        dual = self._dual
        # print({e: a.shape for e, a in op.items()})
        for q in self.frame.ends:
            self.mean_fields[dual(q, None)] = op[q] if q in op else None
        return

    def get_node_mean_field(self, bra_a: OptArray, ket_a: OptArray, p: Node,
                            i: int) -> None | OptArray:
        """
        Calculate the mean field with a specific node.

        Parameters:
            a (OptArray): The array for which the mean field is calculated.
            p (Node): The node for which the mean field is calculated.
            i (int): The index of the node to be excluded from the mean field calculation.

        Returns:
            None or OptArray: The calculated mean field if there are other nodes in the mean field calculation,
            otherwise None.
        """
        order = self.frame.degree(p)
        mfs = self.mean_fields

        op_dict = dict()
        for _i in range(order):
            if _i != i:
                _m = mfs[p, _i]
                if _m is not None:
                    op_dict[_i] = _m
                else:
                    op_dict[_i] = opt_eye(bra_a.shape[_i], ket_a.shape[_i])

        ans = opt_trace(bra_a, opt_multitransform(op_dict, ket_a), i)
        return ans

    def update_mean_fields(self) -> None:
        """From leaves to the root."""
        ket_state = self.ket_state
        bra_state = self.bra_state
        # Calculate mean fields for each node from leaves to root
        for p in reversed(self._node_visitor):
            i = self.node_axes[p]
            ket_a = ket_state[p]
            bra_a = bra_state[p]
            if i is not None:
                self.mean_fields[self._dual(p, i)] = self.get_node_mean_field(
                    bra_a, ket_a, p, i)
            else:
                op_dict = {
                    _i: _m
                    for _i in range(self.frame.degree(p))
                    if (_m := self.mean_fields[p, _i]) is not None
                }
                if op_dict:
                    ans = opt_inner_product(bra_a,
                                            opt_multitransform(op_dict, ket_a))
                else:
                    ans = opt_inner_product(bra_a, ket_a)
                self.expectation_value = ans
        return

    def get_mean_holes(self) -> None:
        """From leaves to the root."""
        ket_state = self.ket_state
        bra_state = self.bra_state

        # From root to leaves
        for p in self._node_visitor[1:]:
            i = self.node_axes[p]
            assert i is not None
            q, j = self._dual(p, i)
            ket_a = ket_state[q]
            bra_a = bra_state[q]
            self.mean_fields[p,
                             i] = self.get_node_mean_field(bra_a, ket_a, q, j)
        return

    def get_density(self) -> None:
        """Calculate the density matrix."""
        ket_state = self.ket_state
        bra_state = self.bra_state
        for e in self.frame.ends:
            p, i = self._dual(e, None)
            ket_a = ket_state[p]
            bra_a = bra_state[p]
            hole = self.mean_fields[p, i]
            self.densities[e] = opt_trace(bra_a,
                                          opt_transform(hole, ket_a, 0, i), i)
        return


class _SVDInfoCache:
    """
    SVD infos for shifted root.
    """

    def __init__(self, frame: Frame, root: Node):
        self.frame = frame

        # Cache for SVD infos
        self.u = dict()  # type: dict[Node, OptArray | None]
        self.s = dict()  # type: dict[Node, OptArray | None]
        self.vh = dict()  # type: dict[Node, OptArray | None]

        # Frame helpers
        self._dual = frame.dual
        self._node_axes = frame.get_node_axes(
            root)  # type: dict[Node, None | int]
        self._node_visitor = frame.node_visitor(
            start=root, method='BFS')  # type: list[Node]
        return

    def _print_keys(self):
        for k in self.s.keys():
            print(k, end=', ')
        print('\n', flush=True)
        return

    def _get_node_svd(self, a: OptArray, p: Node,
                      i: int) -> tuple[OptArray, OptArray, OptArray]:
        ax = self._node_axes[p]

        if ax is not None:
            ax_op = self.s[p].to(opt_dtype)[:, None] * self.vh[p]
            a = opt_transform(ax_op, a, 1, ax)

        shape = list(a.shape)
        l_shape = shape[:i] + shape[i + 1:]
        u, s, vh = opt_svd(a.moveaxis(i, -1).reshape((-1, shape[i])))
        u = u.reshape(l_shape + [-1]).moveaxis(-1, i)
        return u, s, vh

    def update(self, state: Model) -> None:
        """From root to leaves."""
        axes = self._node_axes
        dual = self._dual

        # Update svd infos first
        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                u, s, vh = self._get_node_svd(state[q], q, j)
                self.u[p] = u
                self.s[p] = s
                self.vh[p] = vh
            else:
                self.u[p] = None
                self.s[p] = None
                self.vh[p] = None
        return


class _ProdEOM:
    r"""Solve the equation wrt Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self,
                 op: dict[End, OptArray],
                 frame: Frame,
                 root: Node,
                 reg_atol: float | None = None,
                 reg_method: Literal['truncate', 'extend'] = 'extend',
                 reg_type: Literal['mf', 'mf2', 'ip'] = 'ip') -> None:
        """
        Initialize a _RegSingleTerm.

        Args:
            op (dict[End, OptArray]): The operator dictionary.
            state (CannonialModel): The state of the model.
            frame (Frame): The frame of the model.
            root (Node): The root node of the model.

        Returns:
            None
        """
        self.frame = frame
        self.reg_method = reg_method
        self.reg_type = reg_type
        self.reg_atol = reg_atol

        # Cache for mean fields
        self.mean_fields = dict(
        )  # type: dict[tuple[Node, int], None | OptArray] # noqa

        # Cache for C* adjointness
        self._reg_mean_fields = dict()  # type: dict[Node, None | OptArray]
        self._reg_inner_products = dict()  # type: dict[Node, OptArray]

        # Frame helpers
        self.node_axes = frame.get_node_axes(
            root)  # type: dict[Node, None | int]
        self._dual = frame.dual
        self._node_visitor = frame.node_visitor(
            start=root, method='BFS')  # type: list[Node]
        self._order = frame.degree

        self.update_primitive_mean_fields(op)
        return

    def update_primitive_mean_fields(self, op: dict[End, OptArray]) -> None:
        """
        Update the mean fields of the primitive operator.

        Args:
            op (dict[End, OptArray]): The operator dictionary containing the mean fields.

        Returns:
            None
        """
        dual = self._dual
        # print({e: a.shape for e, a in op.items()})
        for q in self.frame.ends:
            self.mean_fields[dual(q, None)] = op[q] if q in op else None
        return

    def node_eom(self,
                 node: Node,
                 array: OptArray,
                 s: None | OptArray = None,
                 vh: None | OptArray = None) -> OptArray:
        """
        Calculate the equation of motion for a given node.

        Args:
            node (Node): The node for which to calculate the equation of motion.
            a (OptArray): The array of the node.
            s (OptArray): The array of the adjoint singular values
            vh (OptArray): The array of the adjoint basis

        Returns:
            OptArray: The result of the equation of motion calculation.
        """
        dual = self._dual
        ax = self.node_axes[node]
        mfs = self.mean_fields
        op_list = {
            i: _m
            for i in range(array.ndim)
            if i != ax and (_m := mfs[node, i]) is not None
        }

        # if op_list is empty, the projection is zero
        if not op_list:
            return array if ax is None else opt_zeros_like(array)

        ans = opt_multitransform(op_list, array)
        if ax is not None:
            assert s is not None and vh is not None
            # Swap-style projection
            projection = opt_transform(mfs[dual(node, ax)], array, 0, ax)
            ans -= projection
            # Calculate C* adjointness
            csadj = self.get_node_adjointness(node, s, vh)
            if csadj is not None:
                if (csadj != csadj).any():
                    raise RuntimeError('C*Adjointness has NaN.')
                ans = opt_transform(csadj, ans, 1, ax)
        return ans

    def get_krylov(self,
                   a: OptArray,
                   p: Node,
                   i: int,
                   power: int = 1) -> OptArray:
        order = self._order(p)
        mfs = self.mean_fields
        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and (_m := mfs[p, _i]) is not None
        }
        if op_dict:
            for _ in range(power):
                a = opt_multitransform(op_dict, a)
        return a

    def get_node_mean_field(self, a: OptArray, p: Node,
                            i: int) -> None | OptArray:
        """
        Calculate the mean field with a specific node.

        Parameters:
            a (OptArray): The array for which the mean field is calculated.
            p (Node): The node for which the mean field is calculated.
            i (int): The index of the node to be excluded from the mean field calculation.

        Returns:
            None or OptArray: The calculated mean field if there are other nodes in the mean field calculation,
            otherwise None.

        Note:
            a is assumed to be semi-unitary along i axis.
        """
        order = self.frame.degree(p)
        mfs = self.mean_fields
        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and (_m := mfs[p, _i]) is not None
        }
        if op_dict:
            conj_a = a.conj()
            ans = opt_trace(conj_a, opt_multitransform(op_dict, a), i)
        else:
            ans = None
        return ans

    def update_mean_fields(self, state: Model) -> None:
        """From leaves to the root."""
        axes = self.node_axes
        dual = self._dual
        mf = self.get_node_mean_field

        for q in reversed(self._node_visitor):
            j = axes[q]
            if j is not None:
                self.mean_fields[dual(q, j)] = mf(state[q], q, j)
        return

    def _get_node_reg_mf(self, a: OptArray, p: Node,
                         i: int) -> None | OptArray:
        """
        Calculate the mean field with a specific node using the SVD information 
        such that the root is pretend to at (p, i).

        Parameters:
            a (OptArray): The array U from the SVD associated to node `p`.
            p (Node): The node for which the mean field is calculated.
            i (int): The index of the node to be excluded from the mean field calculation.

        Returns:
            None or OptArray: The calculated mean field if there are other nodes in the mean field calculation,
            otherwise None.

        Note:
            a is assumed to be semi-unitary along i axis.
        """
        order = self._order(p)
        ax = self.node_axes[p]
        mfs = self.mean_fields
        reg_mfs = self._reg_mean_fields

        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and _i != ax and (_m := mfs[p, _i]) is not None
        }
        if ax is not None and (_rm := reg_mfs[p]) is not None:
            op_dict[ax] = _rm
        if op_dict:
            conj_a = a.conj()
            ans = opt_trace(conj_a, opt_multitransform(op_dict, a), i)
        else:
            ans = None
        return ans

    def _update_reg_mfs(self, svd_info: _SVDInfoCache) -> None:
        """From root to leaves."""
        axes = self.node_axes
        dual = self._dual
        reg_mfs = self._reg_mean_fields

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                reg_mfs[p] = self._get_node_reg_mf(svd_info.u[p], q, j)
        return

    def _get_node_reg_ip(self, a: OptArray, u: OptArray, p: Node,
                         i: int) -> OptArray:
        order = self._order(p)
        ax = self.node_axes[p]
        mfs = self.mean_fields
        reg_ips = self._reg_inner_products

        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and _i != ax and (_m := mfs[p, _i]) is not None
        }
        if ax is not None:
            op_dict[ax] = reg_ips[p]

        r = opt_trace(u.conj(), opt_multitransform(op_dict, a), i)
        return r

    def _update_reg_ips(self, state: Model, svd_info: _SVDInfoCache) -> None:
        """From root to leaves."""
        axes = self.node_axes
        dual = self._dual
        reg_ip = self._reg_inner_products

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                reg_ip[p] = self._get_node_reg_ip(state[q], svd_info.u[p], q,
                                                  j)
        return

    def update_adjointness(self, state: Model,
                           svd_info: _SVDInfoCache) -> None:
        """From root to leaves."""
        if self.reg_type in {'mf', 'mf2'}:
            self._update_reg_mfs(svd_info)
        elif self.reg_type == 'ip':
            self._update_reg_ips(state, svd_info)
        return

    def get_node_adjointness(self, node: Node, s: OptArray,
                             vh: OptArray) -> None | OptArray:
        reg_atol = self.reg_atol
        reg_method = self.reg_method
        reg_type = self.reg_type
        # C*Adjointness with regularization
        #print(f'({self.reg_method}) s: ', s[:4], flush=True)
        if reg_atol is not None:
            if reg_method == 'extend':
                reg_s = opt_maximum(s, reg_atol * opt_ones_like(s))
                inv_s = (1.0 / reg_s)
            elif reg_method == 'truncate':
                # [Experimental]
                reg_s = s.where(s > reg_atol, 0.0)
                inv_s = (1.0 / reg_s).nan_to_num(nan=0.0,
                                                 neginf=0.0,
                                                 posinf=0.0)
        else:
            reg_s = s
            inv_s = 1.0 / reg_s
        #print(f'({self.reg_method}) after:', reg_s[:4], flush=True)
        inv_s.to(opt_dtype)
        reg_s.to(opt_dtype)
        s.to(opt_dtype)
        if reg_type == 'ip':
            mid_mat = inv_s[:, None] * self._reg_inner_products[node]
            csadj = vh.conj().T @ mid_mat
        elif reg_type == 'mf':
            mf = self._reg_mean_fields[node]
            if mf is None:
                csadj = None
            else:
                mid_mat = inv_s[:, None] * s[None, :] * mf
                csadj = vh.conj().T @ mid_mat @ vh
        elif reg_type == 'mf2':
            mf = self._reg_mean_fields[node]
            if mf is None:
                csadj = None
            else:
                mid_mat = inv_s[:, None] * reg_s[None, :] * mf
                csadj = vh.conj().T @ mid_mat @ vh
        else:
            raise NotImplementedError
        return csadj


class _DirectProdEOM:
    r"""Solve the equation wrt Product operator:
        d/dt |state> = [op] |state>
    """

    def __init__(self,
                 op: dict[End, OptArray],
                 frame: Frame,
                 root: Node,
                 reg_atol: float | None = None,
                 reg_method: Literal['truncate', 'extend'] = 'extend',
                 reg_type: Literal['mf', 'mf2', 'ip'] = 'ip') -> None:
        """
        Initialize a _RegSingleTerm.

        Args:
            op (dict[End, OptArray]): The operator dictionary.
            state (CannonialModel): The state of the model.
            frame (Frame): The frame of the model.
            root (Node): The root node of the model.

        Returns:
            None
        """
        self.frame = frame
        self.reg_method = reg_method
        self.reg_type = reg_type
        self.reg_atol = reg_atol

        # Cache for mean fields
        self.mean_fields = dict(
        )  # type: dict[tuple[Node, int], None | OptArray] # noqa

        # Cache for regularizations
        self._reg_mean_fields = dict()  # type: dict[Node, None | OptArray]
        self._reg_inner_products = dict()  # type: dict[Node, OptArray]

        # Cache for SVD infos
        self._reg_s = dict()  # type: dict[Node, OptArray]
        self._reg_v = dict()  # type: dict[Node, OptArray]

        # Frame helpers
        self.node_axes = frame.get_node_axes(
            root)  # type: dict[Node, None | int]
        self._dual = frame.dual
        self._node_visitor = frame.node_visitor(
            start=root, method='BFS')  # type: list[Node]
        self._order = frame.degree

        self.update_primitive_mean_fields(op)
        return

    def update_primitive_mean_fields(self, op: dict[End, OptArray]) -> None:
        """
        Update the mean fields of the primitive operator.

        Args:
            op (dict[End, OptArray]): The operator dictionary containing the mean fields.

        Returns:
            None
        """
        dual = self._dual
        # print({e: a.shape for e, a in op.items()})
        for q in self.frame.ends:
            self.mean_fields[dual(q, None)] = op[q] if q in op else None
        return

    def node_eom(self, node: Node, array: OptArray) -> OptArray:
        """
        Calculate the equation of motion for a given node.

        Args:
            node (Node): The node for which to calculate the equation of motion.
            a (OptArray): The arry of the node.

        Returns:
            OptArray: The result of the equation of motion calculation.
        """
        dual = self._dual
        ax = self.node_axes[node]
        mfs = self.mean_fields
        op_list = {
            i: _m
            for i in range(array.ndim)
            if i != ax and (_m := mfs[node, i]) is not None
        }

        # if op_list is empty, the projection is zero
        if not op_list:
            return array if ax is None else opt_zeros_like(array)

        ans = opt_multitransform(op_list, array)
        if ax is not None:
            # Swap-style projection
            projection = opt_transform(mfs[dual(node, ax)], array, 0, ax)
            ans -= projection
            # # Projection considering possible change in state[node]
            # p = trace(ans, array.conj(), ax)
            # projection = single_tensordot(p, array, 1, ax)
            csadj = self.get_node_adjointness(node)
            if csadj is not None:
                if (csadj != csadj).any():
                    raise RuntimeError('C*Adjointness has NaN.')
                ans = opt_transform(csadj, ans, 1, ax)
        return ans

    def get_node_adjointness(self, node: Node) -> None | OptArray:
        reg_atol = self.reg_atol
        reg_method = self.reg_method
        reg_type = self.reg_type
        # C*Adjointness with regularization
        s = self._reg_s[node]
        vh = self._reg_v[node]
        #print(f'({self.reg_method}) s: ', s[:4], flush=True)
        if reg_atol is not None:
            if reg_method == 'extend':
                reg_s = opt_maximum(s, reg_atol * opt_ones_like(s))
                inv_s = (1.0 / reg_s)
            elif reg_method == 'truncate':
                # [Experimental]
                reg_s = s.where(s > reg_atol, 0.0)
                inv_s = (1.0 / reg_s).nan_to_num(nan=0.0,
                                                 neginf=0.0,
                                                 posinf=0.0)
        else:
            reg_s = s
            inv_s = 1.0 / reg_s
        #print(f'({self.reg_method}) after:', reg_s[:4], flush=True)
        inv_s.to(opt_dtype)
        reg_s.to(opt_dtype)
        s.to(opt_dtype)
        if reg_type == 'ip':
            mid_mat = inv_s[:, None] * self._reg_inner_products[node]
            csadj = vh.conj().T @ mid_mat
        elif reg_type == 'mf':
            mf = self._reg_mean_fields[node]
            if mf is None:
                csadj = None
            else:
                mid_mat = inv_s[:, None] * s[None, :] * mf
                csadj = vh.conj().T @ mid_mat @ vh
        elif reg_type == 'mf2':
            mf = self._reg_mean_fields[node]
            if mf is None:
                csadj = None
            else:
                mid_mat = inv_s[:, None] * reg_s[None, :] * mf
                csadj = vh.conj().T @ mid_mat @ vh
        else:
            raise NotImplementedError
        return csadj

    def get_krylov(self,
                   a: OptArray,
                   p: Node,
                   i: int,
                   power: int = 1) -> OptArray:
        order = self._order(p)
        mfs = self.mean_fields
        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and (_m := mfs[p, _i]) is not None
        }
        if op_dict:
            for _ in range(power):
                a = opt_multitransform(op_dict, a)
        return a

    def get_node_mean_field(self, a: OptArray, p: Node,
                            i: int) -> None | OptArray:
        """
        Calculate the mean field with a specific node.

        Parameters:
            a (OptArray): The array for which the mean field is calculated.
            p (Node): The node for which the mean field is calculated.
            i (int): The index of the node to be excluded from the mean field calculation.

        Returns:
            None or OptArray: The calculated mean field if there are other nodes in the mean field calculation,
            otherwise None.

        Note:
            a is assumed to be semi-unitary along i axis.
        """
        order = self.frame.degree(p)
        mfs = self.mean_fields
        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and (_m := mfs[p, _i]) is not None
        }
        if op_dict:
            conj_a = a.conj()
            ans = opt_trace(conj_a, opt_multitransform(op_dict, a), i)
        else:
            ans = None
        return ans

    def update_mean_fields(self, state: Model) -> None:
        """From leaves to the root."""
        axes = self.node_axes
        dual = self._dual
        mf = self.get_node_mean_field

        for q in reversed(self._node_visitor):
            j = axes[q]
            if j is not None:
                self.mean_fields[dual(q, j)] = mf(state[q], q, j)
        return

    def _get_node_svd(self, a: OptArray, p: Node,
                      i: int) -> tuple[OptArray, OptArray, OptArray]:
        ax = self.node_axes[p]

        if ax is not None:
            ax_op = self._reg_s[p].to(opt_dtype)[:, None] * self._reg_v[p]
            a = opt_transform(ax_op, a, 1, ax)

        shape = list(a.shape)
        l_shape = shape[:i] + shape[i + 1:]
        u, s, v = opt_svd(a.moveaxis(i, -1).reshape((-1, shape[i])))
        u = u.reshape(l_shape + [-1]).moveaxis(-1, i)
        return u, s, v

    def _get_node_reg_mf(self, a: OptArray, p: Node,
                         i: int) -> None | OptArray:
        """
        Calculate the mean field with a specific node using the SVD information 
        such that the root is pretend to at (p, i).

        Parameters:
            a (OptArray): The array U from the SVD associated to node `p`.
            p (Node): The node for which the mean field is calculated.
            i (int): The index of the node to be excluded from the mean field calculation.

        Returns:
            None or OptArray: The calculated mean field if there are other nodes in the mean field calculation,
            otherwise None.

        Note:
            a is assumed to be semi-unitary along i axis.
        """
        order = self._order(p)
        ax = self.node_axes[p]
        mfs = self.mean_fields
        reg_mfs = self._reg_mean_fields

        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and _i != ax and (_m := mfs[p, _i]) is not None
        }
        if ax is not None and (_rm := reg_mfs[p]) is not None:
            op_dict[ax] = _rm
        if op_dict:
            conj_a = a.conj()
            ans = opt_trace(conj_a, opt_multitransform(op_dict, a), i)
        else:
            ans = None
        return ans

    def _get_csadj_reg_mf(self, state: Model) -> None:
        """From root to leaves."""
        axes = self.node_axes
        dual = self._dual

        # Update svd infos first
        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                u, s, v = self._get_node_svd(state[q], q, j)
                self._reg_s[p] = s
                self._reg_v[p] = v
                reg_mf = self._get_node_reg_mf(u, q, j)
                self._reg_mean_fields[p] = reg_mf
        return

    def _get_node_reg_ip(self, a: OptArray, u: OptArray, p: Node,
                         i: int) -> OptArray:
        order = self._order(p)
        ax = self.node_axes[p]
        mfs = self.mean_fields

        op_dict = {
            _i: _m
            for _i in range(order)
            if _i != i and _i != ax and (_m := mfs[p, _i]) is not None
        }
        if ax is not None:
            op_dict[ax] = self._reg_inner_products[p]

        r = opt_trace(u.conj(), opt_multitransform(op_dict, a), i)
        return r

    def _get_csadj_reg_ip(self, state: Model) -> None:
        """From root to leaves."""
        axes = self.node_axes
        dual = self._dual

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                a = state[q]
                u, s, v = self._get_node_svd(a, q, j)
                r = self._get_node_reg_ip(a, u, q, j)
                self._reg_inner_products[p] = r
                self._reg_s[p] = s
                self._reg_v[p] = v
        return

    def update_adjointness(self, state: Model) -> None:
        """From root to leaves."""
        if self.reg_type in {'mf', 'mf2'}:
            self._get_csadj_reg_mf(state)
        elif self.reg_type == 'ip':
            self._get_csadj_reg_ip(state)
        return


class SparsePropagator:
    keyword_settings = [
        'vmf_atol', 'ps2_atol', 'ps2_ratio', 'ode_method', 'ode_atol',
        'ode_rtol', 'vmf_reg_method', 'vmf_reg_type', 'cache_svd_info'
    ]
    vmf_atol = 1.0e-7  # type: float | None
    ps2_atol = 1.0e-7  # type: float | None
    ps2_ratio = 2.0  # type: float | None
    ode_atol = 1.0e-5  # type: float
    ode_rtol = 1.0e-7  # type: float
    vmf_reg_method = 'extend'  # type: Literal['extend', 'truncate']
    vmf_reg_type = 'ip'  # type: Literal['ip', 'mf', 'mf2']
    ode_method = 'dopri5'  # type: Literal['rk4', 'bosh3', 'dopri5', 'dopri8']
    cache_svd_info = True  # type: bool

    @classmethod
    def info(cls):
        return " | ".join(f"{k}: {getattr(cls, k)}"
                          for k in cls.keyword_settings)

    @classmethod
    def update_settings(cls, **kwargs):
        for k, v in kwargs.items():
            if k in cls.keyword_settings:
                setattr(cls, k, v)
            else:
                raise RuntimeError(f'No such setting named as {k}.')
        return

    def __init__(self,
                 op: SparseSPO,
                 state: Model,
                 frame: Frame,
                 root: Node,
                 renormalize_root=False,
                 init_time=0.0) -> None:

        self.ops = op
        self.state = state
        self.frame = frame
        self.root = root
        self.time = init_time

        self._node_visitor = frame.node_visitor(root, method='DFS')
        self._node_link_visitor = frame.node_link_visitor(root)
        self._depths = frame.get_node_depths(root)

        # For vectorized operations
        self._shape_list = []
        self._size_list = []
        self.update_size_list()

        # Construct pools for the EOMs
        n_ti, n_td = op.n_ti, op.n_td
        if self.cache_svd_info:
            _eom_type = _ProdEOM
            self._svd_info = _SVDInfoCache(frame, root)
        else:
            _eom_type = _DirectProdEOM
            self._svd_info = None
        self.n_terms = n_ti + n_td
        self._ti_eoms = [
            _eom_type(op.get_ti_terms()[n],
                      frame,
                      root,
                      reg_atol=self.vmf_atol,
                      reg_type=self.vmf_reg_type,
                      reg_method=self.vmf_reg_method) for n in range(n_ti)
        ]  # type: list[_DirectProdEOM | _ProdEOM]
        self.is_time_dependent = (n_td > 0)
        self._td_eoms = [
            _eom_type(op.get_td_terms(init_time)[n],
                      frame,
                      root,
                      reg_atol=self.vmf_atol,
                      reg_type=self.vmf_reg_type,
                      reg_method=self.vmf_reg_method) for n in range(n_td)
        ]  # type: list[_DirectProdEOM | _ProdEOM]

        # update regularization parameters
        for eom in self._ti_eoms + self._td_eoms:
            eom.reg_atol = self.vmf_atol
            eom.reg_type = self.vmf_reg_type
            eom.reg_method = self.vmf_reg_method

        # misc
        self.ode_step_counter = 0  # type: int
        if renormalize_root:
            self.renormalization_coefficient = 1.0
            self.renormalize_root()
        else:
            self.renormalization_coefficient = None
        return

    def renormalize_root(self):
        root_array = self.state[self.root]
        norm = opt_linalg.norm(root_array.reshape(-1)).item()
        self.renormalization_coefficient *= norm
        self.state.update({self.root: root_array / norm})
        return

    def update_size_list(self):
        self._shape_list = [self.state.shape(p) for p in self._node_visitor]
        self._size_list = [prod(s) for s in self._shape_list]
        return

    def update_td_terms(self, time: float) -> None:
        td_terms = self.ops.get_td_terms(float(time))
        #print(time, td_terms)
        for n, eom in enumerate(self._td_eoms):
            eom.update_primitive_mean_fields(td_terms[n])
        return

    def propagate(
        self,
        end: float,
        dt: float,
        ps_method: Literal['vmf', 'ps1', 'ps2'] = 'vmf'
    ) -> Generator[tuple[float, Model], None, None]:
        if len(self._node_visitor) < 2 or ps_method == 'vmf':
            # For the naive case no need for projector splitting.
            step = self.vmf_step
        elif ps_method == 'ps1':
            step = self.ps1_step
        elif ps_method == 'ps2':
            step = self.ps2_step
        else:
            raise RuntimeError(f'No method named as {ps_method}.')
        while self.time < end:
            yield (self.time, self.state)
            step(dt)
        return

    def adaptive_propagate(
        self,
        end: float,
        fixed_dt: float,
        fixed_ps_method: Literal['ps1', 'vmf'] = 'vmf',
        fixed_steps: int = 1,
        adaptive_dt: float | None = None,
        adaptive_ps_steps: int = 1,
    ) -> Generator[tuple[float, Model], None, None]:
        indices = list()
        for p, i, q, j in self._node_link_visitor:
            if (q, j) not in indices:
                indices.append((p, i))
        if not indices:
            raise RuntimeError('No need for adaptive propagation.')
        # Start with adaptive step
        if adaptive_dt is None:
            adaptive_dt = fixed_dt
        adaptive_step = self.ps2_step
        if fixed_ps_method == 'ps1':
            fix_step = self.ps1_step
        elif fixed_ps_method == 'vmf':
            fix_step = self.vmf_step
        else:
            raise RuntimeError(f'No method named as {fixed_ps_method}.')
        while self.time < end:
            for _ in range(adaptive_ps_steps):
                yield (self.time, self.state)
                adaptive_step(adaptive_dt)
            self.update_size_list()
            for _ in range(fixed_steps):
                yield (self.time, self.state)
                fix_step(fixed_dt)

    def mixed_propagate(
        self,
        end: float,
        dt: float,
        ending_ps_method: Literal['ps2', 'ps1', 'vmf'] = 'ps1',
        starting_dt: float | None = None,
        starting_ps_method: Literal['ps1', 'ps2'] = 'ps2',
        max_starting_rank: int | None = None,
        max_starting_steps: int | None = None,
    ) -> Generator[tuple[float, Model], None, None]:
        indices = list()
        for p, i, q, j in self._node_link_visitor:
            if (q, j) not in indices:
                indices.append((p, i))

        if not indices:
            raise RuntimeError('No need for mixed propagation.')
        if starting_ps_method == 'ps1':
            start_step = self.ps1_step
        elif starting_ps_method == 'ps2':
            start_step = self.ps2_step
        else:
            raise RuntimeError(f'No method named as {starting_ps_method}.')
        if ending_ps_method == 'ps1':
            end_step = self.ps1_step
        elif ending_ps_method == 'vmf':
            end_step = self.vmf_step
        else:
            raise RuntimeError(f'No method named as {ending_ps_method}.')
        if starting_dt is None:
            starting_dt = dt
        _dt = starting_dt
        _step = start_step
        starting = True
        n_steps = 0
        while self.time < end:
            yield (self.time, self.state)

            # Check the rank and time to switch PS method
            if starting:
                rank_cond = max_starting_rank is not None and any(
                    self.state.dimension(p, i) > max_starting_rank
                    for p, i in indices)
                time_cond = max_starting_steps is not None and n_steps >= max_starting_steps
                if rank_cond or time_cond:
                    starting = False
                    _step = end_step
                    _dt = dt
                    self.update_size_list()
                    if __debug__:
                        print(f'Switched to {ending_ps_method}.', flush=True)

            _step(_dt)
            n_steps += 1
        return

    def vmf_step(self, dt: float) -> None:
        self.ode_step_counter = 0
        init_time = self.time
        eom_func = self._get_vmf_func()
        propagated = self._odeint(eom_func, self._vectorize(), dt)
        self._update_state_from_vector(propagated)
        self.time = init_time + dt
        if self.renormalization_coefficient is not None:
            self.renormalize_root()
        return

    def ps1_step(self, dt: float) -> None:
        self.ode_step_counter = 0
        init_time = self.time
        half_dt = dt / 2.0
        for eom in self._ti_eoms:
            eom.update_mean_fields(self.state)
        if self.is_time_dependent:
            # For TD terms only considering the mean fields during the time step
            self.update_td_terms(init_time + half_dt)
            for eom in self._td_eoms:
                eom.update_mean_fields(self.state)

        self._ps1_forward_step(half_dt)
        self._node_step(self.root, dt)
        self._ps1_backward_step(half_dt)
        self.time = init_time + dt
        if self.renormalization_coefficient is not None:
            self.renormalize_root()
        return

    def ps2_step(self, dt: float) -> None:
        self.ode_step_counter = 0
        init_time = self.time
        half_dt = dt / 2.0
        for eom in self._ti_eoms:
            eom.update_mean_fields(self.state)
        if self.is_time_dependent:
            # For TD terms only considering the mean fields during the time step
            self.update_td_terms(init_time + half_dt)
            for eom in self._td_eoms:
                eom.update_mean_fields(self.state)

        self._ps2_forward_step(half_dt)
        self._ps2_backward_step(half_dt)
        self.time = init_time + dt
        if self.renormalization_coefficient is not None:
            self.renormalize_root()
        return

    def _vectorize(self, tensors: None | list[OptArray] = None) -> OptArray:
        if tensors is None:
            tensors = [self.state[p] for p in self._node_visitor]
        return opt_cat([a.flatten() for a in tensors])

    def _update_state_from_vector(self, vec: OptArray) -> None:
        tensors = [
            a.reshape(s)
            for a, s in zip(opt_split(vec, self._size_list), self._shape_list)
        ]
        self.state.update(zip(self._node_visitor, tensors))
        return

    def _node_step(self, p: Node, dt: float) -> None:

        def _dp(t: float, array: OptArray):
            # ans = opt_zeroes_like(vector)
            self.state.update({p: array})
            ans = opt_zeros_like(array)
            for eom in self._ti_eoms + self._td_eoms:
                ans += eom.node_eom(p, array)
            return ans

        ans = self._odeint(_dp, self.state[p], dt)
        self.state.update({p: ans})
        return

    def _one_site_move(self, p: Node, i: int, q: Node, j: int,
                       dt: None | float) -> None:
        state = self.state
        eoms = self._ti_eoms + self._td_eoms

        # Split the edge_array and update p
        l_tensor, edge_array = _one_site_split(state[p], i)
        r_tensor = state[q]

        # Update all mean fields
        op_list = [None] * self.n_terms
        for _n, eom in enumerate(eoms):
            eom = eoms[_n]
            r_mf = eom.mean_fields.pop((p, i))
            l_mf = eom.get_node_mean_field(l_tensor, p, i)
            eom.mean_fields[q, j] = l_mf
            eom.node_axes[p] = i
            eom.node_axes[q] = None
            op_list[_n] = {
                _i: mf
                for _i, mf in ((0, l_mf), (1, r_mf)) if mf is not None
            }

        # Propagate the edge_array
        if dt is not None:

            def _diff(t: float, a: OptArray) -> OptArray:
                ans = opt_zeros_like(a)
                for op_dict in op_list:
                    if op_dict:
                        ans += opt_multitransform(op_dict, a)
                return ans

            edge_array = self._odeint(_diff, edge_array, dt)

        # Merge and update q
        new_root = _one_site_merge(r_tensor, j, edge_array)
        state.update({p: l_tensor, q: new_root})
        self.root = q
        return

    def _krylov_one_site_move(self, p: Node, i: int, q: Node, j: int,
                              dt: None | float) -> None:
        """Experimental method for adaptive PS1 step.
        Extend the _one_site_move method to include the adaptive rank on edge p-q.
        This is done by extending the column space of p and q according to the 
        First order krylov space {p, F_p} and {q, F_q} when the cannonical center is
        at the edge p-q.
        """
        raise NotImplementedError
        state = self.state
        eoms = self._ti_eoms + self._td_eoms

        # Split the edge_array and update l_tensor
        l_tensor, edge_array = _one_site_split(state[p], i)
        l_dim, r_dim = edge_array.shape
        l_fst_list = [eom.get_krylov(l_tensor, p, i) for eom in eoms]
        l_add = sum(l_fst_list)
        new_l_tensor = _stack_orth(l_tensor, _truncate(l_add, 1, i), i)
        new_l_dim = new_l_tensor.shape[i]
        # get r_tensor
        r_tensor = state[q]
        r_fst_list = [eom.get_krylov(r_tensor, q, j) for eom in eoms]
        r_add = sum(r_fst_list)
        new_r_tensor = _stack_orth(r_tensor, _truncate(r_add, 1, j), j)
        new_r_dim = new_r_tensor.shape[j]
        # print(f"{p}->{q}: l_dim={new_l_dim:}, r_dim={new_r_dim}, old_dim={dim}", end="\n", flush=True) # noqa
        # Update all mean fields
        op_list = [None] * self.n_terms
        for _n, eom in enumerate(eoms):
            eom.mean_fields.pop((p, i))
            r_mf = eom.get_node_mean_field(new_r_tensor, q, j)
            l_mf = eom.get_node_mean_field(new_l_tensor, p, i)
            eom.mean_fields[q, j] = l_mf
            eom.node_axes[p] = i
            eom.node_axes[q] = None
            op_list[_n] = {
                _i: mf
                for _i, mf in ((0, l_mf), (1, r_mf)) if mf is not None
            }

        # Propagate the edge_array
        # print(f'edge_array: {l_dim,r_dim}->{new_l_dim, new_r_dim}', end="\n", flush=True)# noqa
        new_edge_array = opt_zeros((new_l_dim, new_r_dim),
                                   dtype=edge_array.dtype,
                                   device=edge_array.device)
        new_edge_array[:l_dim, :r_dim] = edge_array
        # print(dt)
        if dt is not None:

            def _diff(t: float, a: OptArray) -> OptArray:
                ans = opt_zeros_like(a)
                for op_dict in op_list:
                    if op_dict:
                        ans += opt_multitransform(op_dict, a)
                return ans

            new_edge_array = self._odeint(_diff, new_edge_array, dt)

        # Merge and update q
        state.update({p: new_l_tensor})
        new_root = _one_site_merge(new_r_tensor, j, new_edge_array)
        state.update({q: new_root})
        self.root = q
        return

    def _two_site_move(self, p: Node, i: int, q: Node, j: int,
                       dt: float) -> None:
        target_rank = None  # type: None | int
        eoms = self._ti_eoms + self._td_eoms
        state = self.state
        ord_p = state.order(p)
        ord_q = state.order(q) 
        # the Target rank should be at least the order of the tensor for dynamics
        target_rank = max(ord_p, ord_q) 

        # Merge the two sites
        edge_array = _two_site_merge(state, p, i, q, j)

        # Propagate the edge_array
        op_list = [None] * self.n_terms  # type: list[None | dict[int, OptArray]]
        for _n, eom in enumerate(eoms):
            l_ops = [eom.mean_fields[p, _i] for _i in range(ord_p) if _i != i]
            r_ops = [eom.mean_fields[q, _j] for _j in range(ord_q) if _j != j]
            op_list[_n] = {
                _k: mf
                for _k, mf in enumerate(l_ops + r_ops) if mf is not None
            }

        def _diff(t: float, a: OptArray) -> OptArray:
            ans = opt_zeros_like(a)
            for op_dict in op_list:
                if op_dict:
                    ans += opt_multitransform(op_dict, a)
            return ans

        edge_array = self._odeint(_diff, edge_array, dt)

        # Split the edge_array and update p and q
        l_tensor, r_tensor = _two_site_split(state,
                                             p,
                                             i,
                                             q,
                                             j,
                                             edge_array,
                                             target_rank=target_rank,
                                             atol=self.ps2_atol,
                                             ratio=self.ps2_ratio)
        # Update all mean fields
        for eom in eoms:
            eom.mean_fields.pop((p, i))
            eom.mean_fields[q, j] = eom.get_node_mean_field(l_tensor, p, i)
            eom.node_axes[p] = i
            eom.node_axes[q] = None

        state.update({p: l_tensor, q: r_tensor})
        self.root = q
        return

    def _ps1_forward_step(self, dt: float) -> None:
        depths = self._depths
        links = self._node_link_visitor
        move = self._one_site_move
        node_step = self._node_step

        length = len(links)
        for n, (p, i, q, j) in enumerate(links):
            # print(f"({n+1}/{length}){p, i}--{q,j}:{self.state.dimension(q, j)}", end=" ") # noqa
            assert p is self.root
            if depths[p] < depths[q]:
                move(p, i, q, j, None)
            else:
                node_step(p, dt)
                move(p, i, q, j, -dt)
            # print(f"=> {self.state.dimension(p, i)}", end="\n", flush=True)
        return

    def _ps1_backward_step(self, dt: float) -> None:
        depths = self._depths
        links = self._node_link_visitor
        move = self._one_site_move
        node_step = self._node_step

        for q, j, p, i in reversed(links):
            # print(f"({n+1}/{length}){p, i}--{q,j}:{self.state.dimension(q, j)}", end=" ") # noqa
            assert p is self.root
            if depths[p] < depths[q]:
                move(p, i, q, j, -dt)
                node_step(q, dt)
            else:
                move(p, i, q, j, None)
            # print(f"=> {self.state.dimension(p, i)}", end="\n", flush=True)
        return

    def _ps2_forward_step(self, dt: float) -> None:
        depths = self._depths
        links = self._node_link_visitor
        move1 = self._one_site_move
        move2 = self._two_site_move
        node_step = self._node_step

        end = len(links) - 1
        for n, (p, i, q, j) in enumerate(links):
            # print(f"({n+1}/{length}){p, i}--{q,j}:{self.state.dimension(q, j)}", end=" ") # noqa
            assert p is self.root
            if depths[p] < depths[q]:
                move1(p, i, q, j, None)
            else:
                move2(p, i, q, j, dt)
                if n != end:
                    node_step(q, -dt)
            # print(f"=> {self.state.dimension(p, i)}", end="\n", flush=True)
        return

    def _ps2_backward_step(self, dt: float) -> None:
        depths = self._depths
        links = self._node_link_visitor
        move1 = self._one_site_move
        move2 = self._two_site_move
        node_step = self._node_step

        start = 0
        for n, (q, j, p, i) in enumerate(reversed(links)):
            # print(f"({n+1}/{length}){p, i}--{q,j}:{self.state.dimension(q, j)}", end=" ") # noqa
            assert p is self.root
            if depths[p] < depths[q]:
                if n != start:
                    node_step(p, -dt)
                move2(p, i, q, j, dt)
            else:
                move1(p, i, q, j, None)
            # print(f"=> {self.state.dimension(p, i)}", end="\n", flush=True)
        return

    def _get_vmf_func(self):
        vectorize = self._vectorize
        update_state = self._update_state_from_vector
        update_td_terms = self.update_td_terms
        state = self.state
        svd_info = self._svd_info
        include_td = self.is_time_dependent
        eoms = self._ti_eoms + self._td_eoms
        node_visitor = self._node_visitor

        def _diff(t: float, vector: OptArray):
            update_state(vector)
            # For time-dependent terms update time.
            if include_td:
                update_td_terms(t) 
            if svd_info is not None:
                svd_info.update(state)
            ans_list = [opt_zeros_like(state[p]) for p in node_visitor]
            for eom in eoms:
                eom.update_mean_fields(state)
                if isinstance(eom, _DirectProdEOM):
                    eom.update_adjointness(state)
                    for n, p in enumerate(node_visitor):
                        ans_list[n] += eom.node_eom(p, state[p]) 
                elif isinstance(eom, _ProdEOM):
                    eom.update_adjointness(state, svd_info)
                    for n, p in enumerate(node_visitor):
                        ans_list[n] += eom.node_eom(p, state[p],
                                                    svd_info.s[p],
                                                    svd_info.vh[p])  
                else:
                    raise RuntimeError(
                        f'Unknown EOM type: {eom.__class__.__name__}')
            ans = vectorize(ans_list)
            return ans

        return _diff

    def _get_vmf_func_direct(self):
        vectorize = self._vectorize
        update_state = self._update_state_from_vector
        update_td_terms = self.update_td_terms
        state = self.state 
        include_td = self.is_time_dependent
        eoms = self._ti_eoms + self._td_eoms
        node_visitor = self._node_visitor

        def _diff(t: float, vector: OptArray):
            update_state(vector)
            # For time-dependent terms update time.
            if include_td:
                update_td_terms(t)
            # all terms
            ans = opt_zeros_like(vector)
            for eom in eoms:
                eom.update_mean_fields(state) 
                eom.update_adjointness(state)
                ans += vectorize([eom.node_eom(p, state[p]) 
                                  for p in node_visitor])
            return ans

        return _diff

    def _get_cmf_func(self):
        vectorize = self._vectorize
        update_state = self._update_state_from_vector
        update_td_terms = self.update_td_terms
        state = self.state
        svd_info = self._svd_info
        include_td = self.is_time_dependent
        ti_eoms = self._ti_eoms
        td_eoms = self._td_eoms
        node_visitor = self._node_visitor

        def _diff(t: float, vector: OptArray):
            update_state(vector)
            # all terms
            ans = opt_zeros_like(vector)
            if svd_info is not None:
                svd_info.update(state)
            # For time-independent terms treat mf and adj to be time-independent during a step
            for eom in ti_eoms:
                eom.update_mean_fields(state)
                if isinstance(eom, _DirectProdEOM):
                    ans += vectorize(
                        [eom.node_eom(p, state[p]) for p in node_visitor])
                elif isinstance(eom, _ProdEOM):
                    ans += vectorize([
                        eom.node_eom(p, state[p], svd_info.s[p],
                                     svd_info.vh[p]) for p in node_visitor
                    ])
                else:
                    raise RuntimeError(
                        f'Unknown EOM type: {eom.__class__.__name__}')
            # For time-dependent terms update time.
            if include_td:
                update_td_terms(t)
            for eom in td_eoms:
                eom.update_mean_fields(state)
                if isinstance(eom, _DirectProdEOM):
                    eom.update_adjointness(state)
                    ans += vectorize(
                        [eom.node_eom(p, state[p]) for p in node_visitor])
                elif isinstance(eom, _ProdEOM):
                    eom.update_adjointness(state, svd_info)
                    ans += vectorize([
                        eom.node_eom(p, state[p], svd_info.s[p],
                                     svd_info.vh[p]) for p in node_visitor
                    ])
                else:
                    raise RuntimeError(
                        f'Unknown EOM type: {eom.__class__.__name__}')
            return ans

        return _diff

    def _odeint(self,
                func: Callable[[float, OptArray], OptArray],
                y0: OptArray,
                dt: float,
                tuple_complex: bool = True) -> OptArray:
        if self.ode_method == 'exp':
            ode_method = 'iter' + str(prod(y0.shape))
        else:
            ode_method = self.ode_method

        if tuple_complex:

            def _grad(t, y0):
                # print("evaluating", t)
                re, im = y0
                d = func(t, (re + 1.0j * im))
                self.ode_step_counter += 1
                return opt_stack((d.real, d.imag))

            y0 = opt_stack((y0.real, y0.imag))
        else:

            def _grad(t, y0):
                d = func(t, y0)
                self.ode_step_counter += 1
                return d

        ans = opt_odeint(_grad,
                         self.time,
                         y0,
                         dt,
                         atol=self.ode_atol,
                         rtol=self.ode_rtol,
                         method=ode_method)
        if tuple_complex:
            re, im = ans
            ans = (re + 1.0j * im)
        return ans



class DynamicalSparsePropagator(SparsePropagator):
    def __init__(self,
                 op: SparseSPO,
                 state: Model,
                 frame: Frame,
                 root: Node,
                 renormalize_root=False,
                 init_time=0.0) -> None:
        super().__init__(op, state, frame, root, renormalize_root, init_time)
        self._get_vmf_func = self._get_cmf_func
        return
    
        
    def _adaptive_two_site_move(self, p: Node, i: int, q: Node, j: int,
                                dt: float) -> None:
        """Experimental method for adaptive PS2 step.
        Extend the _two_site_move method to include the adaptive connectivity between the neighbors of p and q.
        """
        raise NotImplementedError 
