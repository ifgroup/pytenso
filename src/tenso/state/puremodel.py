from __future__ import annotations

from math import prod
from typing import Iterable
import numpy as np

from tenso.libs.backend import OptArray, opt_array, opt_save, opt_load
from tenso.state.pureframe import Frame, Node


def triangular(n_list):
    """A Generator yields the natural number in a triangular order.
        """
    length = len(n_list)
    prod_list = [1]
    for n in n_list:
        prod_list.append(prod_list[-1] * n)
    prod_list = prod_list

    def key(case):
        return sum(n * i for n, i in zip(prod_list, case))

    combinations = {0: [[0] * length]}
    for m in range(prod_list[-1]):
        if m not in combinations:
            permutation = [
                case[:j] + [case[j] + 1] + case[j + 1:]
                for case in combinations[m - 1]
                for j in range(length)
                if case[j] + 1 < n_list[j]
            ]
            combinations[m] = []
            for case in permutation:
                if case not in combinations[m]:
                    combinations[m].append(case)
        for case in combinations[m]:
            yield key(case)


class Model:
    """A Model is a Frame with valuation for each node.
    """

    def __init__(self,
                 valuation: dict[Node, OptArray]
                 | Iterable[tuple[Node, OptArray]]) -> None:
        """
        Args:
            frame: Topology of the tensor network;
        """
        self._valuation = dict(valuation)  # type: dict[Node, OptArray]
        return

    def __contains__(self, p: Node) -> bool:
        return p in self._valuation

    def __getitem__(self, p: Node) -> OptArray:
        return self._valuation[p]

    def save(self, filename: str) -> None:
        """Save the model to a file."""
        named_dct = {p.name: a for p, a in self._valuation.items()}
        opt_save(named_dct, filename)
        return

    @classmethod
    def load(cls, filename: str) -> Model:
        """Load the model from a file."""
        named_dct = opt_load(filename)
        valuation = {Node(name=n): a for n, a in named_dct.items()}
        return cls(valuation)

    @property
    def nodes(self) -> set[Node]:
        return set(self._valuation.keys())

    def shape(self, p: Node) -> list[int]:
        return list(self._valuation[p].shape)

    def order(self, p: Node) -> int:
        return self._valuation[p].ndim

    def dimension(self, p: Node, i: int) -> int:
        return self._valuation[p].shape[i]

    def copy(self) -> Model:
        """A shallow copy of the model."""
        return Model(self._valuation)

    def conjugate(self) -> Model:
        """Conjugate the model."""
        new_valuation = {p: a.conj() for p, a in self._valuation.items()}
        new_model = Model(new_valuation)
        return new_model

    def substitute(self,
                   valuation: dict[Node, OptArray]
                   | Iterable[tuple[Node, OptArray]]) -> Model:
        new_model = self.copy()
        new_model.update(valuation)
        return new_model

    def update(self,
               valuation: dict[Node, OptArray]
               | Iterable[tuple[Node, OptArray]]) -> None:
        """
        Update the valuation of the model.
        """
        # v, s = zip(*(((n, a), (n, list(a.shape)))
        #            for n, a in valuation.items()))
        # self._valuation.update(v)
        # self._shapes.update(s)
        self._valuation.update(valuation)
        return
    
    def zero_like(self) -> Model:
        shapes = {k: v.shape for k, v in self._valuation.items()}
        return zeros_model(shapes)


def zeros_model(shapes: dict[Node, list[int]]) -> Model:
    """
    A model with proper shape arrays.
    Specify the dimension for each Edge in dims (default is 1).
    """
    valuation = {p: opt_array(np.zeros(shape)) for p, shape in shapes.items()}
    return Model(valuation)


def eye_model(frame: Frame, root: Node, shapes: dict[Node, list[int]]) -> Model:
    assert root in frame
    axes = frame.get_node_axes(root)

    valuation = dict()  # type: dict[Node, OptArray]
    for p in frame.nodes:
        shape = shapes[p]
        ax = axes[p]
        if ax is None:
            ans = np.zeros([prod(shape)])
            ans[0] = 1.0
            ans = ans.reshape(shape)
        else:
            l_dim = shape[ax]
            r_shape = shape[:ax] + shape[ax + 1:]
            ans = np.zeros([l_dim, prod(r_shape)])
            for v_i, _j in zip(ans, triangular(r_shape)):
                v_i[_j] = 1.0
            ans = ans.reshape([l_dim] + r_shape)
            ans = np.moveaxis(ans, 0, ax)
        valuation[p] = opt_array(ans)
    return Model(valuation)
