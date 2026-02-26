# coding: utf-8
r"""Data structure for topology of tensors in a network

"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Optional
from weakref import WeakValueDictionary

from tenso.libs.utils import iter_round_visitor, iter_visitor, depths


class Point:
    """An abstract class representing a vertex in the graph of the tensor network which 
    is extended by Node and End.

    :param __cache: A weak valued dictionary with keyts of tuple[str, str] and values of Point
    :type __cache: WeakValueDictionary
    """
    __cache = WeakValueDictionary()  # type: WeakValueDictionary[tuple[str, str], Point] # noqa

    def __new__(cls, name: Optional[str] = None):
        """Allocates memory prior to call of the constructor.
        """
        if name is None:
            obj = object.__new__(cls)
        else:
            cache_name = (cls.__name__, name)
            obj = cls.__cache.get(cache_name)
            if obj is None:
                obj = object.__new__(cls)
                cls.__cache[cache_name] = obj
        return obj

    def __init__(self, name: Optional[str] = None) -> None:
        """Constructor which provides an identifying string

        :param name: Identifying string
        :type name: string, optional
        """
        self.name = str(hex(id(self))) if name is None else str(name)
        return


class Node(Point):
    """Class representing a vertex in the graph of the tensor network which has more than
    one neighbor. This extends :class: Point.
    """
    def __repr__(self) -> str:
        """Returns a string giving the Node's unique name.
        """
        return f'({self.name})'


class End(Point):
    """Class representing a vertex in the graph of the tensor network which has only
    one neighbor. This extends :class: Point
    """
    def __repr__(self) -> str:
        """Return a string giving the End's unique name.
        """
        return f'<{self.name}>'


class Frame:
    """Class which holds the topology of the tensor network graph, the :class: Nodes and :class: Ends
    and the edges that link them together.
    """
    def __init__(self):
        self._neighbor = dict()  # type: dict[Point, list[Point]]

        # Convenient properties that are easy to compute when generating
        # but can be calucated from _neighbor
        self._duality = dict()  # type: dict[tuple[Point, None | int], tuple[Point, None | int]] # noqa
        self._axes = dict()  # type: dict[tuple[Point, Point], tuple[None | int, None | int]] # noqa
        return

    def __str__(self) -> str:
        string = ('{' +
                  ", ".join(f"{type(k).__name__}('{k.name}'): [" +
                            ", ".join(
                                f"{type(p).__name__}('{p.name}')" for p in v)
                            + ']'
                            for k, v in self.get_graph().items())
                  + '}')
        return string

    def __contains__(self, p: Point) -> bool:
        return p in self._neighbor

    def copy(self):
        """Return a full copy of the Frame.
        """
        new_frame = Frame()
        new_frame._neighbor = self._neighbor.copy()
        new_frame._duality = self._duality.copy()
        new_frame._axes = self._axes.copy()
        return new_frame

    def add_link(self, p: Point, q: Point) -> None:
        """Add a link between two :class: Points. :class: End can only have one link. :class: Node can have multiple links.

        :param p: First :class: Point to be connected.
        :type p: :class: Point

        :param 1: Second :class: Point to be connected.
        :type q: :class: Point
        """

        is_p_node = isinstance(p, Node)
        is_q_node = isinstance(q, Node)
        if p not in self._neighbor:
            self._neighbor[p] = list()
        else:
            assert is_p_node
        if q not in self._neighbor:
            self._neighbor[q] = list()
        else:
            assert is_q_node

        i = len(self._neighbor[p]) if is_p_node else None
        j = len(self._neighbor[q]) if is_q_node else None
        self._axes[(p, q)] = (i, j)
        self._axes[(q, p)] = (j, i)
        self._duality[(p, i)] = (q, j)
        self._duality[(q, j)] = (p, i)
        self._neighbor[p].append(q)
        self._neighbor[q].append(p)
        return

    @property
    def points(self) -> set[Point]:
        return set(self._neighbor.keys())

    @property
    def nodes(self) -> set[Node]:
        return {p for p in self._neighbor.keys() if isinstance(p, Node)}

    @property
    def ends(self) -> set[End]:
        return {p for p in self._neighbor.keys() if isinstance(p, End)}

    def degree(self, p: Node):
        return len(self._neighbor[p])

    def dual(self, p: Point, i: None | int) -> tuple[Point, None | int]:
        return self._duality[p, i]

    def axes(self, p: Point, q: Point) -> tuple[int, None | int]:
        return self._axes[p, q]

    def near_points(self, key: Point) -> list[Point]:
        return list(self._neighbor[key])

    def near_nodes(self, key: Node) -> list[Node]:
        return [n for n in self._neighbor[key] if isinstance(n, Node)]

    def node_link_visitor(self,
                          start: Node) -> list[tuple[Node, int, Node, int]]:
        paired = list(
            pairwise(n for n in iter_round_visitor(start, self.near_nodes))
        )
        axes_list = [self._axes[n1, n2] for n1, n2 in paired]
        return [
            (p, i, q, j) for (p, q), (i, j) in zip(paired, axes_list)
        ]

    def point_link_visitor(self,
                           start: Point) -> list[tuple[Point, int, Point, int]]:
        paired = list(
            pairwise(n for n in iter_round_visitor(start, self.near_points))
        )
        axes_list = [self._axes[n1, n2] for n1, n2 in paired]
        return [
            (p, i, q, j) for (p, q), (i, j) in zip(paired, axes_list)
        ]

    def node_visitor(self,
                     start: Node,
                     method: Literal['DFS', 'BFS'] = 'DFS') -> list[Node]:
        return list(iter_visitor(start, self.near_nodes, method=method))

    def point_visitor(self,
                      start: Node,
                      method: Literal['DFS', 'BFS'] = 'DFS') -> list[Node]:
        return list(iter_visitor(start, self.near_points, method=method))

    def get_node_depths(self, start: Node) -> dict[Node, int]:
        return depths(start, self.near_nodes)

    def get_node_axes(self, start: Point) -> dict[Node, Optional[int]]:
        ans = {start: None}  # type: dict[Point, Optional[int]]
        for p, _, q, j in self.point_link_visitor(start):
            if p in ans and q not in ans:
                ans[q] = j
        return {k: v for k, v in ans.items() if isinstance(k, Node)}

    def get_graph(self) -> dict[Node, list[Point]]:
        return {k: v for k, v in self._neighbor.items() if isinstance(k, Node)}

    def construct_from_graph(self, graph: dict[Node, list[Point]]) -> None:
        assert not self._neighbor and not self._duality and not self._axes
        added = set()
        for n, children in graph.items():
            for child in children:
                if child not in added:
                    self.add_link(n, child)
            added.add(n)
        return
