# coding: utf-8
"""This file includes miscellaneous utility functions without another obvious home."""
from __future__ import annotations

from builtins import map, zip
from collections import OrderedDict
from itertools import tee
from operator import itemgetter
from typing import (Any, Callable, Generator, Iterable, Literal, Optional,
                    TypeVar)

T = TypeVar('T')


def lazyproperty(func: Callable[..., T]) -> Callable[..., T]:
    """Experimental function to implement just in time evaluation
    of a property (lazy evaluation)"""
    name = '__lazy_' + func.__name__

    @property
    def lazy(self) -> T:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


def count_calls(f: Callable[..., T]) -> Callable[..., T]:
    """Debugging function used to measure call counts"""
    def wrapped(*args: ..., **kwargs: ...) -> T:
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def iter_round_visitor(
        start: T,
        r: Callable[[T], list[T]]) -> Generator[tuple[T, bool], None, None]:
    """Generator used as a round-trip tree visitor with the 'DFS' (depth first search) method

    :param start: The object to begin search from
    :type start: :class:`Node`

    :param r: Callable relation function taking start and returning the neighbor list of start
    :type r: Callable

    :returns: A tuple containing the next node in the tree
    :rtype: :class:`Node`
    """
    stack, visited = [start], set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            nexts = [n for n in r(vertex) if n not in visited]
            stack.extend(nexts[i // 2] if i % 2 else vertex
                         for i in range(2 * len(nexts)))
        yield vertex


def iter_visitor(
        start: T,
        r: Callable[[T], list[T]],
        method: Literal['DFS', 'BFS'] = 'DFS') -> Generator[T, None, None]:
    """Generator used as an iterative visitor supporting depth first or breadth first 

    :param start: The object to begin search from
    :type start: :class:`Node`

    :param r: Callable relation function taking start and returning the neighbor list of start
    :type r: Callable

    :param method: Either 'DFS', depth first, or 'BFS', breadth first method, will be used
    :type method: string

    :returns: A tuple containing the next node in the tree
    :rtype: :class:`Node`
    """
    stack, visited = [start], set()
    while stack:
        if method == 'DFS':
            stack, vertex = stack[:-1], stack[-1]
        else:
            assert method == 'BFS'
            vertex, stack = stack[0], stack[1:]
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(n for n in r(vertex) if n not in visited)
            yield vertex


def depths(start: T, r: Callable[[T], list[T]]) -> dict[T, int]:
    """Iteratively generate the depth of each component in the tree

    :param start: The object to begin search from
    :type start: :class:`Node`

    :param r: Callable relation function taking start and returning the neighbor list of start
    :type r: Callable

    :returns: Dictionary where keys are :class:`Node`s in the tree and values are their depths
    :rtype: dictionary
    """
    ans = {start: 0}
    stack, visited = [start], set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            n_ = {n: ans[vertex] + 1 for n in r(vertex) if n not in visited}
            stack.extend(n_.keys())
            ans.update(n_)
    return ans


def path(start: T, stop: T, r: Callable[[T], list[T]]) -> None | list[T]:
    """Determine the path through the tree between the :class:`Node`s start and stop

    :param start: The object to begin search from
    :type start: :class:`Node`

    :param start: The object to search for
    :type start: :class:`Node`

    :param r: Callable relation function taking start and returning the neighbor list of start
    :type r: Callable

    :returns: Nothing if start and stop are the same, otherwise a list containing the path between start and stop
    :rtype: list or None
    """
    stack, visited = [[start]], set()
    while stack:
        path = stack.pop()
        vertex = path[-1]
        if vertex is stop:
            return path

        if vertex not in visited:
            visited.add(vertex)
            stack.extend(path + [n] for n in r(vertex) if n not in visited)

    return None


def huffman_tree(sources: list[T],
                 new_obj: Callable[[], T],
                 importances: Optional[list[int]] = None,
                 n_ary: int = 2) -> tuple[OrderedDict[T, list[T]], T]:
    """Generate a Tree for the sources as leaves using Huffman coding method.

    :param sources: The leaves, :class:`End`s, for the tree
    :type sources: list of :class:`End`s

    :param importances: Importance ranking of the leaves to guide generation
    :type importances: list[int], Optional

    :param n_ary: Order of the tree (how many links an interior :class:`Node` should have)
    :type n_ary: int

    :returns: The tree as a tuple containing 1. an ordered dictionary representing the graph where each key is a :class:`Node` and the values are a list of :class:`Node`s representing its neighbors 2. the root of the tree
    :rtype: tuple[OrderedDict, :class:`Node`]
    """
    if importances is None:
        importances = [1] * len(sources)

    def fst(x):
        return x[0]

    def snd(x):
        return x[1]

    sequence = list(zip(sources, importances))
    graph = OrderedDict()
    while len(sequence) > 1:
        sequence.sort(key=snd)
        try:
            branch, sequence = sequence[:n_ary], sequence[n_ary:]
        except IndexError:
            branch, sequence = sequence, []
        p = sum(map(snd, branch))
        new = new_obj()
        graph[new] = list(map(fst, branch))
        sequence.insert(0, (new, p))
    return OrderedDict(reversed(graph.items())), fst(sequence[0])


def unzip(iterable: Iterable) -> Iterable[Iterable]:
    """The same as zip(*iter) but returns iterators, instead
    of expand the iterator. Mostly used for large sequence.
    Reference: https://gist.github.com/andrix/1063340
    """
    _tmp, iterable = tee(iterable, 2)
    iters = tee(iterable, len(next(_tmp)))
    return (map(itemgetter(i), it) for i, it in enumerate(iters))
