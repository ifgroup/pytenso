# coding: utf-8
"""Metas."""
from __future__ import annotations

from builtins import map, zip
from collections import OrderedDict
from itertools import tee
from operator import itemgetter
from typing import (Any, Callable, Generator, Iterable, Literal, Optional,
                    TypeVar)

T = TypeVar('T')


def lazyproperty(func: Callable[..., T]) -> Callable[..., T]:
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

    def wrapped(*args: ..., **kwargs: ...) -> T:
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def iter_round_visitor(
        start: T,
        r: Callable[[T], list[T]]) -> Generator[tuple[T, bool], None, None]:
    """Iterative round-trip visitor. Only support 'DFS' (depth first) method.

    Args:
        start: Initial object
        r: Relation function.
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
    """Iterative visitor.

    Args:
        start: Initial object
        r: Relation function.
        method: in {'DFS', 'BFS'}. 'DFS': Depth first; 'BFS': Breadth first.
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
    """Iteratively geerate the depth of each component.

    Args:
        start: Initial object
        r: Relation function.
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
    """Iteratively generate the depth of each component.

    Args:
        start: Initial object
        r: Relation function.
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
    """Generate a Tree for the soureces as leaves using Huffman coding method.
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
