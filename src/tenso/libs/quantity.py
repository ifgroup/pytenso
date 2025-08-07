# coding: utf-8
r"""Unit transformations.
"""
from __future__ import annotations

from typing import Literal, Optional

from scipy.constants import physical_constants as constants

# 1 au = `atomic_unit_in['some-unit']` some-unit
atomic_unit_in = {
    None: 1,
    # Energy
    'J': constants["Hartree energy"][0],
    'eV': constants["Hartree energy in eV"][0],
    'meV': 1.e3 * constants["Hartree energy in eV"][0],
    '/cm': 0.01 * constants["hartree-inverse meter relationship"][0],
    'K': constants["hartree-kelvin relationship"][0],
    # Time
    's': constants["atomic unit of time"][0],
    'fs': 1.e15 * constants["atomic unit of time"][0],
    'ps': 1.e12 * constants["atomic unit of time"][0],
    '/K': constants["kelvin-hartree relationship"][0],
    # Length
    'A': 1.e10 * constants["atomic unit of length"][0],
    'm': constants["atomic unit of length"][0],
    'pm': 1.e12 * constants["atomic unit of length"][0],
    # Mass
    'kg': constants['atomic unit of mass'][0]
}

synonyms = {
    None: ['au', 'a.u.'],
    'K': ['kelvin', 'Kelvin'],
    'eV': ['ev'],
    'meV': ['mev'],
    '/K': ['K-1', r'K^(-1)', r'K^{-1}'],
    '/cm': ['cm-1', r'cm^(-1)', r'cm^{-1}']
}


class Quantity(object):
    def __init__(self, value: float, unit: Optional[str] = None) -> None:
        """
        Parameters:
        """
        self.value = float(value)
        self.unit = self.standardize(unit)
        return

    @staticmethod
    def standardize(unit: Optional[str]) -> Optional[str]:
        if unit is not None and unit not in atomic_unit_in:
            found = False
            for key, l in synonyms.items():
                if unit in l:
                    unit = key
                    found = True
                    break
            if not found:
                raise ValueError("Cannot recognize unit {} as any in {}."
                                 .format(unit, list(atomic_unit_in.keys())))
        return unit

    @property
    def au(self) -> float:
        return self.value / atomic_unit_in[self.unit]

    def convert_to(self, unit: Optional[str] = None) -> Quantity:
        unit = self.standardize(unit)
        self.value = self.au * atomic_unit_in[unit]
        self.unit = unit
        return self

    def to(self, unit: Optional[str] = None) -> Quantity:
        return self.convert_to(unit)

    # A simplified and incomplete implementation for +, -, *, /
    # + - only allowed between Quantities
    # * / only allowed between Quantities float

    def __neg__(self) -> Quantity:
        cls = type(self)
        return cls(-self.value, self.unit)

    def __add__(self, other: Quantity) -> Quantity:
        cls = type(self)
        return cls(self.au + other.au)

    def __sub__(self, other: Quantity) -> Quantity:
        cls = type(self)
        return cls(self.au - other.au)

    def __mul__(self, other: float) -> Quantity:
        cls = type(self)
        return cls(self.value * other, unit=self.unit)

    def __truediv__(self, other: float) -> Quantity:
        cls = type(self)
        return cls(self.value / other, unit=self.unit)

    def __eq__(self, other: Quantity | Literal[0]) -> Quantity:
        if hasattr(other, "au"):
            return self.au == other.au
        elif other == 0:
            return self.value == 0
        else:
            raise TypeError(f"Quantity can only compare with Quantity or 0, not {type(other)}.")

    def __gt__(self, other: Quantity | Literal[0]) -> Quantity:
        if hasattr(other, "au"):
            return self.au > other.au
        elif other == 0:
            return self.value > 0
        else:
            raise TypeError(f"Quantity can only compare with Quantity or 0, not {type(other)}.")

    def __str__(self) -> str:
        unit = "a.u." if self.unit is None else self.unit
        return f"{self.value:.8f}_{unit}"

    def __repr__(self) -> str:
        return f"<{str(self)}>"
