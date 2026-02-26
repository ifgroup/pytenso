# coding: utf-8
r"""Objects and functions handling transformation between default, external units
and internal TENSO units
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
        """Class dedicated to unit conversions between the default input
        units and the internal program units.

        :param value: The value under consideration
        :type value: float
        :param unit: Whether energy, time, or inverse temperature is 
            under consideration, designated as 'time', 'energy', or
            'inverse_temperature' respectively
        :type unit: string, optional
        """
        self.value = float(value)
        self.unit = self.standardize(unit)
        return

    @staticmethod
    def standardize(unit: Optional[str]) -> Optional[str]:
        """Search the list of synonyms for units that may be supplied and
        choose the standard representation for that unit.
        :param unit: Potentially non-standard string representing a unit
        :type unit: string, optional
        :returns: The standardized string representing the unit
        :rtype: string
        """
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
        """Return the value in atomic units.
        """
        return self.value / atomic_unit_in[self.unit]

    def convert_to(self, unit: Optional[str] = None) -> Quantity:
        """Change the value in the given unit to atomic units.
        :param unit: The current unit in use
        :type unit: string, optional
        """
        unit = self.standardize(unit)
        self.value = self.au * atomic_unit_in[unit]
        self.unit = unit
        return self

    def to(self, unit: Optional[str] = None) -> Quantity:
        """Change the value in the given unit to atomic units.
        :param unit: The current unit in use
        :type unit: string, optional
        """
        return self.convert_to(unit)

    # A simplified and incomplete implementation for +, -, *, /
    # + - only allowed between Quantities
    # * / only allowed between Quantities float

    def __neg__(self) -> Quantity:
        """Negate in experimental arithmetic implementation."""
        cls = type(self)
        return cls(-self.value, self.unit)

    def __add__(self, other: Quantity) -> Quantity:
        """Add two quantities in experimental arithmetic implementation.
        :param other: Quantity to add to self
        :type other: :class: Quantity
        :returns: The sum of the two quantities
        "rtype: :class: Quantity
        """
        cls = type(self)
        return cls(self.au + other.au)

    def __sub__(self, other: Quantity) -> Quantity:
        """Subtract a quantity in experimental arithmetic implementation.
        :param other: Quantity to subtract from self
        :type other: :class: Quantity
        :returns: The result of subtracting other from self
        "rtype: :class: Quantity
        """
        cls = type(self)
        return cls(self.au - other.au)

    def __mul__(self, other: float) -> Quantity:
        """Multiply two quantities in experimental arithmetic implementation.
        :param other: Quantity to multiply with the given quanitty
        :type other: :class: Quantity
        :returns: The product of the two quantities
        "rtype: :class: Quantity
        """
        cls = type(self)
        return cls(self.value * other, unit=self.unit)

    def __truediv__(self, other: float) -> Quantity:
        """Divide by the quantity other in experimental arithmetic implementation.
        :param other: Quantity to divide the self by
        :type other: :class: Quantity
        :returns: The quotient of self over other
        "rtype: :class: Quantity
        """
        cls = type(self)
        return cls(self.value / other, unit=self.unit)

    def __eq__(self, other: Quantity | Literal[0]) -> boolean:
        """Determines if two quantities are equal in experimental arithmetic implementation.
        :param other: Quantity to compare to the given quantity
        :type other: :class: Quantity
        :returns: True if the two quantites are equal, otherwise false
        "rtype: boolean
        """
        if hasattr(other, "au"):
            return self.au == other.au
        elif other == 0:
            return self.value == 0
        else:
            raise TypeError(f"Quantity can only compare with Quantity or 0, not {type(other)}.")

    def __gt__(self, other: Quantity | Literal[0]) -> boolean:
        """Compare two quantities in experimental arithmetic implementation.
        :param other: Quantity to compare to the given quantity
        :type other: :class: Quantity
        :returns: Whether the Quantity it is called on is larger than the argument.
        :rtype: boolean
        """
        if hasattr(other, "au"):
            return self.au > other.au
        elif other == 0:
            return self.value > 0
        else:
            raise TypeError(f"Quantity can only compare with Quantity or 0, not {type(other)}.")

    def __str__(self) -> str:
        """Return a string representation of the quantity consisting of the value and unit.
        :returns: Descriptive string
        :rtype: string
        """
        unit = "a.u." if self.unit is None else self.unit
        return f"{self.value:.8f}_{unit}"

    def __repr__(self) -> str:
        """Return a descriptive string
        :returns: Descriptive string
        :rtype: string
        """
        return f"<{str(self)}>"
