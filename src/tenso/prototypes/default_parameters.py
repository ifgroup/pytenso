# coding: utf-8
"""Default parameters for the prototypes in the package.
"""

from typing import Literal
from tenso.libs.quantity import Quantity as __

__all__ = [
    "default_units",
    "default_kwargs",
    "default_extension",
    "get_default_kwargs",
]

default_extension = {
    'input': '.json',
    'output': '.dat.log',
    'debug': '.debug.log',
    'checkpoint': '.pt',
}

default_units = {
    "inverse_temperature": "/K",
    "time": "fs",
    "energy": "/cm",
    "unital_energy": 1000.0,
}

default_kwargs = {
    # For bath correlation function
    "bcf.decomposition_method": "Pade",
    "bcf.n_ltc": 3,
    "bcf.include_lindblad": False,
    "bcf.use_ht_function": False,
    "bcf.use_cross": False,
    "bcf.temperature": 300,
    # For HEOM
    "heom.dim": 5,
    "heom.use_dvr": False,
    "heom.dvr_type": "sine",
    "heom.dvr_length": 32,
    "heom.metric": "re",
    # For QLE
    "qle.dim": 5,
    "qle.use_dvr": False,
    "qle.dvr_type": "sine",
    "qle.dvr_length": 32,
    "qle.hermite": False,
    # For FT-MCTDH
    "mctdh.dim": 5,
    # For static tensor network
    "tn.frame_method": "tree2",
    "tn.rank": 3,
    # For state propagation
    "propagation.ode_method": "dopri5",
    "propagation.ode_rtol": 1e-5,
    "propagation.ode_atol": 1e-7,
    "propagation.vmf_atol": 1e-7,
    "propagation.ps2_atol": 1e-7,
    "propagation.ps2_ratio": 2.0,
    "propagation.stepwise_method": "mix",
    "propagation.start_time": 0.0,
    "propagation.end_time": 1000.0,
    "propagation.ps_method": "vmf",
    "propagation.step_time": 1.0,
    "propagation.auxiliary_ps_method": "ps2",
    "propagation.auxiliary_step_time": None,
    "propagation.max_auxiliary_steps": None,
    "propagation.max_auxiliary_rank": 32,
    # For debugging and testing
    "propagation.renormalize": False,
    "propagation.vmf_reg_method": 'extend',
    "propagation.vmf_reg_type": 'ip',
    "propagation.cache_svd_info": True,
    "tn.visualize_frame": False,
    "tn.load_checkpoint_from_file": False,
    "tn.save_checkpoint_to_file": False,
}


def quantity(
        value: float, unit_type: Literal['time', 'energy',
                                         'inverse_temperature']) -> float:
    """Convert the value to the internal unit.
    """
    ue = __(default_units["unital_energy"], default_units['energy']).au
    unit = default_units[unit_type]
    if unit_type in ['inverse_temperature', 'time']:
        return __(value, unit).au * ue
    elif unit_type in ['energy']:
        return __(value, unit).au / ue


def value(
        quantity: float, unit_type: Literal['time', 'energy',
                                            'inverse_temperature']) -> float:
    """Convert the quantity to the external unit.
    """
    ue = __(default_units["unital_energy"], default_units['energy']).au
    unit = default_units[unit_type]
    if unit_type in ['inverse_temperature', 'time']:
        return __(quantity / ue).convert_to(unit).value
    elif unit_type in ['energy']:
        return __(quantity * ue).convert_to(unit).value


def get_default_kwargs(domains: list[str]) -> dict:
    """Load the default parameters for the prototypes.
    """
    ans = dict()
    for key, value in default_kwargs.items():
        _d, _k = key.split('.', maxsplit=1)
        if _d in domains:
            if _k not in ans:
                ans[_k] = value
            else:
                raise ValueError(f"Duplicate key {_k} in {domains}")
    # Sort all kw in Alphabet order
    ans = {k: ans[k] for k in sorted(ans.keys())}
    return ans


def get_keys(domains: list[str]) -> set:
    """Load the registered keys in domains.
    """
    ans = set()
    for key in default_kwargs.keys():
        _d, _k = key.split('.', maxsplit=1)
        if _d in domains:
            if _k not in ans:
                ans.add(_k)
            else:
                raise ValueError(f"Duplicate key {_k} in {domains}")
    return ans
