# coding: utf-8

from math import sqrt
import numpy as np

from tenso.bath.correlation import BoseEinstein, Correlation
from tenso.bath.star import StarBosons, Fourier
from tenso.bath.sd import Drude, OhmicExp, SpectralDensity, UnderdampedBrownian
from tenso.bath.tedopa import Tedopa
from tenso.libs.quantity import Quantity as __
from tenso.prototypes.default_parameters import default_units, quantity


def gen_comb_bcf(
    # Drude bath
    include_drude: bool,
    re_d: float,
    width_d: float,
    # Brownian bath
    include_brownian: bool,
    dof_b: int,
    freq_max_b: float,
    re_b: float,
    width_b: float,
    # Vibrational bath
    include_discrete: bool,
    dof_v: int,
    freq_max_v: float,
    re_v: float,
    # LTC bath
    temperature: float,
    decomposition_method: str,
    n_ltc: int,
    **kwargs,
) -> Correlation:
    """
    Generate a correlation function for a comb-like spectral density.
    
    Parameters
    ---------- 
    include_drude : bool
        Whether to include the Drude term for the low-frequency modes.
    re_d : float
        Reorganization energy of the Drude term.
    width_d : float
        Cutoff frequency of the Drude term.
    include_brownian : bool
        Whether to include the Brownian term for the high-frequency modes.
    dof_b : int
        Number of Brownian modes.
    freq_max_b : float  
        Maximum frequency space that the Brownian modes span.
    re_b : float
        Reorganization energy of the Brownian modes.
    width_b : float
        Cutoff frequency of the Brownian modes.
    include_discrete : bool
        Whether to include the discrete vibrational modes.
    dof_v : int
        Number of discrete vibrational modes.
    freq_max_v : float
        Maximum frequency space that the discrete vibrational modes span.
    re_v : float
        Reorganization energy of the discrete vibrational modes.
    temperature : float
        Temperature of the bath.
    decomposition_method : str
        Method to decompose the Bose-Einstein distribution.
    n_ltc : int
        Number of low-temperature correction terms in the decomposition of the Bose-Einstein distribution. 
    **kwargs : dict
        Ignored arguments.

    Returns
    -------
    Correlation
        The correlation function for the comb-like spectral density.
    """

    # Bath settings:
    corr = Correlation()
    beta = quantity(1 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distr = BoseEinstein(n=n_ltc, beta=beta)

    sds = []  # type:list[SpectralDensity]
    if include_drude:
        drude = Drude(quantity(re_d, 'energy'), quantity(width_d, 'energy'))
        sds.append(drude)

    if include_brownian:
        freq_space = [freq_max_b / (dof_b + 1) * (n + 1) for n in range(dof_b)]
        re_space = [re_b / dof_b / (n + 1) for n in range(dof_b)]
        for re, freq in zip(re_space, freq_space):
            b = UnderdampedBrownian(quantity(re, 'energy'),
                                    quantity(freq, 'energy'),
                                    quantity(width_b, 'energy'))
            sds.append(b)
    corr.add_spectral_densities(sds, distr)

    if include_discrete:
        freq_space = [freq_max_v / (dof_v + 1) * (n + 1) for n in range(dof_v)]
        g = sqrt(re_v * freq_max_v) / dof_v
        for _n, freq in enumerate(freq_space):
            corr.add_discrete_vibration(quantity(freq, 'energy'),
                                        quantity(g, 'energy'), beta)

    return corr


def gen_bcf(
    # Drude bath
    include_drude: bool = True,
    re_d: list[float] | None = None,
    width_d: list[float] | None = None,
    # Brownian bath
    include_brownian: bool = True,
    freq_b: list[float] | None = None,
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    # Vibrational bath
    include_discrete: bool = True,
    freq_v: list[float] | None = None,
    re_v: list[float] | None = None,
    # LTC bath
    temperature: float = 300.0,
    decomposition_method: str = 'Pade',
    n_ltc: int = 0,
    include_lindblad: bool = False,
    # Debugging
    use_cross: bool = False,
    use_ht_function: bool = False,
    **kwargs,
) -> Correlation:
    """
    Generate a correlation function for a composite spectral density for HEOM.

    Parameters
    ----------
    include_drude : bool
        Whether to include the Drude term for the low-frequency modes.
    re_d : list[float]
        Reorganization energies of every Drude modes.
    width_d : list[float]
        Cutoff frequencies of every Drude modes.
    include_brownian : bool
        Whether to include the Brownian term for the high-frequency modes.
    freq_b : list[float]
        Frequencies of every Brownian modes.
    re_b : list[float]
        Reorganization energies of every Brownian modes.
    width_b : list[float]
        Broadening of every Brownian modes.
    include_discrete : bool
        Whether to include the discrete vibrational modes.
    freq_v : list[float]
        Frequencies of every discrete vibrational modes.
    re_v : list[float]
        Reorganization energies of every discrete vibrational modes.
    temperature : float
        Temperature of the bath.
    decomposition_method : str
        Method to decompose the Bose-Einstein distribution.
    n_ltc : int
        Number of low-temperature correction terms in the decomposition of the Bose-Einstein distribution.
    include_lindblad : bool
        Whether to include the Lindblad rate as in Tanimura's HEOM.
    use_cross : bool
        Whether to use the cross-correlation that includes the trigonometric functions.
    use_ht_function : bool
        Whether to use the high-temperature approximation [2 / (beta omega)] of the Bose-Einstein distribution
        instead of the hyperbolic tangent function.
    
    Returns
    -------
    Correlation
        The correlation function for the composite spectral density.
    """
    # Bath settings:
    corr = Correlation()
    if temperature is not None:
        beta = quantity(1 / temperature, 'inverse_temperature')
    else:
        beta = None

    # Add continuous spectral densities
    BoseEinstein.decomposition_method = decomposition_method
    distr = BoseEinstein(n=n_ltc, beta=beta)
    sds = []  # type:list[SpectralDensity]
    if include_drude:
        if re_d is None:
            re_d = []
        if width_d is None:
            width_d = []
        for l, g in zip(re_d, width_d):
            drude = Drude(quantity(l, 'energy'), quantity(g, 'energy'))
            sds.append(drude)
    if include_brownian:
        if freq_b is None:
            freq_b = []
        if re_b is None:
            re_b = []
        if width_b is None:
            width_b = []
        for w, l, g in zip(freq_b, re_b, width_b):
            b = UnderdampedBrownian(quantity(l,
                                             'energy'), quantity(w, 'energy'),
                                    quantity(g, 'energy'))
            sds.append(b)
    if use_cross:
        corr.add_trigonometric(sds, distr)
    else:
        corr.add_spectral_densities(sds,
                                    distr,
                                    use_ht_function=use_ht_function)

    # Add discrete vibrational modes
    if include_discrete:
        if freq_v is None:
            freq_v = []
        if re_v is None:
            re_v = []
        for w, l in zip(freq_v, re_v):
            g = sqrt(l * w)
            if use_cross:
                _add_v = corr.add_discrete_trigonometric
            else:
                _add_v = corr.add_discrete_vibration
            _add_v(quantity(w, 'energy'), quantity(g, 'energy'), beta)

    # Add the Lindblad rate
    if include_lindblad and beta is not None:
        # summerize the lindblad rate from the reorganization energy and temperature
        if corr.lindblad_rate is not None:
            raise NotImplementedError(
                'Lindblad rate calculation is not implemented yet. '
                'Specify .lindblad_rate manually if needed.'
            )
        # print('Calculating Lindblad rate', flush=True)
        # print('Current')
        # re = 0.0
        # if include_drude:
        #     re += sum(re_d)
        # if include_brownian:
        #     re += sum(re_b)
        # if include_discrete:
        #     re += sum(re_v)
        # re = quantity(re, 'energy')
        # corr.lindblad_rate = 2.0 * re / beta
    return corr


def gen_tedopa_bcf(
    # Drude bath
    re_d: list[float],
    width_d: list[float],
    # Brownian bath
    freq_b: list[float],
    re_b: list[float],
    width_b: list[float],
    # Misc bath settings
    temperature: float,
    frequency_cutoff: float,
    n_frequency: int,
    n_discretization: int,
    cutoff_type: str = 'Lorentz',
    include_brownian: bool = True,
    include_drude: bool = True,
    **kwargs,
) -> StarBosons:
    """
    Generate a correlation function for a composite spectral density based on TEDOPA-type of decompostion 
    but without changing to the chain-like picture.

    Parameters
    ----------
    include_drude : bool
        Whether to include the Drude term for the low-frequency modes.
    re_d : float
        Reorganization energy of the Drude term.
    width_d : float
        Cutoff frequency of the Drude term.
    cutoff_type : str
        Type of cutoff function for the Drude term.
        Allowed values are 'Lorentz' (Drude-Lorentz) and 'Exp' (Ohmic with exponential cutoff).
    include_brownian : bool
        Whether to include the Brownian term for the high-frequency modes.
    freq_b : list[float]
        Frequencies of the Brownian modes.
    re_b : list[float]
        Reorganization energies of the Brownian modes.
    width_b : list[float]
        Broadening of the Brownian modes.
    temperature : float
        Temperature of the bath.
    frequency_cutoff : float
        Cutoff frequency for the spectral density.
    n_frequency : int
        Number of frequency points for the spectral density.
    n_discretization : int
        Number of discretization points for the spectral density.
    **kwargs : dict
        Ignored arguments.

    Returns
    -------
    StarBosons
        The correlation function for the composite spectral density. 
    """

    # Bath settings:
    if temperature is not None:
        beta = quantity(1 / temperature, 'inverse_temperature')
    else:
        beta = None
    sds = []  # type:list[SpectralDensity]
    if include_drude:
        if cutoff_type == 'Lorentz':
            for l, g in zip(re_d, width_d):
                drude = Drude(quantity(l, 'energy'), quantity(g, 'energy'))
                sds.append(drude)
        elif cutoff_type == 'Exp':
            for l, g in zip(re_d, width_d):
                drude = OhmicExp(quantity(l, 'energy'), quantity(g, 'energy'))
                sds.append(drude)
        else:
            raise ValueError('Invalid cutoff type')

    if include_brownian:
        for w, l, g in zip(freq_b, re_b, width_b):
            b = UnderdampedBrownian(quantity(l,
                                             'energy'), quantity(w, 'energy'),
                                    quantity(g, 'energy'))
            sds.append(b)

    freq_space = np.linspace(
        0,
        quantity(frequency_cutoff, 'energy'),
        n_frequency,
    )
    j = np.zeros_like(freq_space)
    for sd in sds:
        j += sd.function(freq_space)
    tedopa_solver = Tedopa(freq_space, j, beta, n_discretization)
    ww, gg = tedopa_solver.star_parameters()
    corr = StarBosons()
    for i, (w, g) in enumerate(zip(ww, gg)):
        corr.couplings.append(complex(g))
        corr.conj_couplings.append(complex(g))
        corr.frequencies[i, i] = -1.0j * w
    return corr


def gen_heom_like_star_boson(
    # Drude bath
    include_drude: bool = True,
    re_d: list[float] | None = None,
    width_d: list[float] | None = None,
    # Brownian bath
    include_brownian: bool = True,
    freq_b: list[float] | None = None,
    re_b: list[float] | None = None,
    width_b: list[float] | None = None,
    # Vibrational bath
    include_discrete: bool = True,
    freq_v: list[float] | None = None,
    re_v: list[float] | None = None,
    # LTC bath
    temperature: float = 300.0,
    decomposition_method: str = 'Pade',
    n_ltc: int = 0,
    include_lindblad: bool = False,
    # Debugging
    use_cross: bool = False,
    use_ht_function: bool = False,
    **kwargs,
) -> StarBosons:
    """
    Generate a correlation function for a composite spectral density for HSEOM but based on HEOM-style BCF decompostion.

    Parameters
    ----------
    include_drude : bool
        Whether to include the Drude term for the low-frequency modes.
    re_d : list[float]
        Reorganization energies of every Drude modes.
    width_d : list[float]
        Cutoff frequencies of every Drude modes.
    include_brownian : bool
        Whether to include the Brownian term for the high-frequency modes.
    freq_b : list[float]
        Frequencies of every Brownian modes.
    re_b : list[float]
        Reorganization energies of every Brownian modes.
    width_b : list[float]
        Broadening of every Brownian modes.
    include_discrete : bool
        Whether to include the discrete vibrational modes.
    freq_v : list[float]
        Frequencies of every discrete vibrational modes.
    re_v : list[float]
        Reorganization energies of every discrete vibrational modes.
    temperature : float
        Temperature of the bath.
    decomposition_method : str
        Method to decompose the Bose-Einstein distribution.
    n_ltc : int
        Number of low-temperature correction terms in the decomposition of the Bose-Einstein distribution.
    include_lindblad : bool
        Whether to include the Lindblad rate as in Tanimura's HEOM.
    use_cross : bool
        Whether to use the cross-correlation that includes the trigonometric functions.
    use_ht_function : bool
        Whether to use the high-temperature approximation [2 / (beta omega)] of the Bose-Einstein distribution
        instead of the hyperbolic tangent function.
    
    Returns
    -------
    Correlation
        The correlation function for the composite spectral density.
    """
    # Bath settings:
    corr = Correlation()
    if temperature is not None:
        beta = quantity(1 / temperature, 'inverse_temperature')
    else:
        beta = None

    # Add continuous spectral densities
    BoseEinstein.decomposition_method = decomposition_method
    distr = BoseEinstein(n=n_ltc, beta=beta)
    sds = []  # type:list[SpectralDensity]
    if include_drude:
        if re_d is None:
            re_d = []
        if width_d is None:
            width_d = []
        for l, g in zip(re_d, width_d):
            drude = Drude(quantity(l, 'energy'), quantity(g, 'energy'))
            sds.append(drude)
    if include_brownian:
        if freq_b is None:
            freq_b = []
        if re_b is None:
            re_b = []
        if width_b is None:
            width_b = []
        for w, l, g in zip(freq_b, re_b, width_b):
            b = UnderdampedBrownian(quantity(l,
                                             'energy'), quantity(w, 'energy'),
                                    quantity(g, 'energy'))
            sds.append(b)
    if use_cross:
        corr.add_trigonometric(sds, distr)
    else:
        corr.add_spectral_densities(sds,
                                    distr,
                                    use_ht_function=use_ht_function)

    # Add discrete vibrational modes
    if include_discrete:
        if freq_v is None:
            freq_v = []
        if re_v is None:
            re_v = []
        for w, l in zip(freq_v, re_v):
            g = sqrt(l * w)
            if use_cross:
                _add_v = corr.add_discrete_trigonometric
            else:
                _add_v = corr.add_discrete_vibration
            _add_v(quantity(w, 'energy'), quantity(g, 'energy'), beta)

    # Find the proper star-boson representation
    bath = StarBosons()
    for k in range(corr.k_max):
        c = corr.coefficients[k]
        r = np.sqrt(np.abs(c))
        theta = np.angle(c) / 2.0 - np.pi  # Assuse in the lower half plane
        assert theta < 0
        print(c, (r * np.exp(1j * theta))**2)
        coupling = r * np.exp(1j * theta)  # imaginary part is negative
        conj_coupling = r * np.exp(1j * theta)  # imaginary part is negative
        frequency = corr.derivatives[k, k]
        bath.couplings.append(coupling)
        bath.conj_couplings.append(conj_coupling)
        bath.frequencies[k, k] = frequency
    return bath



def gen_star_boson(
    # Misc bath settings
    temperature: float,
    cutoff: float,
    n_discretization: int,
    # Drude bath
    re_d: list[float] = None,
    width_d: list[float] = None,
    # Brownian bath
    freq_b: list[float] = None,
    re_b: list[float] = None,
    width_b: list[float] = None,
    # Vibrational bath
    freq_v: list[float] = None,
    re_v: list[float] = None,
    # bath settings with default values
    include_drude: bool = True,
    include_brownian: bool = True,
    include_discrete: bool = True,
    discretization_method: str = 'Fourier',
    ohmic_type: str = 'Lorentz',
    shift_frequency: bool = True,
    absorb_rate: None | float = None,
    log_base: int = 10,
    log_minimum_frequency: None | float = None,
    int_grid_size: float | None = None,
    int_lower_bound: float | None = None,
    **kwargs,
) -> StarBosons:
    """
    Generate a correlation function for a composite spectral density for the star-like decomposition as in spin-boson model.
    
    Parameters
    ----------
    include_drude : bool
        Whether to include the Drude term for the low-frequency modes.
    re_d : list[float]
        Reorganization energies of the Drude modes.
    width_d : list[float]
        Cutoff frequencies of the Drude modes.
    include_brownian : bool
        Whether to include the Brownian term for the high-frequency modes.
    freq_b : list[float]
        Frequencies of the Brownian modes.
    re_b : list[float]
        Reorganization energies of the Brownian modes.
    width_b : list[float]
        Broadening of the Brownian modes.
    include_discrete : bool
        Whether to include the discrete vibrational modes.
    freq_v : list[float]    
        Frequencies of the discrete vibrational modes.
    re_v : list[float]
        Reorganization energies of the discrete vibrational modes.
    temperature : float
        Temperature of the bath.
    cutoff : float
        Cutoff frequency for the spectral density.
    n_discretization : int
        Number of discretization points for the spectral density.
    discretization_method : str
        Method to discretize the spectral density.
        Allowed values are 'Fourier', 'LogFourier', 'EqualReorganizationEnergy', 'Chebyshev', and 'TEDOPA'.
        Currently, only 'Fourier', 'LogFourier', 'EqualReorganizationEnergy' are tested.
    ohmic_type : str
        Type of the Drude term.
        Allowed values are 'Lorentz' (Drude-Lorentz) and 'Exp' (Ohmic with exponential cutoff).
    shift_frequency : bool
        Whether to shift the frequency to the positive domain.
    **kwargs : dict
        Ignored arguments.

    Returns
    -------
    StarBosons
        The correlation function for the composite spectral density.
    """

    if temperature is not None:
        beta = quantity(1 / temperature, 'inverse_temperature')
    else:
        beta = None

    sds = []  # type:list[SpectralDensity]
    if include_drude and re_d is not None and width_d is not None:
        if ohmic_type == 'Lorentz':
            drude_cls = Drude
        elif ohmic_type == 'Exp':
            drude_cls = OhmicExp
        else:
            raise ValueError('Invalid Drude type')
        for l, g in zip(re_d, width_d):
            drude = drude_cls(quantity(l, 'energy'), quantity(g, 'energy'))
            sds.append(drude)

    if include_brownian and freq_b is not None and re_b is not None and width_b is not None:
        for w, l, g in zip(freq_b, re_b, width_b):
            b = UnderdampedBrownian(quantity(l,
                                             'energy'), quantity(w, 'energy'),
                                    quantity(g, 'energy'))
            sds.append(b)

    vib_parameters = []
    if include_discrete and freq_v is not None and re_v is not None:
        for w, l in zip(freq_v, re_v):
            g = sqrt(l * w)
            vib_parameters.append((quantity(w,
                                            'energy'), quantity(g, 'energy')))

    corr = StarBosons()
    if sds:
        corr.add_spectral_densities(
            sds,
            beta,
            n_discretization,
            cutoff=quantity(cutoff, 'energy'),
            method=discretization_method,
            shift_frequency=shift_frequency,
            log_base=log_base,
            log_minimum_frequency=(quantity(log_minimum_frequency, 'energy')
                                   if log_minimum_frequency is not None else
                                   None),
            absorb_rate=absorb_rate,
            int_grid_size=(quantity(int_grid_size, 'energy')
                           if int_grid_size is not None else None),
            int_lower_bound=(quantity(int_lower_bound, 'energy')
                             if int_lower_bound is not None else None),
        )
    if vib_parameters:
        corr.add_discrete_vibrations(vib_parameters, beta)
    return corr


"""
def gen_real_bcf(
    unit_energy: float,
    # Drude bath
    include_drude: bool,
    re_d: list[float],
    width_d: list[float],
    # Brownian bath
    include_brownian: bool,
    freq_b: list[float],
    re_b: list[float],
    width_b: list[float],
    # Vibrational bath
    include_discrete: bool,
    freq_v: list[float],
    re_v: list[float],
    # LTC bath
    temperature: float,
    decomposition_method: str,
    n_ltc: int,
    **kwargs,
) -> RealCorrelation:
    # Generate a correlation function for a composite spectral density but allows zero fluctuation.
    # Bath settings:
    corr = RealCorrelation()
    beta = quantity(1 / temperature, 'inverse_temperature')
    BoseEinstein.decomposition_method = decomposition_method
    distr = BoseEinstein(n=n_ltc, beta=beta)

    sds = []  # type:list[SpectralDensity]
    if include_drude:
        for l, g in zip(re_d, width_d):
            drude = Drude(quantity(l, 'energy'), quantity(g, 'energy'))
            sds.append(drude)
    if include_brownian:
        for w, l, g in zip(freq_b, re_b, width_b):
            b = UnderdampedBrownian(quantity(l,
                                             'energy'), quantity(w, 'energy'),
                                    quantity(g, 'energy'))
            sds.append(b)
    corr.add_spectral_densities(sds, distr)

    if include_discrete:
        for w, l in zip(freq_v, re_v):
            g = sqrt(l * w)
            corr.add_discrete_vibration(quantity(w, 'energy'),
                                        quantity(g, 'energy'), beta)

    return corr
"""
