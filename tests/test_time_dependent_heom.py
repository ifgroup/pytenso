"""This file contains a few short calculations with time dependent 
Hamiltonians. The same sort of spot checks for exact numbers and
then thorough matrix legality checks are carried out.

Some of these tests require slightly more numerical tolerance than
the standard pytest relative tolerance of 1e-6.
"""
from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm
from tenso.prototypes.bath import gen_bcf
from math import ceil
from tenso.prototypes.heom import system_multibath
from tenso.prototypes.heom import system_single_bath 
from tenso.prototypes.default_parameters import quantity
import pytest

def test_td1(subtests,tmp_path):
    """ Tests a noisy Hadamard gate by means of a time-dependent
    Hamiltonian using the spin-boson propagator
    """
    bath_simulation = gen_bcf(
        re_d=[540], # Reorganization energy (cm^{-1})
        width_d=[70], # Width of Drude-Lorentz spectral density (cm^{-1})
        freq_b=[1243], # Brownian spectral density frequency (cm^{-1})
        re_b=[161.6], # Brownian spectral reorganization energy (cm^{-1})
        width_b=[10], # Brownian spectral width (cm^{-1})
        temperature=300, # Temperature of the bath
        decomposition_method='Pade',
        n_ltc=1,) # Low temperature correction terms
    # 1. PHYSICAL CONSTANTS & UNIT CONVERSIONS
    # --- Standard SI Constants ---
    h_Js          = 6.62607015e-34      # Planck constant (J*s)
    hbar_Js       = 1.054571817e-34     # Reduced Planck constant (J*s)
    c_ms          = 299792458.0         # Speed of light (m/s)
    c_cms         = c_ms * 100.0        # Speed of light (cm/s)
    # --- Atomic Unit Conversions ---
    Hartree_to_cm = 219474.6313705 
    fs_to_au      = 41.341374576 
    # 2. INPUT PARAMETERS
    # --- System Parameters ---
    gap_cm  = 1500.0              # Energy gap in cm^-1
    mu_au         = 3.31356454592       # Dipole moment (a.u.)
    # --- Drive (Laser) Parameters ---
    FWHM_fs       = 5.0                 # Pulse duration (FWHM) in fs
    t0_fs         = 6.0                 # Pulse center time in fs
    target_area   = np.pi               # Target pulse area (Pi pulse)
    # 3. CALCULATIONS & CONVERSIONS
    # --- A. Frequency Conversion (cm^-1 -> rad/fs) ---
    # 1. Convert cm^-1 to Joules (E = h * c * wavenumber)
    E_Joules = gap_cm * h_Js * c_cms
    # 2. Convert Joules to Angular Frequency in rad/s (omega = E / hbar)
    omega_rad_s = E_Joules / hbar_Js
    # 3. Convert rad/s to rad/fs (1 fs = 1e-15 s)
    omega_rad_fs = omega_rad_s * 1e-15
    # --- B. Field Amplitude & Envelope ---
    # 1. Convert FWHM to Gaussian Sigma
    sigma_fs = FWHM_fs / (2 * np.sqrt(2 * np.log(2)))
    # 2. Convert Sigma to Atomic Units (needed for Area calculation)
    sigma_au_val = sigma_fs * fs_to_au
    # 3. Calculate Electric Field Amplitude (E0) in a.u.
    # Formula: Area = mu * E0 * sigma * sqrt(2*pi)
    E0_au = target_area / (mu_au * sigma_au_val * np.sqrt(2 * np.pi))
    # 4. Convert Max Interaction Energy to cm^-1
    interaction_max_cm = (mu_au * E0_au) * Hartree_to_cm
    # 4. LASER FIELD FUNCTION
    def laser_field_hadamard(t):
        """
        Returns the laser field interaction energy at time t.
        t: Time in femtoseconds (fs)
        """
        envelope = np.exp(-0.5 * ((t - t0_fs) / sigma_fs)**2)
        # Unitless argument inside cos: (rad/fs) * fs = radians
        carrier = np.cos(omega_rad_fs * t)
        return -interaction_max_cm * envelope * carrier

    end_time = 5.05 # fs
    dt = 0.05 # fs
    wfn = np.array([0.0, 1.0], dtype=np.complex128) # Start in Ground State |0> (convention [exc, gnd])
    out=os.path.join(tmp_path,'hr1')

    propagator = system_single_bath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        # System Hamiltonian
        sys_ham=np.array([[gap_cm/2, 0.0], [0.0, -gap_cm/2]], 
                         dtype=np.complex128),
        # System-Bath coupling operator (Sigma_z / 2)
        sys_op=np.array([[0.5, 0.0], [0.0, -0.5]], 
                        dtype=np.complex128),
        bath_correlation=bath_simulation,
        # Time-dependent driving function & Operator
        td_f=laser_field_hadamard,
        td_op=np.array([[0.0, 1.0], [1.0, 0.0]],
                       dtype=np.complex128), #(sigma_x)
        dim=30,
        end_time=end_time,
        step_time=dt,
    )
    # Perform the dynamics simulation and generate a progress bar
    progress_bar = tqdm(propagator,total=ceil(end_time/dt))
    for _t in progress_bar:
       progress_bar.set_description(f'@{_t: .2f} fs')

    # Load results from the output file into a local array
    results = np.loadtxt(out+".dat.log",dtype=np.complex128)
    assert results[50,1].real == pytest.approx(0.01599877,rel=1e-4)
    assert results[50,1].imag == pytest.approx(0.0,rel=1e-4)
    assert results[75,3].real == pytest.approx(0.09724658,rel=1e-4)
    assert results[75,3].imag == pytest.approx(-0.26203125,rel=1e-4) 
    assert results[100,2].real == pytest.approx(0.20570225,rel=1e-4)
    assert results[100,2].imag == pytest.approx(0.32263329,rel=1e-4)
    for i in range(0,10):
        with subtests.test("Values add to 1", i =i):
            assert results[i,1].real + results[i,4] == pytest.approx(1.0)
    for i in range(0,10):
        with subtests.test("No imaginary Populations", i=i):
            assert results[i,1].imag <= 1e-5
            assert results[i,4].imag <= 1e-5
    for i in range(0,10):
        with subtests.test("No negative Populations", i=i):
            assert np.absolute(results[i,1].real) >= -1e-5
            assert np.absolute(results[i,4].real) >= -1e-5
    for i in range(0,10):
        with subtests.test("Hermitian matrix", i=i):
            assert results[i,2].imag == pytest.approx(-1.0*results[i,3].imag)
            assert results[i,2].real == pytest.approx(results[i,3].real)
    for i in range(0,10):
        with subtests.test("No illegally large coherences", i=i): # Tolerance of 1% allowed
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,2])
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,3])




def test_td2(subtests,tmp_path):
    """ Tests a noisy Hadamard gate by means of a time-dependent
    Hamiltonian using the system_multibath propagator
    """
    bath_simulation = gen_bcf(
        re_d=[540], # Reorganization energy (cm^{-1})
        width_d=[70], # Width of Drude-Lorentz spectral density (cm^{-1})
        freq_b=[1243], # Brownian spectral density frequency (cm^{-1})
        re_b=[161.6], # Brownian spectral reorganization energy (cm^{-1})
        width_b=[10], # Brownian spectral width (cm^{-1})
        temperature=300, # Temperature of the bath
        decomposition_method='Pade',
        n_ltc=1,) # Low temperature correction terms
    # 1. PHYSICAL CONSTANTS & UNIT CONVERSIONS
    # --- Standard SI Constants ---
    h_Js          = 6.62607015e-34      # Planck constant (J*s)
    hbar_Js       = 1.054571817e-34     # Reduced Planck constant (J*s)
    c_ms          = 299792458.0         # Speed of light (m/s)
    c_cms         = c_ms * 100.0        # Speed of light (cm/s)
    # --- Atomic Unit Conversions ---
    Hartree_to_cm = 219474.6313705 
    fs_to_au      = 41.341374576 
    # 2. INPUT PARAMETERS
    # --- System Parameters ---
    gap_cm  = 1500.0              # Energy gap in cm^-1
    mu_au         = 3.31356454592       # Dipole moment (a.u.)
    # --- Drive (Laser) Parameters ---
    FWHM_fs       = 5.0                 # Pulse duration (FWHM) in fs
    t0_fs         = 6.0                 # Pulse center time in fs
    target_area   = np.pi               # Target pulse area (Pi pulse)
    # 3. CALCULATIONS & CONVERSIONS
    # --- A. Frequency Conversion (cm^-1 -> rad/fs) ---
    # 1. Convert cm^-1 to Joules (E = h * c * wavenumber)
    E_Joules = gap_cm * h_Js * c_cms
    # 2. Convert Joules to Angular Frequency in rad/s (omega = E / hbar)
    omega_rad_s = E_Joules / hbar_Js
    # 3. Convert rad/s to rad/fs (1 fs = 1e-15 s)
    omega_rad_fs = omega_rad_s * 1e-15
    # --- B. Field Amplitude & Envelope ---
    # 1. Convert FWHM to Gaussian Sigma
    sigma_fs = FWHM_fs / (2 * np.sqrt(2 * np.log(2)))
    # 2. Convert Sigma to Atomic Units (needed for Area calculation)
    sigma_au_val = sigma_fs * fs_to_au
    # 3. Calculate Electric Field Amplitude (E0) in a.u.
    # Formula: Area = mu * E0 * sigma * sqrt(2*pi)
    E0_au = target_area / (mu_au * sigma_au_val * np.sqrt(2 * np.pi))
    # 4. Convert Max Interaction Energy to cm^-1
    interaction_max_cm = (mu_au * E0_au) * Hartree_to_cm
    # 4. LASER FIELD FUNCTION
    def laser_field_hadamard(t):
        """
        Returns the laser field interaction energy at time t.
        t: Time in femtoseconds (fs)
        """
        envelope = np.exp(-0.5 * ((t - t0_fs) / sigma_fs)**2)
        # Unitless argument inside cos: (rad/fs) * fs = radians
        carrier = np.cos(omega_rad_fs * t)
        return -interaction_max_cm * envelope * carrier

    end_time = 5.05 # fs
    dt = 0.05 # fs
    wfn = np.array([0.0, 1.0], dtype=np.complex128) # Start in Ground State |0> (convention [exc, gnd])
    out=os.path.join(tmp_path,'hr1')

    propagator = system_multibath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        # System Hamiltonian
        sys_ham=np.array([[gap_cm/2, 0.0], [0.0, -gap_cm/2]], 
                         dtype=np.complex128),
        # System-Bath coupling operator (Sigma_z / 2)
        sys_ops=[np.array([[0.5, 0.0], [0.0, -0.5]], 
                        dtype=np.complex128)],
        bath_correlations=[bath_simulation],
        # Time-dependent driving function & Operator
        td_f=laser_field_hadamard,
        td_op=np.array([[0.0, 1.0], [1.0, 0.0]],
                       dtype=np.complex128), #(sigma_x)
        dim=30,
        end_time=end_time,
        step_time=dt,
    )
    # Perform the dynamics simulation and generate a progress bar
    progress_bar = tqdm(propagator,total=ceil(end_time/dt))
    for _t in progress_bar:
       progress_bar.set_description(f'@{_t: .2f} fs')

    # Load results from the output file into a local array
    results = np.loadtxt(out+".dat.log",dtype=np.complex128)
    assert results[50,1].real == pytest.approx(0.01599877,rel=1e-4)
    assert results[50,1].imag == pytest.approx(0.0,rel=1e-4)
    assert results[75,3].real == pytest.approx(0.09724658,rel=1e-4)
    assert results[75,3].imag == pytest.approx(-0.26203125,rel=1e-4) 
    assert results[100,2].real == pytest.approx(0.20570225,rel=1e-4)
    assert results[100,2].imag == pytest.approx(0.32263329,rel=1e-4)
    for i in range(0,10):
        with subtests.test("Values add to 1", i =i):
            assert results[i,1].real + results[i,4] == pytest.approx(1.0)
    for i in range(0,10):
        with subtests.test("No imaginary Populations", i=i):
            assert results[i,1].imag <= 1e-5
            assert results[i,4].imag <= 1e-5
    for i in range(0,10):
        with subtests.test("No negative Populations", i=i):
            assert np.absolute(results[i,1].real) >= -1e-5
            assert np.absolute(results[i,4].real) >= -1e-5
    for i in range(0,10):
        with subtests.test("Hermitian matrix", i=i):
            assert results[i,2].imag == pytest.approx(-1.0*results[i,3].imag)
            assert results[i,2].real == pytest.approx(results[i,3].real)
    for i in range(0,10):
        with subtests.test("No illegally large coherences", i=i): # Tolerance of 1% allowed
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,2])
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,3])


def test_td3(subtests,tmp_path):
    """ Tests a case with no carrier wave, only Gaussian.
    """
    # Generate the decomposed bath correlation function
    # it for HEOM propagation
    bath_simulation = gen_bcf(
        re_d=[300], # Drude-Lorentz Reorganization energy (cm^{-1})
        width_d=[30], # Drude-Lorentz spectral density cut-off frequency (cm^{-1})
        freq_b=[1400], # Brownian spectral density frequency (cm^{-1})
        re_b=[100.0], # Brownian spectral reorganization energy (cm^{-1})
        width_b=[8], # Brownian spectral width (cm^{-1})
        temperature=200, # Temperature of the bath
        decomposition_method='Pade',
        n_ltc=3,) # Low temperature correction terms

    # --- 1. Physical Constants & System Parameters ---
    Ha_to_cm = 219474.63    # 1 Hartree in cm^-1
    fs_to_au = 41.341       # 1 fs in atomic units (time)

    gap_cm = 1000.0       # Energy gap (omega_0) in cm^-1
    mu_au  = 2.5          # Transition dipole moment (a.u.)

    # --- 2. Pulse Physics (Rabi Area Calculation) ---
    FWHM_fs = 5.0
    t0_fs   = 6.0           # Pulse center (fs)
    sigma_fs = FWHM_fs / 2.35482 # FWHM to Gaussian Sigma

    # We require a Pi-pulse: Area = mu * Integral(E(t))dt
    # Area = mu * E0 * sigma_au * sqrt(2*pi)
    target_area = np.pi 
    sigma_au    = sigma_fs * fs_to_au

    # Solve for Electric Field Amplitude E0 (a.u.)
    denom = mu_au * sigma_au * np.sqrt(2 * np.pi)
    E0_au = target_area / denom

    # Max Interaction Energy V0 = mu * E0 (cm^-1)
    V0_cm = (mu_au * E0_au) * Ha_to_cm

    sigma_fs = FWHM_fs / (2 * np.sqrt(2 * np.log(2)))
    def laser_field_hadamard(t):
        """Returns interaction V(t) = -mu * E(t)"""
        # Gaussian Envelope
        env = np.exp(-0.5 * ((t - t0_fs)/sigma_fs)**2)
        return -V0_cm * env 

    # --- 4. Propagator Construction ---
    end_time = 5.05; 
    dt = 0.05
    wfn = np.array([0.0, 1.0], dtype=np.complex128) 
    out=os.path.join(tmp_path,'hr1')
    propagator = system_single_bath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        # System Hamiltonian
        sys_ham=np.array([[gap_cm/2, 0.0], [0.0, -gap_cm/2]], 
                         dtype=np.complex128),
        # System-Bath coupling operator (Sigma_z / 2)
        sys_op=np.array([[0.5, 0.0], [0.0, -0.5]], 
                        dtype=np.complex128),
        bath_correlation=bath_simulation,
        # Time-dependent driving function & Operator
        td_f=laser_field_hadamard,
        td_op=np.array([[0.0, 1.0], [1.0, 0.0]],
                       dtype=np.complex128), #(sigma_x)
        dim=30,
        end_time=end_time,
        step_time=dt,
    )

    progress_bar = tqdm(propagator,total=ceil(end_time/dt))
    for _t in progress_bar:
        progress_bar.set_description(f'@{_t: .2f} fs')
    # Load results from the output file into a local array
    results = np.loadtxt(out+".dat.log",dtype=np.complex128)
    assert results[50,1].real == pytest.approx(0.02154068,rel=1e-4)
    assert results[50,1].imag == pytest.approx(0.0,rel=1e-4)
    assert results[75,3].real == pytest.approx(0.08207166,rel=1e-4)
    assert results[75,3].imag == pytest.approx(-0.37433094,rel=1e-4) 
    assert results[100,2].real == pytest.approx(0.19069181,rel=1e-4)
    assert results[100,2].imag == pytest.approx(0.42186558,rel=1e-4)
    for i in range(0,10):
        with subtests.test("Values add to 1", i =i):
            assert results[i,1].real + results[i,4] == pytest.approx(1.0)
    for i in range(0,10):
        with subtests.test("No imaginary Populations", i=i):
            assert results[i,1].imag <= 1e-5
            assert results[i,4].imag <= 1e-5
    for i in range(0,10):
        with subtests.test("No negative Populations", i=i):
            assert np.absolute(results[i,1].real) >= -1e-5
            assert np.absolute(results[i,4].real) >= -1e-5
    for i in range(0,10):
        with subtests.test("Hermitian matrix", i=i):
            assert results[i,2].imag == pytest.approx(-1.0*results[i,3].imag)
            assert results[i,2].real == pytest.approx(results[i,3].real)
    for i in range(0,10):
        with subtests.test("No illegally large coherences", i=i): # Tolerance of 1% allowed
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,2])
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,3])


def test_td4(subtests,tmp_path):
    """ Tests a case with no carrier wave, only Gaussian.
    """
    # Generate the decomposed bath correlation function
    # it for HEOM propagation
    bath_simulation = gen_bcf(
        re_d=[300], # Drude-Lorentz Reorganization energy (cm^{-1})
        width_d=[30], # Drude-Lorentz spectral density cut-off frequency (cm^{-1})
        freq_b=[1400], # Brownian spectral density frequency (cm^{-1})
        re_b=[100.0], # Brownian spectral reorganization energy (cm^{-1})
        width_b=[8], # Brownian spectral width (cm^{-1})
        temperature=200, # Temperature of the bath
        decomposition_method='Pade',
        n_ltc=3,) # Low temperature correction terms

    # --- 1. Physical Constants & System Parameters ---
    Ha_to_cm = 219474.63    # 1 Hartree in cm^-1
    fs_to_au = 41.341       # 1 fs in atomic units (time)

    gap_cm = 1000.0       # Energy gap (omega_0) in cm^-1
    mu_au  = 2.5          # Transition dipole moment (a.u.)

    # --- 2. Pulse Physics (Rabi Area Calculation) ---
    FWHM_fs = 5.0
    t0_fs   = 6.0           # Pulse center (fs)
    sigma_fs = FWHM_fs / 2.35482 # FWHM to Gaussian Sigma

    # We require a Pi-pulse: Area = mu * Integral(E(t))dt
    # Area = mu * E0 * sigma_au * sqrt(2*pi)
    target_area = np.pi 
    sigma_au    = sigma_fs * fs_to_au

    # Solve for Electric Field Amplitude E0 (a.u.)
    denom = mu_au * sigma_au * np.sqrt(2 * np.pi)
    E0_au = target_area / denom

    # Max Interaction Energy V0 = mu * E0 (cm^-1)
    V0_cm = (mu_au * E0_au) * Ha_to_cm

    sigma_fs = FWHM_fs / (2 * np.sqrt(2 * np.log(2)))
    def laser_field_hadamard(t):
        """Returns interaction V(t) = -mu * E(t)"""
        # Gaussian Envelope
        env = np.exp(-0.5 * ((t - t0_fs)/sigma_fs)**2)
        return -V0_cm * env 

    # --- 4. Propagator Construction ---
    end_time = 5.05; 
    dt = 0.05
    wfn = np.array([0.0, 1.0], dtype=np.complex128) 
    out=os.path.join(tmp_path,'hr1')
    propagator = system_multibath(
        fname=out,
        init_rdo=np.outer(wfn, wfn.conj()),
        # System Hamiltonian
        sys_ham=np.array([[gap_cm/2, 0.0], [0.0, -gap_cm/2]], 
                         dtype=np.complex128),
        # System-Bath coupling operator (Sigma_z / 2)
        sys_ops=[np.array([[0.5, 0.0], [0.0, -0.5]], 
                        dtype=np.complex128)],
        bath_correlations=[bath_simulation],
        # Time-dependent driving function & Operator
        td_f=laser_field_hadamard,
        td_op=np.array([[0.0, 1.0], [1.0, 0.0]],
                       dtype=np.complex128), #(sigma_x)
        dim=30,
        end_time=end_time,
        step_time=dt,
    )

    progress_bar = tqdm(propagator,total=ceil(end_time/dt))
    for _t in progress_bar:
        progress_bar.set_description(f'@{_t: .2f} fs')
    # Load results from the output file into a local array
    results = np.loadtxt(out+".dat.log",dtype=np.complex128)
    assert results[50,1].real == pytest.approx(0.02154068,rel=1e-4)
    assert results[50,1].imag == pytest.approx(0.0,rel=1e-4)
    assert results[75,3].real == pytest.approx(0.08207166,rel=1e-4)
    assert results[75,3].imag == pytest.approx(-0.37433094,rel=1e-4) 
    assert results[100,2].real == pytest.approx(0.19069181,rel=1e-4)
    assert results[100,2].imag == pytest.approx(0.42186558,rel=1e-4)
    for i in range(0,10):
        with subtests.test("Values add to 1", i =i):
            assert results[i,1].real + results[i,4] == pytest.approx(1.0)
    for i in range(0,10):
        with subtests.test("No imaginary Populations", i=i):
            assert results[i,1].imag <= 1e-5
            assert results[i,4].imag <= 1e-5
    for i in range(0,10):
        with subtests.test("No negative Populations", i=i):
            assert np.absolute(results[i,1].real) >= -1e-5
            assert np.absolute(results[i,4].real) >= -1e-5
    for i in range(0,10):
        with subtests.test("Hermitian matrix", i=i):
            assert results[i,2].imag == pytest.approx(-1.0*results[i,3].imag)
            assert results[i,2].real == pytest.approx(results[i,3].real)
    for i in range(0,10):
        with subtests.test("No illegally large coherences", i=i): # Tolerance of 1% allowed
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,2])
            assert np.sqrt(np.absolute(results[i,1]))*np.sqrt(np.absolute(results[i,4])) >= 0.99*np.abs(results[i,3])



