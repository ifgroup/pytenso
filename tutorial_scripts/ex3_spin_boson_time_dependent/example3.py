from math import ceil
import os
import json as json
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath #Propagator setup function
from tenso.prototypes.bath import gen_bcf #Bath correlation function generator
import matplotlib.pyplot as plt
from tenso.prototypes.default_parameters import quantity


bath_simulation = gen_bcf(
  re_d=[540], #Reorganization energy in cm-1
  width_d=[70], #Width of the DL spectral density in cm-1
  freq_b=[1243], # Central Frequency of the Brownian Spectral density in cm-1
  re_b=[161.6], #Reorganization energy of the Brownian Spectral density in cm-1
  width_b=[10], #Width of the spectral density in cm-1
  temperature=300, #Temperature in Kelvin
  decomposition_method='Pade', #Decomposition method for the bath correlation function
  n_ltc=1, #Number of low-temperature correction terms
)


end_time = 40.0 # fs
dt = 0.05 # fs
wfn = np.array([0.0, 1.0], dtype=np.complex128) # Start in Ground State |0> (convention [exc, gnd])

import numpy as np

# ==========================================
# 1. PHYSICAL CONSTANTS & UNIT CONVERSIONS
# ==========================================
# --- Standard SI Constants ---
h_Js          = 6.62607015e-34      # Planck constant (J*s)
hbar_Js       = 1.054571817e-34     # Reduced Planck constant (J*s)
c_ms          = 299792458.0         # Speed of light (m/s)
c_cms         = c_ms * 100.0        # Speed of light (cm/s)

# --- Atomic Unit Conversions ---
Hartree_to_cm = 219474.6313705 
fs_to_au      = 41.341374576 

# ==========================================
# 2. INPUT PARAMETERS
# ==========================================
# --- System Parameters ---
delta_Eps_cm  = 1500.0              # Energy gap in cm^-1
mu_au         = 3.31356454592       # Dipole moment (a.u.)

# --- Drive (Laser) Parameters ---
FWHM_fs       = 5.0                 # Pulse duration (FWHM) in fs
t0_fs         = 6.0                 # Pulse center time in fs
target_area   = np.pi               # Target pulse area (Pi pulse)

# ==========================================
# 3. CALCULATIONS & CONVERSIONS
# ==========================================

# --- A. Frequency Conversion (cm^-1 -> rad/fs) ---
# 1. Convert cm^-1 to Joules (E = h * c * wavenumber)
E_Joules = delta_Eps_cm * h_Js * c_cms

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

# ==========================================
# 4. LASER FIELD FUNCTION
# ==========================================

def laser_field_hadamard(t):
    """
    Returns the laser field interaction energy at time t.
    t: Time in femtoseconds (fs)
    """
    # Gaussian Envelope
    # Unitless argument inside exp: (fs / fs)^2
    envelope = np.exp(-0.5 * ((t - t0_fs) / sigma_fs)**2)
    
    # Oscillatory Carrier
    # Unitless argument inside cos: (rad/fs) * fs = radians
    carrier = np.cos(omega_rad_fs * t)
    
    return -interaction_max_cm * envelope * carrier

# ==========================================
# 5. OUTPUT & VERIFICATION
# ==========================================
print(f"--- Frequency Conversion Details ---")
print(f"Energy Gap:       {delta_Eps_cm} cm^-1")
print(f"Energy in Joules: {E_Joules:.4e} J")
print(f"Omega in rad/s:   {omega_rad_s:.4e} rad/s")
print(f"Omega in rad/fs:  {omega_rad_fs:.5f} rad/fs")
print(f"\n--- Pulse Details ---")
print(f"Max Interaction:  {interaction_max_cm:.2f} cm^-1")
print(f"Pulse Width:      {sigma_fs:.3f} fs")
print(f"The laser field t0_fs parameter is {t0_fs}")
print(f"The laser field omega_rad_fs is {omega_rad_fs}")
print(f"The laser field interaction_max_cm is {interaction_max_cm}")
print(f"The laser field sigma_fs is {sigma_fs}")

propagator = system_multibath(
  fname='laser_example3',
  init_rdo=np.outer(wfn, wfn.conj()),
  sys_ham=np.array([[delta_Eps_cm/2, 0.0], [0.0, -delta_Eps_cm/2]], dtype=np.complex128),#System Hamiltonian
  sys_ops=[np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)],#System operator (sigma_z/2)
  bath_correlations=[bath_simulation],
  td_f=laser_field_hadamard,#Laser field
  td_op = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),# light-matter coupling (sigma_x)
  dim=30,
  end_time=end_time,
  step_time=dt,
  save_checkpoint_to_file=True,
)

progress_bar = tqdm(propagator, total=ceil(end_time / dt))
for _t in (progress_bar):
  progress_bar.set_description(f'@{_t:.2f} fs')
