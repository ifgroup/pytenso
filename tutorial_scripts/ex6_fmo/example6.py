# %% [markdown]
# # The Fenna–Matthews–Olson (FMO) complex
#
# The FMO complex is a water-soluble complex found in green sulfur bacteria. It plays a key role in photosynthesis by mediating energy transfer between light-absorbing components, and this process can be modeled as an open quantum system.
#
# The FMO complex is modeled as a network of sites (usually 7 or 8), each representing a bacteriochlorophyll molecule. The system Hamiltonian is written as:
#
# $$
# H_S=\sum_n\epsilon_n \ket{n}\bra{n}+ \sum_{n\ne m}J_{nm}\ket{n}\bra{m}
# $$
#
# where $\epsilon_n$ are site energies and $J_{nm}$ are electronic couplings between sites.
#
# For the system-bath interaction, each FMO site $n$ is coupled to a local bosonic bath representing protein and solvent motions:
#
# $$
# H_{SB}=\sum_n \ket{n}\bra{n}\otimes \sum_jg_{nj}(a_{nj}+a_{nj}^\dagger)
# $$

# %% [markdown]
# # Setup

# %%
from math import ceil
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath
from tenso.prototypes.bath import gen_bcf

# %% [markdown]
# # System Hamiltonian and Initial State

# %%
H = np.array([[200, -87.7,  5.5],
              [-87.7, 320, 30.8],
              [5.5,  30.8,    0]], dtype=np.complex128)

sys_ops = []
for i in range(3):
    op = np.zeros((3, 3))
    op[i, i] = 1.0
    sys_ops.append(op)

psi0 = np.zeros(3)
psi0[0] = 1.0
rho0 = np.outer(psi0, psi0.conj())

end_time = 1000
dt = 0.1

# %% [markdown]
# # Bath Correlation Functions
#
# Three BCFs are compared:
# - **Structured** (6-peak): full vibrational structure of the FMO protein environment
# - **3-Peak**: first three peaks of the structured BCF
# - **Drude**: single Drude–Lorentz oscillator (unstructured bath)

# %%
def make_bath(temperature):
    n_ltc = 3 if temperature <= 77 else 1
    bath_structured = gen_bcf(
        re_b=[26.24, 13.832, 101.479, 57.575, 25.764, 9.126],
        freq_b=[160, 247, 763, 1175, 1356, 1521],
        width_b=[133, 53, 76, 29, 29, 15],
        temperature=temperature,
        decomposition_method='Pade',
        n_ltc=n_ltc,
    )
    bath_three_peak = gen_bcf(
        re_b=[26.24, 13.832, 101.479],
        freq_b=[160, 247, 763],
        width_b=[133, 53, 76],
        temperature=temperature,
        decomposition_method='Pade',
        n_ltc=n_ltc,
    )
    bath_drude = gen_bcf(
        re_d=[35],
        width_d=[106.17674918],
        temperature=temperature,
        decomposition_method='Pade',
        n_ltc=n_ltc,
    )
    return {"structured": bath_structured, "three_peak": bath_three_peak, "drude": bath_drude}

# %% [markdown]
# # Simulations
#
# Each configuration runs the same 3-site FMO system with a different BCF and temperature.
# Output file names match those expected by `all_tenso_plots.ipynb`.

# %%
configs = [
    {"fname": "3_level_FMO_structured_300K", "temperature": 300, "bath_key": "structured"},
    {"fname": "3_level_FMO_structured_77_2",  "temperature":  77, "bath_key": "structured"},
    {"fname": "3_level_FMO_three_peak_300K",  "temperature": 300, "bath_key": "three_peak"},
    {"fname": "3_level_FMO_threepeak_77K",    "temperature":  77, "bath_key": "three_peak"},
    {"fname": "3_level_FMO_dl300_2",          "temperature": 300, "bath_key": "drude"},
    {"fname": "fmo_77K_dl",                   "temperature":  77, "bath_key": "drude"},
]

for cfg in configs:
    baths = make_bath(cfg["temperature"])
    bath = baths[cfg["bath_key"]]

    print(f"\n--- Running: {cfg['fname']} ({cfg['temperature']} K) ---")
    propagator = system_multibath(
        fname=cfg["fname"],
        init_rdo=rho0,
        sys_ham=H,
        sys_ops=sys_ops,
        bath_correlations=[bath] * 3,
        end_time=end_time,
        step_time=dt,
        ps_method="ps1",
        dim=20,
    )
    progress_bar = tqdm(propagator, total=ceil(end_time / dt))
    for _t in progress_bar:
        progress_bar.set_description(f'@{_t:.2f} fs')
