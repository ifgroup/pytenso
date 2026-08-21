# ⚛️ TENSO: Tensor Equations for Non-Markovian Structured Open Systems

[![Documentation](https://img.shields.io/badge/docs-website-blue.svg)](https://ifgroup.github.io/pytenso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**TENSO** is a tensor network-based method and software package for generating and integrating master equations for open quantum dynamics in structured thermal environments. 

The code is written in Python and leverages **PyTorch** for efficient, hardware-agnostic tensor operations (running seamlessly on CPUs and GPUs). It is designed to be highly scalable, allowing for the simulation of large open quantum systems by utilizing polynomial-scaling Tree Tensor Network (TTN) topologies instead of exponentially scaling approaches.

---

## 🚀 Quick Setup

### Development Installation

1. **Create a virtual environment** (Python >= 3.10 is required).
2. **Ensure core dependencies are available**: `numpy`, `scipy`, `pytorch`, `torchdiffeq`, `tqdm`.
3. **Install TENSO** in development mode using `pip`:
   ```bash
   python -m pip install -e .
   ```
4. *(Optional)* For tutorials and testing, we recommend installing `jupyter-lab` and `matplotlib`.

---

## 📚 Documentation & Tutorials

For a detailed guide, API reference, and interactive examples, please visit our official documentation website:

🌐 **[TENSO Official Documentation](https://ifgroup.github.io/pytenso)**

Input files accompanying the TENSO software tutorial can be found in the directory `tutorial_scripts/` within this repository.

---

## 🖋️ How to Cite

If you use TENSO in your research, please consider citing our work:

### 1. The TENSO Software Package
*Describes the software implementation and usage tutorials.*

> Rodriguez-Betancourt, J. C., Anderson, M. C., Niu, L., Chen, X., & Franco, I. (2026). TENSO: Software Package for Numerically Exact Open Quantum Dynamics Based on Efficient Tree Tensor Network Decomposition of the Hierarchical Equations of Motion. *Journal of Chemical Theory and Computation*, **22**(14), 7048-7069. [DOI: 10.1021/acs.jctc.6c00525](https://doi.org/10.1021/acs.jctc.6c00525)

<details>
<summary><b>Show BibTeX</b></summary>

```bibtex
@article{rodriguez2026tenso,
  author  = {Rodriguez-Betancourt, Juan C. and Anderson, Michelle C. and Niu, Luchang and Chen, Xinxian and Franco, Ignacio},
  title   = {{TENSO}: Software Package for Numerically Exact Open Quantum Dynamics Based on Efficient Tree Tensor Network Decomposition of the Hierarchical Equations of Motion},
  journal = {J. Chem. Theory Comput.},
  volume  = {22},
  number  = {14},
  pages   = {7048-7069},
  year    = {2026},
  doi     = {10.1021/acs.jctc.6c00525}
}
```
</details>

### 2. TTN-HEOM theory and algorithms
*Introduces the underlying TTN-HEOM method and time-dependent variational principle.*

> Chen, X. & Franco, I. (2025). Tree tensor network hierarchical equations of motion based on time-dependent variational principle for efficient open quantum dynamics in structured thermal environments. *The Journal of Chemical Physics* **163**, 104109. [DOI: 10.1063/5.0278591](https://doi.org/10.1063/5.0278591)

<details>
<summary><b>Show BibTeX</b></summary>

```bibtex
@article{Chen2025,
   author = {Xinxian Chen and Ignacio Franco},
   title = {Tree tensor network hierarchical equations of motion based on time-dependent variational principle for efficient open quantum dynamics in structured thermal environments},
   journal = {The Journal of Chemical Physics},
   volume = {163},
   issue = {10},
   pages = {104109},
   year = {2025},
   doi = {10.1063/5.0278591}
}
```
</details>

### 3. Theoretical Foundation (Bexcitonics)
*Develops the bexcitonic quasiparticle picture that underlies the HEOM generalization.*

> Chen, X. & Franco, I. (2024). Bexcitonics: Quasiparticle approach to open quantum dynamics. *The Journal of Chemical Physics*, **160**(20), 204116. [DOI: 10.1063/5.0198567](https://doi.org/10.1063/5.0198567)

<details>
<summary><b>Show BibTeX</b></summary>

```bibtex
@article{chen2024bexcitonics,
    author = {Chen, Xinxian and Franco, Ignacio},
    title = {Bexcitonics: Quasiparticle approach to open quantum dynamics},
    journal = {The Journal of Chemical Physics},
    volume = {160},
    number = {20},
    pages = {204116},
    year = {2024},
    doi = {10.1063/5.0198567}
}
```
</details>
