# TENSO: Tensor Equations for Non-Markovian Structured Open systems

This repository contains the code for the paper: 
- X. Chen and I. Franco. [Tree tensor network hierarchical equations of motion based on time-dependent variational principle for efficient open quantum dynamics in structured thermal environments](https://doi.org/10.1063/5.0278591), *The Journal of Chemical Physics* **163**, 104109 (2025).

A tutorial is available for TENSO:
- J. C. Rodriguez Betancourt, M. C. Anderson, L. Niu, X. Chen, and I. Franco. [TENSO: Software Package for Numerically Exact Open Quantum Dynamics Based on Efficient Tree Tensor Network Decomposition of the Hierarchical Equations of Motion](https://arxiv.org/abs/2603.17711), (2026).
Input files accompanying this tutorial are found in the directory tutorial\_scripts.

If you find this repository useful, please consider citing our work.

```bibtex
@article{Chen2025,
   author = {Xinxian Chen and Ignacio Franco},
   doi = {10.1063/5.0278591},
   issue = {10},
   journal = {The Journal of Chemical Physics},
   month = {9},
   pages = {104109},
   title = {Tree tensor network hierarchical equations of motion based on time-dependent variational principle for efficient open quantum dynamics in structured thermal environments},
   volume = {163},
   url = {https://pubs.aip.org/jcp/article/163/10/104109/3361762/Tree-tensor-network-hierarchical-equations-of},
   year = {2025}
}
```


## Overview

This repository contains the code for the TENSO algorithm, which is a tensor network based method for generating and integration the master equations for open quantum dynamics in structured thermal environments. 
The code is written in Python and uses PyTorch for tensor operations. It is designed to be efficient and scalable, allowing for the simulation of large open quantum systems using different tree tensor network topologies.
Details of the algorithm can be found in the paper. Detailed documentation is under preparation.

## Quick setup

- Development setup: 

    0. Create a python virtural environment with python vesion >= 3.10.

    1. Prepare dependencies: `numpy`, `scipy`, `pytorch`, `torchdiffeq`, `tqdm`

    2. Install `tenso` in develop mode using `pip`:

            python -m pip install -e .

    3. For testing, consider `jupyter-lab`, `matplotlib`, etc.

