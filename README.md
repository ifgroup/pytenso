# TENSO: Tensor Equations for Non-Markovian Structured Open systems

For Paper: X. Chen and I. Franco. Tree tensor network hierarchical equations of motion based on time-dependent variational principle for efficient open quantum dynamics in structured thermal environments. 
[arXiv:2505.00126](https://arxiv.org/pdf/2505.00126)

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
