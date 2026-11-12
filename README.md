# cosmic-voids-research
Python tools and numerical methods for cosmic void analysis and cosmological simulations.

# Cosmic Voids and Dark Matter

Computational tools and analysis notebooks developed as part of my
M.Sc. Astronomy research at IIT Indore.

## Research Objective

Investigating how the structure and statistics of cosmic voids
can be used to probe the nature of dark matter, particularly
differences between Cold Dark Matter (CDM) and Warm Dark Matter
(WDM) models.

## Methods and Tools

- Python, NumPy, SciPy, Matplotlib, scikit-image
- FFT, Watershed Algorithm,etc


## Data Availability

Simulation datasets are not included in this repository. Users must obtain any required data themselves or simulate it using MP-GADGET, etc. I am currently simulating my own CDM and WDM fields (using a transfer function) with MP-GADGET.

## What each code does

cic.py---Cloud in Cell density assignment
vogel_pk.py---uses transfer function from vogel and abazajian 2023, PRD to calcualte initial power spectrums for wdm masses 1,2,3 and 5 keV. Modify the masses line to get the same for other masses. This is spin 1/2 thermal wdm only. For spin 3/2, see  	arXiv:2210.10753 

Research code under active development. Methods and results
are documented as the analysis progresses.
