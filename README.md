## GATE vs ORCESTRA

This repo contains the files to compare the measurements (mostly soundings) from GATE to ORCESTRA. 

The GATE campaign is desribed in Kuettner (1974) and on the [GATE-website](https://www.eol.ucar.edu/field_projects/gate). 
The ORCESTRA campaign is described on [this website](https://orcestra-campaign.org/intro.html), and ORCESTRA data can be found using the [Data Browser](https://browser.orcestra-campaign.org/#/?s=). 

Kuettner, J. P.: General description and central program of GATE, Bulletin of the
American Meteorological Society, 55, 526–530, 1974.

### Accessing the Data on IPFS

Most of the scripts in this repo use data that is puplicly available on [IPFS](https://docs.ipfs.tech/concepts/what-is-ipfs/). To obtain data from the IPFS network, you need access to an [IPFS Gateway](https://docs.ipfs.tech/concepts/ipfs-gateway).

We recommend that you use your own local gateway for faster access and local caching.
The simplest solution is to install [IPFS Desktop](https://docs.ipfs.tech/install/ipfs-desktop/), which provides a graphical user interface and runs a [Kubo daemon](https://docs.ipfs.tech/install/command-line/) in the background.

If you are **unable** to install software on your machine (e.g. work laptop), you can configure IPFS to use the public HTTPS gateway by setting:

```
export IPFS_GATEWAY=https://ipfs.io
```

To then access the data using Python, you will need to install the [`ipfsspec>=0.6.0`](http://pypi.org/project/ipfsspec/) package.
It is essential to install `ipfsspec` using pip, the version provided via `conda-forge` is outdated and **broken**.

### Accessing additional data

#### Berkeley Earth sea surface temperature

The subselected [Berkeley Earth Surface Temperature data](https://berkeleyearth.org) that we used can be found in the `data` folder. The original data is provided on the linked website. 

#### PIRATA buoy data
The [PIRATA buoy data](https://www.pmel.noaa.gov/gtmba/pmel-theme/atlantic-ocean-pirata) can be found in the data folder. 

### Running the scripts in this repo

The python environment for this repo was build using [uv](https://astral.sh/blog/uv). All dependencies  can be found in the `pyproject.toml`, in case you want to build your own environment.

The plots, tables, and calculations in the corresponding paperdraft can be found in the following scripts:
| Figure/Table   | short description | script |
| -------- | ------- | ------- | 
| Fig 1 | map with sonde position | `go_spatial.py`    |
| Tab 1 | sonde mean positions and times | `go_spatial.py` |
| Fig 2 | thermodynamic structure | `compare_soundings.py` |
| Fig 3 | $\Delta T$ |  `compare_soundings.py` |
| Fig 4 | relative humidity | `rh_2d_plot.py` | 
| Tab 2 | heights ($z_0, z_\text{ct}, z_\text{cp}$) | `compare_soundings.py` |
| Fig 5 | surface temperature  | `pirata_analysis_gate_domain.py` | 
| | 1D RCE konrad | `rce_simulation.py` |
| | PAMTRA calculations | `passive-microwave-halo.py` | 

## Re-creating data based on other packages

Although most of the (plotting) scripts can be run with the above-specified environment, there are a few special cases where plots require data or calculations that depend on packages not included in the environment.  The use of these is described below.

### Microwave temperature estimates with PAMTRA

To recreate the microwave temperature estimates we use PAMTRA with input data from the various sounding datasets through the python scripting provided in this repo.   Below we provide instructions for setting up PAMTRA on a Mac and on the DKRZ machine Levante (linux with intel).  This and the [PAMTRA documentation](https://pamtra.readthedocs.io/en/latest/installation.html) should provide guidance for implementing on other systems.

Using Levante (linux, intel)

```
module load git
module load netcdf-c
module load fftw
module load gcc
spack load /tpmfvwu # openblas

CC=gcc uv run pip install git+https://github.com/igmk/pamtra
``` 

Using a Mac (ARM)

```
brew install openblas pkgconf netcdf fftw gcc@14
export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
CC=gcc-14 uv run pip install git+https://github.com/igmk/pamtra
```
Other gcc versions can also be used, however, they apparently have to be smaller than 15. 

### RCE calculations with Konrad

To re-create the RCE calculations requires running Konrad. Konrad relies on the CliMT package, which unfortunately is no longer maintained. Consequently, installation with `uv` is not possible right away. However, the following workaround should install Konrad and CliMT to the virtual environment:

```
# Activate the `uv` environment directly
uv sync
source .venv/bin/activate

# Set environment variables for C anf Fortran compilers
export CC=gcc-12
export FC=gfortran-12

# Set the target architecture (different for Apple M1 [arm64])
[[ $(uname -p) == arm64 ]] && export TARGET=ARMV8 || export TARGET=HASWELL

# Install a trimmed down version of CliMT that ships RRTMG only
python3 -m pip install git+https://github.com/atmtools/climt@rrtmg-only

# Install konrad itself
python3 -m pip install konrad
```

It should then be possible to run `uv run gate_vs_orcestra/rce_simulation.py`.


### High Level radiosonde datasets with Pydropsonde

For consistency, the GATE Level 1 files as well as the RAPSODI Level 1 data are processed with [pydropsonde (version >= 0.5.1)](https://github.com/atmdrops/pydropsonde). To use this processing tool  entering: 
```
uv add pydropsonde
```
should be sufficient to run the `pydropsonde4gate.py` and the `reprocess_rapsodi.py` scripts to reproduce the data used in this repo.

If you would like to recreate the GATE Leve 1 data from the tar-archives.   The Fortran code for doing so is provided and described through a separate `README.md` in the `gate_processing` folder


### Radiant energy fluxes with RRTMG

Calculations were performed with EcRad which can be downloaded and installed from the its [github repo](https://github.com/ecmwf-ifs/ecrad), which also provides usage isntructions.  Trace gas concentrations used in the input are taken from the reanlaysis as distributed by the Copernicus Climate Change Services.
