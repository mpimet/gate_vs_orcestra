## GATE vs ORCESTRA

This repo contains the files to compare the measurements (mostly soundings) from GATE to ORCESTRA. 

The GATE campaign is desribed in Kuettner (1974) and on the [GATE-website](https://www.eol.ucar.edu/field_projects/gate). 
The ORCESTRA campaign is described on [this website](https://orcestra-campaign.org/intro.html), and ORCESTRA data can be found using the [Data Browser](https://browser.orcestra-campaign.org/#/?s=). 

Kuettner, J. P.: General description and central program of GATE, Bulletin of the
American Meteorological Society, 55, 526–530, 1974.

### Using PAMTRA for the sondes

This package is build using uv. Using the `uv.lock` is sufficient to run most of the python plotting scripts. However, to use the pamtra model, it has to be installed separately. On levante, the following steps suffice to do that, if run from within the repo.

```
module load git
module load netcdf-c
module load fftw
module load gcc
spack load /tpmfvwu # openblas

CC=gcc uv run pip install git+https://github.com/igmk/pamtra
``` 

On a MAC, 
```
brew install openblas pkgconf netcdf fftw gcc@14
export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
CC=gcc-14 uv run pip install git+https://github.com/igmk/pamtra
```
Other gcc versions can also be used, however, they apparently have to be smaller than 15. 

### Running konrad

Konrad relies on the CliMT package, which unfortunately is no longer maintained.
Consequently, installation with `uv` is not possible right away.
However, the following workaround should install Konrad and CliMT to the virtual environment:

```
# Activate the `uv` environment directly#
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
