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
