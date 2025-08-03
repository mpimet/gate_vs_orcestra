# %%
import glob
import xarray as xr
import numpy as np

# IMPORTANT FOR DETERMINISTIC CIDs
import numcodecs

import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.functions as mt

numcodecs.blosc.set_nthreads(1)


def process_gate(fdir):
    """
    convert GATE sounding data from netcdf files to a gridded product
    """

    es = svp.liq_wagner_pruss
    #
    #
    dz = 10
    alt_bin_centers = np.arange(0, 31000.0 + dz / 2, dz)
    x = []
    for f in sorted(glob.glob(f"{fdir}/*.nc")):
        ds = xr.open_dataset(f)
        x.append(
            ds.groupby_bins(
                ds.alt, bins=(alt_bin_centers - dz / 2), labels=alt_bin_centers[:-1]
            )
            .mean()
            .assign(
                sonde_id=(("sonde", [f], {"long_name": "sonde identifier"})),
                platform_id=(
                    ("sonde", [ds.platform], {"long_name": "platform identifier"})
                ),
                launch_lon=(
                    (
                        "sonde",
                        [np.float32(ds.launch_end_position.split()[0])],
                        {
                            "long_name": "longitude of drop end position",
                            "units": "degrees_east",
                        },
                    )
                ),
                launch_lat=(
                    (
                        "sonde",
                        [np.float32(ds.launch_end_position.split()[1])],
                        {
                            "long_name": "latitude of drop end position",
                            "units": "degrees_north",
                        },
                    )
                ),
                launch_time=(
                    (
                        "sonde",
                        ds.time.values,
                        {
                            "long_name": "launch time of the sounding",
                            "standard_name": "time",
                        },
                    )
                ),
            )
        )

    sondes = (
        xr.concat(x, dim="sonde")
        .drop_vars(["va_err", "ua_err", "hus_err", "ta_err"])
        .rename(
            {
                "alt_bins": "altitude",
                "plev": "p",
                "ua": "u",
                "va": "v",
                "hus": "q",
            }
        )
    )

    sondes = sondes.assign(
        p=(
            sondes.p.dims,
            (sondes.p * 100).values,
            {
                "standard_name": "air_pressure",
                "units": "Pa",
                "long_name": "atmospheric pressure",
            },
        ),
        q=(
            sondes.q.dims,
            (sondes.q / 1000).values,
            {
                "standard_name": "specific_humidity",
                "units": "kg/kg",
                "long_name": "specific humidity",
            },
        ),
        ta=(
            sondes.ta.dims,
            (sondes.ta + 273.15).values,
            {
                "standard_name": "air_temperature",
                "units": "K",
                "long_name": "air temperature",
            },
        ),
    )
    sondes = sondes.assign(
        rh=(
            sondes.q.dims,
            mt.specific_humidity_to_relative_humidity(
                sondes.q, sondes.p, sondes.ta, es=es
            ).values,
            {
                "standard_name": "relative_humidity",
                "units": "'",
                "long_name": "relative humidity with respect to liquid",
                "description": "Wagner-Pruss saturation vapor pressure",
            },
        ),
        theta=(
            sondes.ta.dims,
            mt.theta(sondes.ta, sondes.p).values,
            {
                "standard_name": "air_potential_temperature",
                "units": "K",
                "long_name": "dry potential temperature",
            },
        ),
    ).set_coords(["launch_lat", "launch_lon"])

    sondes.attrs = dict(
        creator_name="Bjorn Stevens",
        creator_email="bjorn.stevens@mpimet.mpg.de",
        title="GATE Level2 Sounding Data",
        license="CC-BY-4.0",
        summary=(
            "GATE ship-soundings mapped to a uniform 10 m grid using the groupby_bins('alt' ... ).mean() "
        ),
    )

    return sondes.sortby("launch_time")


# %%
# - create gate sounding data
#
src = "/Users/m219063/work/data/orcestra/gate/sondes/"
fname = "/Users/m219063/data/gate-l2.zarr"
process_gate(src).to_zarr(fname)
