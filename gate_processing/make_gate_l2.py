# %%
import glob
import xarray as xr
import numpy as np

# IMPORTANT FOR DETERMINISTIC CIDs
import numcodecs

import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.functions as mt

numcodecs.blosc.set_nthreads(1)


def process_gate(fdir, alt_bin_centers):
    """
    convert GATE sounding data from netcdf files to a gridded product
    """

    es = svp.liq_wagner_pruss
    dz = np.diff(alt_bin_centers)[0]
    #
    #
    x = []
    for f in sorted(glob.glob(f"{fdir}/*.nc")):
        ds = xr.open_dataset(f)
        x.append(
            ds.groupby_bins(
                ds.alt, bins=(alt_bin_centers) - dz / 2, labels=alt_bin_centers[:-1]
            )
            .mean()
            .assign(
                {
                    "sonde_id": (
                        "sonde",
                        [f[f.rfind("/") + 1 :]],
                        {"long_name": "sonde identifier"},
                    ),
                    "platform_id": (
                        "sonde",
                        [ds.platform],
                        {"long_name": "platform identifier"},
                    ),
                    "launch_lon": (
                        "sonde",
                        [np.float32(ds.launch_end_position.split()[0])],
                        {
                            "long_name": "longitude of drop end position",
                            "units": "degrees_east",
                        },
                    ),
                    "launch_lat": (
                        "sonde",
                        [np.float32(ds.launch_end_position.split()[1])],
                        {
                            "long_name": "latitude of drop end position",
                            "units": "degrees_north",
                        },
                    ),
                    "launch_time": (
                        "sonde",
                        ds["time"].values,
                        {
                            "long_name": "launch time of the sounding",
                            "standard_name": "time",
                        },
                    ),
                },
            )
        )

    sondes = (
        xr.concat(x, dim="sonde")
        .drop_vars(["va_err", "ua_err", "hus_err", "ta_err"])
        .set_coords(["launch_lat", "launch_lon"])
        .sortby("launch_time")
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
    sondes = (
        sondes.assign(
            {
                "p": (
                    sondes.p.dims,
                    (sondes["p"] * 100).values,
                    {
                        "standard_name": "air_pressure",
                        "units": "Pa",
                    },
                ),
                "q": (
                    sondes.q.dims,
                    (sondes["q"] / 1000).values,
                    {
                        "standard_name": "specific_humidity",
                        "units": "kg/kg",
                    },
                ),
                "ta": (
                    sondes.ta.dims,
                    (sondes["ta"] + 273.15).values,
                    {
                        "standard_name": "air_temperature",
                        "units": "K",
                    },
                ),
            }
        )
        .assign(
            {
                "rh": (
                    sondes.q.dims,
                    mt.specific_humidity_to_relative_humidity(
                        sondes["q"], sondes["p"], sondes["ta"], es=es
                    ).values,
                    {
                        "standard_name": "relative_humidity",
                        "units": "1",
                        "description": "Wagner-Pruss saturation vapor pressure over liquid",
                    },
                ),
                "theta": (
                    sondes.ta.dims,
                    mt.theta(sondes["ta"], sondes["p"]).values,
                    {
                        "standard_name": "air_potential_temperature",
                        "units": "K",
                        "description": "calculated with dry air gas constants and 1000 hPa as standard pressure",
                    },
                ),
            }
        )
        .assign_attrs(
            {
                "title": "GATE radiosonde dataset (Level 2)",
                "summary": "GATE ship based soundings mapped to a uniform 10 m grid using the groupby_bins('alt' ... ).mean() ",
                "featureType": "profile",
                "creator_name": "Bjorn Stevens",
                "creator_email": "bjorn.stevens@mpimet.mpg.de",
                "license": "CC-BY-4.0",
                "processing_level": "2",
                "project": "orcestra",
                "institution": "Max Planck Institute for Meteorologie",
                "source": "radiosonde",
                "history": "Processed netCDF version of archived data provided by Rene Redler",
            }
        )
    )

    del sondes.attrs["launch_start_position"]
    del sondes.attrs["launch_end_position"]

    return sondes


# %%
# - create gate sounding data
#
src = "/Users/m219063/work/data/orcestra/gate/sondes/"
fname = "/Users/m219063/data/gate-l2.zarr"

altitude_bin_centers = np.arange(0, 3100) * 10
process_gate(src, altitude_bin_centers).to_zarr(fname)
#
# ipfs files ls -l
# ipfs pin rm -r /cid
# ipfs add --recursive --hidden --raw-leaves --chunker=size-1048576 --quieter
# ipfs files cp /ipfs/cid gate-l2.zarr

# %%
# gate = xr.open_zarr("ipfs://QmVkdfVm9ryi4Vn3eK6LrwC4tSLZtSkVTrUHqhLjWNv4CV")
gate = xr.open_zarr(fname)
gate.rh.max().values
# %%
