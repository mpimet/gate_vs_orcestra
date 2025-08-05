# %%
import glob
import xarray as xr
import numpy as np

import utilities.data_utils as dus
import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.functions as mt

# IMPORTANT FOR DETERMINISTIC CIDs
import numcodecs

numcodecs.blosc.set_nthreads(1)

hus_to_rh = mt.specific_humidity_to_relative_humidity
attr_dict = dus.variable_attribute_dict


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
                            "standard_name": "longitude",
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
                            "standard_name": "latitude",
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
    ).sortby("launch_time")

    sondes = sondes.assign(
        p=(sondes.p.dims, (sondes.p * 100).values),
        q=(sondes.q.dims, (sondes.q / 1000).values),
        ta=(sondes.ta.dims, (sondes.ta + 273.15).values),
    )
    sondes = sondes.assign(
        rh=(
            sondes.q.dims,
            hus_to_rh(sondes.q, sondes.p, sondes.ta, es=es).values,
        ),
        theta=(sondes.ta.dims, mt.theta(sondes.ta, sondes.p).values),
        reference_pressure=xr.DataArray(
            100000.0,
            attrs={
                "units": "Pa",
                "standard_name": "reference_pressure",
                "description": "Used for calculation of potential temperature",
            },
        ),
    ).set_coords(["launch_lat", "launch_lon"])

    for var, attrs in attr_dict.items():
        sondes[var].attrs = attrs

    sondes = sondes.assign_attrs(
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

    del sondes.attrs["launch_start_position"]
    del sondes.attrs["launch_end_position"]
    del sondes.attrs["platform"]

    return sondes


# %%
# - create gate sounding data
#
src = "/Users/m219063/work/data/orcestra/gate/sondes/"
fname = "/Users/m219063/data/gate-l2.zarr"

altitude_bin_centers = np.arange(0, 3100) * 10
process_gate(src, altitude_bin_centers).to_zarr(fname)
