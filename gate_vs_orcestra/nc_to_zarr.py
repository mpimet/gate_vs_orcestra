# %%
import glob
import xarray as xr
import numpy as np

# IMPORTANT FOR DETERMINISTIC CIDs
import numcodecs

numcodecs.blosc.set_nthreads(1)


def process_gate(fdir):
    """
    convert GATE sounding data provied in netcdf files to a gridded product
    """
    import moist_thermodynamics.constants as constants
    import moist_thermodynamics.saturation_vapor_pressures as svp

    es = svp.liq_wagner_pruss
    P0 = constants.P0
    Rd = constants.Rd
    Rv = constants.Rv

    kappa = constants.Rd / constants.cpd
    #
    #
    dz = 10
    alt_bins = np.arange(0, 31000.1, dz) - dz / 2
    alt_bins[:-1]
    x = []
    for f in sorted(glob.glob(f"{fdir}/*.nc")):
        ds = xr.open_dataset(f)
        x.append(
            ds.assign_coords(alt=ds.alt.mean(dim="time"))
            .swap_dims({"level": "alt"})
            .groupby_bins("alt", bins=alt_bins, labels=alt_bins[:-1])
            .mean()
            .assign(
                {
                    "sonde_id": f,
                    "sonde": 1,
                    "platform_id": ds.platform,
                    "launch_lon": ds.launch_end_position.split()[0],
                    "launch_lat": ds.launch_end_position.split()[1],
                }
            )
        )

    sondes = (
        xr.concat(x, dim="time")
        .drop_vars(["va_err", "ua_err", "hus_err", "ta_err"])
        .rename(
            {
                "time": "launch_time",
                "alt_bins": "altitude",
                "plev": "p",
                "ua": "u",
                "va": "v",
                "hus": "q",
            }
        )
    )

    sondes = (
        sondes.assign(p=sondes.p * 100, q=sondes.q / 1000, ta=sondes.ta + 273.15)
        .assign(
            rh=sondes.q * Rv / (Rd + (Rv - Rd) * sondes.q) * sondes.p / es(sondes.ta),
            theta=sondes.ta * (P0 / sondes.p) ** kappa,
        )
        .set_coords(["launch_lat", "launch_lon"])
        .swap_dims({"launch_time": "sonde"})
        .drop_vars("sonde")
    )

    sondes.ta.attrs = {"units": "K"}
    sondes.p.attrs = {"units": "Pa"}
    sondes.q.attrs = {"units": "kg/kg"}
    sondes.theta.attrs = {"units": "K", "long_name": "potential temperature"}
    sondes.rh.attrs = {
        "units": "'",
        "long_name": "relative humidity with respect to liquid saturation",
    }

    del sondes.attrs["platform"]
    del sondes.attrs["launch_start_position"]
    del sondes.attrs["launch_end_position"]
    sondes.attrs["creator_name"] = "Bjorn Stevens"
    sondes.attrs["creator_email"] = "bjorn.stevens@mpimet.mpg.de"
    sondes.attrs["title"] = "GATE phase 2 and 3 ship-soundings"
    sondes.attrs["license"] = "CC-BY-4.0"
    sondes.attrs["summary"] = (
        "GATE ship-soundings mapped to a uniform 10 m grid using the groupby_bins('alt' ... ).mean() "
    )

    return sondes


# %%
# - create gate sounding data
#
src = "/Users/m219063/work/data/orcestra/gate/sondes/"
fname = "/Users/m219063/data/gate-radiosondes.zarr"
process_gate(src).to_zarr(fname)
