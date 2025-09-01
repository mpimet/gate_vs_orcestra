import os
import numpy as np
import xarray as xr
from matplotlib.path import Path
from utilities.settings_and_colors import gate_A, percusion_E
import moist_thermodynamics.functions as mt
import utilities.data_utils as data
import easygems.healpix as egh


def sel_sub_domain(
    ds, polygon, item_var="sonde", lon_var="launch_lon", lat_var="launch_lat"
):
    """
    select points from dataset that lie within the polygon
    """
    points = np.column_stack([ds[lon_var].values, ds[lat_var].values])
    inside = Path(polygon).contains_points(points)
    return ds.sel(**{item_var: inside})


def sel_gate_A(ds, **kwargs):
    """
    select points from dataset that lie within the gate_A polygon
    """
    return sel_sub_domain(ds, gate_A, **kwargs)


def sel_percusion_E(ds, **kwargs):
    """
    select points from dataset that lie within the percusion_E polygon
    """
    return sel_sub_domain(ds, percusion_E, **kwargs)


def interpolate_gaps(ds):
    akima_vars = ["u", "v"]
    linear_vars = ["theta", "q", "p"]

    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method="akima", max_gap=1500)
            for var in akima_vars
        }
    )
    ds = ds.assign(p=np.log(ds.p))
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method="linear", max_gap=1500)
            for var in linear_vars
        }
    )
    ds = ds.assign(p=np.exp(ds.p))
    ds = ds.assign(
        ta=mt.theta2T(ds.theta, ds.p),
    )
    ds = ds.assign(
        rh=mt.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta),
    )

    return ds


def extrapolate_sfc(ds):
    """
    Extrapolate surface values to the lowest level.
    This function assumes that the dataset has an altitude dimension.
    """
    constant_vars = ["u", "v", "theta", "q"]
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(
                dim="altitude", method="nearest", max_gap=300, fill_value="extrapolate"
            )
            for var in constant_vars
        }
    )
    ds = ds.assign(
        p=np.exp(
            np.log(ds.p).interpolate_na(
                dim="altitude", method="linear", max_gap=300, fill_value="extrapolate"
            )
        )
    )
    ds = ds.assign(
        ta=mt.theta2T(ds.theta, ds.p),
    )
    ds = ds.assign(
        rh=mt.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta),
    )
    return ds


def preprocess_sfc_temperatures(extent="orcestra_east"):
    if extent == "orcestra_east":
        extent = [-34, -20, 3.5, 13.5]
    if extent == "gate_ab":
        extent = [-27, -20, 5, 12]
    temperatures = {}
    reanalysis = data.open_reanalysis(chunks={}, zoom=7)

    extent = egh.get_extent_mask(reanalysis["ERA5"], extent=extent)
    temperatures["ERA5"] = (
        reanalysis["ERA5"]["2t"]
        .sel(time=reanalysis["ERA5"]["2t"].time.dt.month.isin([8, 9]))
        .where(extent)
        .groupby("time.year")
        .mean()
    )

    temperatures["MERRA2"] = (
        reanalysis["MERRA2"]["t2m"]
        .sel(time=reanalysis["MERRA2"]["t2m"].time.dt.month.isin([8, 9]))
        .where(extent)
        .dropna(dim="time", how="all")
        .groupby("time.year")
        .mean()
    )

    temperatures["JRA3Q"] = (
        reanalysis["JRA3Q"]["mean2t"]
        .sel(time=reanalysis["JRA3Q"]["mean2t"].time.dt.month.isin([8, 9]))
        .where(extent)
        .isel(time=slice(1, None))
        .groupby("time.year")
        .mean()
    )

    return temperatures


def get_tsrf_berkeley(
    fname="../data/best",
    src="/work/mh0066/m301046/Data/BEST/Global_TAVG_Gridded_1deg.nc",
    extent="gate_ab",
):
    if extent == "orcestra_east":
        lat_slice = slice(3.5, 13.5)
        lon_slice = slice(-34, -20)
    elif extent == "gate_ab":
        lat_slice = slice(5, 12)
        lon_slice = slice(-27, -20)
    T0 = 273.15
    fname = fname + f"_{extent}" + ".zarr"
    if not os.path.exists(fname):
        best = xr.open_dataset(src)

        def get_useful_times(ds):
            years = ds.time.astype(int)
            months = np.ceil((best.time - best.time.astype(int)) * 12).astype(int)

            return ds.assign(
                time=[
                    np.datetime64(f"{year}-{month:02d}-01")
                    for year, month in zip(years.values, months.values)
                ],
                months=(("time"), months.values),
                year=(("time"), years.values),
            )

        best_data = get_useful_times(best).sel(latitude=lat_slice, longitude=lon_slice)

        tsrf_anal = (
            (
                best_data.temperature.sel(time=best_data.time.dt.month.isin([8, 9]))
                + np.concatenate(
                    [best_data.climatology.sel(month_number=slice(7, 9))]
                    * int(350 / 2),
                    axis=0,
                )
                + T0
            )
            .groupby("time.year")
            .mean()
            .sel(year=slice(1974, None))
            * best_data.areal_weight
        ).sum(["latitude", "longitude"]) / best_data.areal_weight.sum()
        tsrf_anal.rename("temperature").to_zarr(fname)

    return xr.open_dataset(fname, engine="zarr").temperature


def get_pirata():
    path = "../data/"
    lon_west = 23
    lats_north = ["4", "12"]
    temp_res = "dy"

    var_help = {
        "AT": {"key": "t_air", "depth": -3.0},
    }

    vars = list(var_help.keys())

    for varname in vars:
        q_varname = f"Q{varname}"
        var_help[q_varname] = {
            "key": f"q_{var_help[varname]['key']}",
            "depth": var_help[varname]["depth"],
        }
        s_varname = f"S{varname}"
        var_help[s_varname] = {
            "key": f"s_{var_help[varname]['key']}",
            "depth": var_help[varname]["depth"],
        }

    ds_all = {}

    for lat_north in lats_north:
        filename = f"{path}*{lat_north}n{lon_west}w_{temp_res}.cdf"
        print(filename)
        ds = xr.open_mfdataset(filename)
        ds_all[lat_north] = ds

    pirata = xr.combine_by_coords(
        [ds_all[lat_north] for lat_north in lats_north],
        compat="broadcast_equals",
        combine_attrs="drop_conflicts",
    )

    for var in pirata.data_vars:
        var_new = var.split("_")[0]
        pirata = pirata.rename({var: var_new})

        if var_new in var_help:
            selected_depth = var_help[var_new]["depth"]
            pirata[var_new] = pirata[var_new].sel(depth=selected_depth)
            pirata[var_new].attrs["sel_depth_m"] = float(selected_depth)
            pirata = pirata.rename({var_new: var_help[var_new]["key"]})

        else:
            print(f"Need to add {var_new} to var_help.")

    if "depth" in pirata.coords:
        pirata = pirata.drop_vars("depth")

    pirata = pirata.compute()
    return pirata.where(pirata.q_t_air.isin([1, 2]), drop=True)
