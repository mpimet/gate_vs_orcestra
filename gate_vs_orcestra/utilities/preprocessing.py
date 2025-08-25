import intake
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
    reanalysis["ERA5"] = (
        reanalysis["ERA5"]
        .sel(time=reanalysis["ERA5"].time.dt.month.isin([8, 9]))
        .where(extent)
    )

    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")

    icon = cat["ICON"]["ngc3028"](zoom=7, chunks="auto").to_dask()
    sea = icon.assign_coords(cell=icon.cell).where(icon.ocean_fraction_surface > 0.9)
    temperatures["ERA5"] = (
        reanalysis["ERA5"]["2t"]
        .where(~np.isnan(sea.ocean_fraction_surface))
        .groupby("time.year")
        .mean()
    )

    merra2tocean = reanalysis["MERRA2"]["t2m"].where(
        ~np.isnan(sea.ocean_fraction_surface)
    )
    temperatures["MERRA2"] = (
        merra2tocean.sel(time=merra2tocean.time.dt.month.isin([8, 9]))
        .where(extent)
        .dropna(dim="time", how="all")
        .groupby("time.year")
        .mean()
    )
    jra2tocean = reanalysis["JRA3Q"]["mean2t"].where(
        ~np.isnan(sea.ocean_fraction_surface)
    )
    temperatures["JRA3Q"] = (
        jra2tocean.sel(time=jra2tocean.time.dt.month.isin([8, 9]))
        .where(extent)
        .isel(time=slice(1, None))
        .groupby("time.year")
        .mean()
    )

    best = xr.open_dataset("/work/mh0066/m301046/Data/BEST/Global_TAVG_Gridded_1deg.nc")

    def get_useful_times(ds):
        years = ds.time.astype(int)
        months = np.ceil((best.time - best.time.astype(int)) * 12).astype(int)

        return ds.assign(
            time=[
                np.datetime64(f"{year}-{month:02d}-01")
                for year, month in zip(years.values, months.values)
            ]
        )

    best_data = get_useful_times(best).sel(
        latitude=slice(5, 12), longitude=slice(-27, -20)
    )

    temperatures["BEST"] = (
        best_data.sel(time=best_data.time.dt.month.isin([8, 9]))
        .groupby("time.year")
        .mean()
    )
    return temperatures
