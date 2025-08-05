# %%
# - import modules and define functions
import xarray as xr
import numpy as np
import utilities.data_utils as dus


def fill_gaps(ds, max_igap=1500, max_egap=300):
    """
    Interpolate large gaps, and fill remaining smaller gaps at the boundaries by extrapolating
    using nearest values.
    """
    from moist_thermodynamics import functions as mt
    from moist_thermodynamics import saturation_vapor_pressures as svp

    es = svp.liq_wagner_pruss

    if max_egap > max_igap:
        raise ValueError("extrapolation gaps must be smaller than interpolation gaps")

    ivars = {"u": "akima", "v": "akima", "rh": "akima", "ta": "linear", "lnp": "linear"}
    evars = {
        "u": "nearest",
        "v": "nearest",
        "theta": "nearest",
        "q": "nearest",
        "lnp": "linear",
    }

    ds = ds.assign(lnp=np.log(ds.p))
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method=method, max_gap=max_igap)
            for var, method in ivars.items()
        }
    )
    ds = ds.assign(p=np.exp(ds.lnp))
    ds = ds.assign(theta=mt.theta(ds.ta, ds.p))
    ds = ds.assign(
        q=mt.relative_humidity_to_specific_humidity(ds.rh, ds.p, ds.ta, es=es)
    )
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(
                dim="altitude",
                method=method,
                max_gap=max_egap,
                fill_value="extrapolate",
            )
            for var, method in evars.items()
        }
    )
    ds = ds.assign(p=np.exp(ds.lnp))
    ds = ds.assign(ta=mt.theta2T(ds.theta, ds.p))
    ds = ds.assign(rh=mt.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta))
    ds = ds.drop_vars("lnp")

    return ds


def in_bnds(da, lo, hi):
    return da.where((da > lo) & (da < hi))


def mask_unphysical(ds):
    """remove unphyscial values from the data set as determined by physical upper and lower bounds"""

    physical_bnds = {
        "ta": [185, 305],
        "rh": [0.0, 1.05],
        "u": [-50, 50],
        "v": [-20, 20],
        "p": [1000, 102500],
    }

    ds = ds.assign(
        {fld: in_bnds(ds[fld], *bnds) for fld, bnds in physical_bnds.items()}
    )

    # striping is evident in the meteor sondes at about two altitudes.  this removes it

    for z1 in [2050, 4900]:
        z2 = z1 + 300
        rh_slice = ds["rh"].sel(altitude=slice(z1, z2))
        problem_platform = ds.platform_id == "METEOR"
        dry_spike_in_region = (rh_slice.min(dim="altitude") < 0.2) & (
            rh_slice.max(dim="altitude") > 0.4
        )
        bad_sondes = ds.isel(sonde=(problem_platform & dry_spike_in_region))
        outside_error_region = (ds.altitude < z1) | (ds.altitude > z2)
        not_bad_sonde = ~ds.sonde_id.isin(bad_sondes.sonde_id.values)
        ds = ds.assign({"rh": ds["rh"].where(not_bad_sonde | outside_error_region)})

    return ds


def mask_outliers(ds, nstd=3, dim="sonde") -> xr.Dataset:
    """
    Identify outliers as those points outside of the specified range of control data and set to nan.
    """

    flds = ["ta", "rh", "u", "v", "p"]

    def statistical_bnds(da, nstd=3, dim="sonde"):
        lo = da.mean(dim=dim) - da.std(dim=dim, ddof=1) * nstd
        hi = da.mean(dim=dim) + da.std(dim=dim, ddof=1) * nstd
        return [lo, hi]

    cntrl = ds.isel(sonde=(ds.platform_id != "METEOR")).copy(deep=True)
    return ds.assign(
        {
            fld: in_bnds(ds[fld], *statistical_bnds(cntrl[fld], nstd, dim=dim))
            for fld in flds
        }
    )


def coverage(xx, sf=100):
    """Return fraction of data that is not missing"""
    return sf * (xx.notnull().sum().values / np.prod(np.shape(xx.stack())))


# %%
# - clean and fill for l3 data
#
gate_l2_cid = "QmatY7VJU1kVbwvthFqBxMvZ9dLphrT6X7GNHedTXXzgzj"
gate_l2 = dus.open_gate(gate_l2_cid)

gate_l3 = (
    gate_l2.pipe(mask_unphysical)
    .pipe(fill_gaps, max_igap=1000, max_egap=300)
    .pipe(mask_outliers)
    .sel(altitude=slice(-10, 25000))
    .assign_attrs(
        {
            "title": "GATE radiosonde dataset (Level 3)",
            "summary": "GATE ship radiosondes (filled and cleaned) subsetted to ORCESTRA time of year and lower 25 km",
            "creator_name": "Bjorn Stevens",
            "creator_email": "bjorn.stevens@mpimet.mpg.de",
            "license": "CC-BY-4.0",
            "processing_level": "3",
            "institution": "Max Planck Institute for Meteorologie",
            "source": "radiosonde",
            #            "history": f"Processed Gate level 2 radiosonde data with {gate_l2_cid} cid",
        }
    )
)
print(
    f"GATE level 3 ta data coverage:\n "
    f"initially = {coverage(gate_l3.ta): .8f}%\n "
    f"finally = {coverage(gate_l3.ta):.8f}%\n "
)

gate_l3.to_zarr("~/data/gate-l3.zarr")
