import xarray as xr
import numpy as np
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp

rh_to_hus = mtf.relative_humidity_to_specific_humidity
hus_to_rh = mtf.specific_humidity_to_relative_humidity
es = svp.liq_wagner_pruss


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
        problem_platform = ds.platform_id == "METEOR"
        rh_slice = ds["rh"].sel(altitude=slice(z1, z2))
        dry_spike = (rh_slice.min(dim="altitude") < 0.2) & (
            rh_slice.max(dim="altitude") > 0.4
        )
        bad_sondes = ds.isel(sonde=(problem_platform & dry_spike))
        outside_error_region = (ds.altitude < z1) | (ds.altitude > z2)
        good_sondes = ~ds.sonde_id.isin(bad_sondes.sonde_id.values)
        ds = ds.assign({"rh": ds["rh"].where(good_sondes | outside_error_region)})

    return ds


def fill_gaps(ds, max_igap=1500, max_egap=300):
    """
    Interpolate large gaps, and fill remaining smaller gaps at the boundaries by extrapolating
    using nearest values.
    """

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
    ds = ds.assign(theta=mtf.theta(ds.ta, ds.p))
    ds = ds.assign(q=rh_to_hus(ds.rh, ds.p, ds.ta, es=es))
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
    ds = ds.assign(ta=mtf.theta2T(ds.theta, ds.p))
    ds = ds.assign(rh=hus_to_rh(ds.q, ds.p, ds.ta))
    ds = ds.drop_vars("lnp")

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

    cntrl = ds.isel(sonde=(ds.platform_id != "METEOR"))
    ds = ds.assign(
        {
            fld: in_bnds(ds[fld], *statistical_bnds(cntrl[fld], nstd, dim=dim))
            for fld in flds
        }
    )
    ds = ds.assign(theta=mtf.theta(ds.ta, ds.p))
    ds = ds.assign(q=rh_to_hus(ds.rh, ds.p, ds.ta, es=es))
    return ds


def coverage(xx, sf=100):
    """Return fraction of data that is not missing"""
    return sf * (xx.notnull().sum().values / np.prod(np.shape(xx.stack())))
