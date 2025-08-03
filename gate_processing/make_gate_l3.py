# %%
# - import modules and define functions
import xarray as xr
import numpy as np
import utilities.data_utils as dus

from moist_thermodynamics import functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp
from moist_thermodynamics import constants


es = svp.liq_wagner_pruss
P0 = constants.P0
Rd = constants.Rd
Rv = constants.Rv

kappa = constants.Rd / constants.cpd


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


def mask_unphysical(ds, max_alt=22000):
    """remove unphyscial values from the data set as determined by physical upper and lower bounds"""

    physical_bnds = {
        "ta": [185, 305],
        "rh": [0.0, 1.05],
        "u": [-50, 50],
        "v": [-20, 20],
        "p": [1000, 102500],
    }
    x = ds.sel(altitude=slice(-10, max_alt))
    for fld, val in physical_bnds.items():
        x[fld] = x[fld].where((x[fld] > val[0]) & (x[fld] < val[1]))

    # striping is evident in the meteor sondes at about two altitudes.  this removes it

    for z1 in [2050, 4900]:
        mask1 = x["rh"].sel(altitude=slice(z1, z1 + 300)).min(axis=1) < 0.2
        mask2 = x["rh"].sel(altitude=slice(z1, z1 + 300)).max(axis=1) > 0.4
        mask3 = x["platform_id"] == "Meteor"
        bad_sondes = x.isel(sonde=(mask1 & mask2 & mask3))
        mask1 = x.altitude < z1
        mask2 = x.altitude > z1 + 300
        x["rh"] = x["rh"].where(
            ~ds.sonde_id.isin(bad_sondes.sonde_id.values) & (mask1 | mask2)
        )

    return x


def mask_outliers(ds, cntrl, nstd, dim="sonde") -> xr.Dataset:
    """
    Identify outliers as those points outside of the specified range of control data and set to nan.
    """
    for fld in ["ta", "rh", "u", "v", "p"]:
        x_sig = cntrl[fld].std(dim=dim, ddof=1)
        x_bar = cntrl[fld].mean(dim=dim)
        condition = (ds[fld] > x_bar - nstd * x_sig) & (ds[fld] < x_bar + nstd * x_sig)
        ds[fld] = ds[fld].where(condition)

    return ds


def coverage(xx, sf=100):
    return sf * (xx.notnull().sum().values / np.prod(np.shape(xx.stack())))


# %%
# - clean and interpolate for l3 data
#
gate = dus.open_gate("Qmcc6oJjJ9gjw1WB3Kuv3v3GdhwsM7u37fj7o6CLkh4mSH")

g0 = mask_unphysical(gate).reset_coords(["launch_time", "launch_lat", "launch_lon"])
g1 = fill_gaps(g0, max_igap=1000, max_egap=300)
gc = g1.isel(sonde=(g1.platform_id != "METEOR").compute())
gate_clean = g1.pipe(mask_outliers, gc, nstd=3)
print(
    f"GATE ta data coverage:\n "
    f"initially = {coverage(gate.ta): .8f}%\n "
    f"plausible = {coverage(g0.ta): .8f}%\n "
    f"after filling & cleaning = {coverage(g1.ta):.8f}%\n "
)

gate_clean.attrs = dict(
    creator_name="Bjorn Stevens",
    creator_email="bjorn.stevens@mpimet.mpg.de",
    title="GATE Level3 Sounding Data",
    license="CC-BY-4.0",
    summary=("GATE ship-soundings filled and clean during ORCESTRA period "),
)

gate_clean.to_zarr("~/data/gate-l3.zarr")

# %%
