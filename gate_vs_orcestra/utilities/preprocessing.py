import numpy as np
from matplotlib.path import Path
from utilities.settings_and_colors import gate_A, percusion_E
import moist_thermodynamics.constants as constants
import moist_thermodynamics.saturation_vapor_pressures as svp


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


def theta_to_t(theta, p, qv=0, ql=0, qi=0):
    """
    Convert potential temperature to temperature.
    """
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    cl = constants.liquid_water_specific_heat
    ci = constants.frozen_water_specific_heat
    P0 = constants.P0
    qd = 1.0 - qv - ql - qi
    kappa = (qd * Rd + qv * Rv) / (qd * cpd + qv * cpv + ql * cl + qi * ci)

    return theta / ((P0 / p) ** kappa)


def q_to_rh(q, p, T, es=svp.liq_wagner_pruss):
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    x = es(T) * Rd / Rv / (p - es(T))
    return q * (1 + x) / x


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
    ds = ds.assign(ta=theta_to_t(ds.theta, ds.p), rh=q_to_rh(ds.q, ds.p, ds.ta))

    return ds


def extrapolate_sfc(ds):
    """
    Extrapolate surface values to the lowest level.
    This function assumes that the dataset has an altitude dimension.
    """
    constant_vars = ["u", "v", "theta", "q"]
    ds = ds.assign(
        **{var: ds[var].bfill("altitude", limit=30) for var in constant_vars}
    )
    ds = ds.assign(
        p=np.exp(
            np.log(ds.p).interpolate_na(
                dim="altitude", method="linear", max_gap=300, fill_value="extrapolate"
            )
        )
    )
    ds = ds.assign(ta=theta_to_t(ds.theta, ds.p), rh=q_to_rh(ds.q, ds.p, ds.ta))
    return ds
