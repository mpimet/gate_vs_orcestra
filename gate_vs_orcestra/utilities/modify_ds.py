from xhistogram.xarray import histogram
import numpy as np
import moist_thermodynamics.functions as mtf


def get_hist_of_ta(
    da_t, da_var, var_binrange, ta_binrange=(240, 305), var_bin_num=100, ta_bin_num=200
):
    bins_ta = np.linspace(ta_binrange[0], ta_binrange[1], ta_bin_num)
    bins_var = np.linspace(var_binrange[0], var_binrange[1], var_bin_num)
    bin_name = da_var.name + "_bin"
    ta_name = da_t.name
    hist = histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude"]).compute()
    return (
        ((hist * hist[bin_name]).sum(dim=bin_name) / hist.sum(bin_name))
        .interpolate_na(dim=f"{ta_name}_bin")
        .rename({f"{ta_name}_bin": ta_name})
    )


def get_hist_of_ta_2d(
    da_t, da_var, var_binrange, ta_binrange=(240, 305), var_bin_num=100, ta_bin_num=200
):
    bins_ta = np.linspace(ta_binrange[0], ta_binrange[1], ta_bin_num)
    bins_var = np.linspace(var_binrange[0], var_binrange[1], var_bin_num)
    return histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=da_t.dims)


def get_surface(ds, es):
    P_sfc = ds.p.sel(altitude=slice(10, 50)).max(dim="altitude") + 100
    RH_sfc = ds.rh.sel(altitude=slice(0, 50)).max(dim="altitude")
    T_sfc = mtf.theta2T(
        ds.theta.sel(altitude=slice(None, 400)).mean(dim="altitude"), P_sfc
    )
    q_sfc = mtf.partial_pressure_to_specific_humidity(RH_sfc * es(T_sfc), P_sfc)
    return {
        "P": P_sfc.mean().values,
        "RH": RH_sfc.mean().values,
        "T": T_sfc.mean().values,
        "q": q_sfc.mean().values,
    }
