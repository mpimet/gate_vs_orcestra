import xarray as xr
import numpy as np
from matplotlib.path import Path
from utilities.settings_and_colors import gate_A, percusion_E


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
