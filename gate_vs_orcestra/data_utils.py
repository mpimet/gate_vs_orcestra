import intake
import hashlib
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram


def hash_xr_var(da):
    return np.array(
        [
            hashlib.sha256(str(entry).encode("ascii")).hexdigest()[:16]
            for entry in da.values
        ]
    )


def open_dropsondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    ds = (
        ds.rename(
            {
                "aircraft_latitude": "launch_lat",
                "aircraft_longitude": "launch_lon",
                "aircraft_msl_altitude": "launch_altitude",
            }
        )
        .reset_coords(["launch_altitude"])
        .swap_dims({"sonde": "sonde_id"})
    )
    try:
        return ds.swap_dims({"circle": "circle_id"})
    except ValueError:
        return ds


def open_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "sounding": "sonde_id",
                "flight_time": "bin_average_time",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "flight_lat", "flight_lon", "bin_average_time"])
        .swap_dims({"launch_time": "sonde_id"})
    )


def open_gate(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    ds = ds.assign_coords({"sonde_id": ("time", hash_xr_var(ds.time))})
    return (
        ds.rename(
            {
                "alt": "altitude",
                "lat_beg": "launch_lat",
                "lon_beg": "launch_lon",
                "ua": "u",
                "va": "v",
                "platforms": "platform_id",
                "time": "launch_time",
            }
        )
        .set_coords(["launch_lat", "launch_lon"])
        .swap_dims({"launch_time": "sonde_id"})
    )


def open_reanalysis(chunks=None, **kwargs):
    if chunks is None:
        chunks = {}
    cat = intake.open_catalog("http://data.nextgems-h2020.eu/catalog.yaml")
    return {
        "ERA5": cat.ERA5(chunks=chunks, **kwargs).to_dask(),
        "MERRA2": cat.MERRA2(chunks=chunks, **kwargs).to_dask(),
        "JRA3Q": cat.JRA3Q(chunks=chunks, **kwargs).to_dask(),
    }


def get_cids():
    orcestra_main = "QmPNVTb5fcN59XUi2dtUZknPx5HNnknBC2x4n7dtxuLdwi"
    return {
        "orcestra": orcestra_main,
        "gate": "QmeAFUdB3PZHRtCd441HjRGZPmEadtskXsnL34C9xigH3A",  # "QmU4TMq2mwuc5h1QgFrAo67EFw8YJoXSEEGXWEwU9vZXCF",
        "radiosondes": f"{orcestra_main}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr",
        "dropsondes": f"{orcestra_main}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    }


def get_gate_region(ds, lats=(5, 12), lons=(-27, -20)):
    """
    ascent_flag: 0 for descending, 1 for ascending
    """

    return ds.where(
        (lons[0] < ds.launch_lon)
        & (ds.launch_lon < lons[1])
        & (lats[0] < ds.launch_lat)
        & (ds.launch_lat < lats[1]),
        drop=True,
    )


def get_hist_of_ta(da_t, da_var, bins_var, bins_ta=np.linspace(240, 305, 200)):
    bin_name = da_var.name + "_bin"
    ta_name = da_t.name
    hist = histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude"]).compute()
    return (
        ((hist * hist[bin_name]).sum(dim=bin_name) / hist.sum(bin_name))
        .interpolate_na(dim=f"{ta_name}_bin")
        .rename({f"{ta_name}_bin": ta_name})
    )
