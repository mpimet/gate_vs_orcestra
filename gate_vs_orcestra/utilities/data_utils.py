import intake
import hashlib
import numpy as np
import xarray as xr


def hash_xr_var(da):
    return np.array(
        [
            hashlib.sha256(str(entry).encode("ascii")).hexdigest()[:16]
            for entry in da.values
        ]
    )


def open_dropsondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return ds.rename(
        {
            "aircraft_latitude": "launch_lat",
            "aircraft_longitude": "launch_lon",
            "aircraft_msl_altitude": "launch_altitude",
            "lat": "latitude",
            "lon": "longitude",
        }
    ).reset_coords(["launch_altitude"])


def open_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "sounding": "sonde_id",
                "flight_time": "bin_average_time",
                "flight_lat": "latitude",
                "flight_lon": "longitude",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "latitude", "longitude", "bin_average_time", "sonde_id"])
        .drop_dims(["nv"])
        .swap_dims({"launch_time": "sonde"})
    )


def open_gate(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "lat_end": "launch_lat",
                "lon_end": "launch_lon",
                "ua": "u",
                "va": "v",
                "platforms": "platform_id",
                "time": "launch_time",
            }
        )
        .sel(launch_time=slice("1974-08-10", "1974-09-30"))
        .set_coords(["launch_lat", "launch_lon"])
        .swap_dims({"launch_time": "sonde"})
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


def summarize_platforms(gate: xr.Dataset):
    unique_platforms, counts = np.unique(gate.platform_id.values, return_counts=True)
    print(f"Platforms in dataset: {len(unique_platforms)}")
    for platform, n in zip(unique_platforms, counts):
        print(f"{platform:10s} : {n:5d} sondes")
