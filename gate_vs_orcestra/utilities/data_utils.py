import glob
import intake
import hashlib
import numpy as np
import xarray as xr
import fsspec

variable_attribute_dict = {
    "ta": {
        "standard_name": "air_temperature",
        "units": "K",
    },
    "p": {
        "standard_name": "air_pressure",
        "units": "Pa",
    },
    "q": {
        "standard_name": "specific_humidity",
        "units": "kg/kg",
    },
    "u": {
        "standard_name": "eastward_wind",
        "units": "m/s",
    },
    "v": {
        "standard_name": "northward_wind",
        "units": "m/s",
    },
    "rh": {
        "standard_name": "relative_humidity",
        "units": "1",
        "description": "Relative to Wagner-Pruss saturation vapor pressure over liquid",
    },
    "theta": {
        "standard_name": "air_potential_temperature",
        "units": "K",
        "description": "Use dry air gas constants and 1000 hPa as reference pressure",
    },
}


def hash_xr_var(da):
    return np.array(
        [
            hashlib.sha256(str(entry).encode("ascii")).hexdigest()[:16]
            for entry in da.values
        ]
    )


def open_dropsondes(cid, local=False):
    if local:
        ds = xr.open_dataset(cid, engine="zarr")
    else:
        ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr").reset_coords(
            ["launch_altitude"]
        )
    return ds


def open_radiosondes(cid, local=False):
    if local:
        ds = xr.open_dataset(cid, engine="zarr")
    else:
        ds = (
            xr.open_dataset(f"ipfs://{cid}", engine="zarr")
            # .rename(
            #    {
            #        "height": "altitude",
            #        "platform": "platform_id",
            #    }
            # )
            # .reset_coords(["p", "lat", "lon", "interpolated_time", "sonde_id"])
            # .swap_dims({"launch_time": "sonde"})
            .set_coords(["launch_lat", "launch_lon", "launch_time"])
        )
    return ds


def open_gate(cid, local=False):
    if local:
        ds = xr.open_dataset(cid, engine="zarr")
    else:
        ds = (
            xr.open_dataset(f"ipfs://{cid}", engine="zarr")
            .set_coords(["launch_lat", "launch_lon", "launch_time"])
            .swap_dims({"sonde": "launch_time"})
            .sel(launch_time=slice("1974-08-10", "1974-09-30"))
            .swap_dims({"launch_time": "sonde"})
        )
    return ds


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
    orcestra_main = "QmXkSUDo97PaDxsPzCPXJXwCFDLBMp7AVdPdV5CBQoagUN"
    return {
        "gate": "QmWZryTDTZu68MBzoRDQRcUJzKdCrP2C4VZfZw1sZWMJJc",
        "orcestra": orcestra_main,
        "radiosondes": "QmcQRuqCgLRUVyCXjzmKfRVL34xxnxzL91PWTJSELrtQxa",  # f"{orcestra_main}/products/Radiosondes/Level_2/RAPSODI_RS_ORCESTRA_level2.zarr",
        "dropsondes": f"{orcestra_main}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
        "halo": "bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",
        "meteor3": "bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44",
    }


def summarize_platforms(gate: xr.Dataset):
    unique_platforms, counts = np.unique(gate.platform_id.values, return_counts=True)
    print(f"Platforms in dataset: {len(unique_platforms)}")
    for platform, n in zip(unique_platforms, counts):
        print(f"{platform:10s} : {n:5d} sondes")


def open_halo(cid=get_cids()["halo"]):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return ds.rename_vars(
        {
            "IRS_LAT": "latitude",
            "IRS_LON": "longitude",
            "IRS_ALT": "altitude",
        }
    ).set_coords(({"latitude", "longitude", "altitude"}))


def open_meteor2(path):
    f = sorted(glob.glob(path))
    ds = (
        xr.open_mfdataset((f[0], f[3], f[4], f[5], f[2]))
        .sel(time=slice("1974-08-10", "1974-09-30"))
        .squeeze()
    )
    ds["sst"] = ds["sst"] + 273.15
    ds["sst"].attrs = {"long_name": "sea surface temperature", "units": "kelvin"}
    ds = ds.rename(
        {"temperature": "ta", "pressure": "p", "latitude": "lat", "longitude": "lon"}
    )
    ds["ta"] = ds["ta"] + 273.15
    ds["ta"].attrs = {"long_name": "air temperature", "units": "kelvin"}
    ds["p"] = ds["p"] * 100
    ds["p"].attrs = {"long_name": "air pressure", "units": "Pa"}
    return ds


def open_meteor3(cid):
    print(cid)
    ds = (
        xr.open_dataset(f"ipfs://{cid}", engine="zarr")
        .sel(time=slice("2024-08-10", "2024-09-30"))
        .squeeze()
    )
    ds["sst"] = (ds.sst_extern_port + ds.sst_extern_board) / 2.0
    ds["ta"] = (ds.t_air_port + ds.t_air_board) / 2.0
    ds["sst"].attrs = {"long_name": "sea surface temperature", "units": "kelvin"}
    ds["ta"].attrs = {"long_name": "air temperature", "units": "kelvin"}
    return ds


def fsglob(pattern):
    schema = pattern.split(":")[0]
    fs = fsspec.filesystem(schema)
    return fs.glob(pattern)


def fsls(path):
    schema = path.split(":")[0]
    fs = fsspec.filesystem(schema)
    return fs.ls(path, detail=False)
