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


def open_gate_ras(cids):
    ra_dict = {}
    for key, cid in cids.items():
        if key.startswith("RA_"):
            ra_dict[key] = xr.open_dataset(f"ipfs://{cid}", engine="zarr").sortby(
                "time"
            )
    return ra_dict


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
        "gate": "bafybeiccxo3qnccbqki2ccpv3bpjby72kkwg5sc3yv5rxu54l3rfnn3jgy",
        "orcestra": orcestra_main,
        "radiosondes": "QmcQRuqCgLRUVyCXjzmKfRVL34xxnxzL91PWTJSELrtQxa",
        "dropsondes": "ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q",
        "halo": "bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",
        "meteor3": "bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44",
        "RA_NOAA_DC6": "bafybeigvptjjii53y7mqoandkmr4nhkb5w5z6bkejctooi2zjde5zacm7e",
        "RA_NCAR_Queen_Air": "bafybeieziu2ncc5drxafjewt6zddcfulmyf5uqyjgh4olwuxwk3yszmfc4",
        "RA_NOAA_DC6_39_Charlie": "bafybeica7ulu6ezv4horvgpravnudjhceiajqdlxfvqxxlapb7pvyvgci4",
        "RA_NCAR_Elektra": "bafybeiflwpofkprgfiusp3ihh3g7grlwndin5khwic2xliohqj6ncxqlzm",
        "RA_NCAR_Sabreliner": "bafybeih2aw3gjhj5oa4igcfco6uvdo6z3qjiq7frg53hfcmlb7b4abfa3i",
        "RA_NOAA_C130": "bafybeiboomwk3tk6tifs5qok7ms2rdvfh6bjwmgvh7ejiiz4wg6xnly72u",
        "RA_DC7_1S": "bafybeiepibh5btgq2eumfgivhuttl5wmzswyd5ualnzhuh5jdbdgjiw5a4",
        "RA_UK_Hercules_XV208_100F": "bafybeibewu6outdtft5nx2jwgmnvhxttfq2kefhcnj4gzv6b6mmz3dhcqq",
        "RA_NASA_Convair_990": "bafybeiejizhiw5fnholonatocpwkjnaifageaxnu4y2xorvqq7v5fci264",
    }


# redundant datasets
#        'RA_UK_Hercules_XV208a_100F': 'bafybeiehni2sjyeoptljn4op4ee5yy4qo3ufwziw6gidzhnqakdpxb5c6a',
#        'RA_DC7_1M': 'bafybeig4av2q7yx4yvktc2uvo4goc7dsbocqjlqhuuc4c7a3l7iridn6h4',
#        'RA_UK_Hercules_XV208b_001F': 'bafybeigb3pcyb2qscsvtsof2avnm4aoptkwlfefgdkbbzvqagocwkgj53y',
#        'RA_UK_Hercules_XV208a_001F': 'bafybeieqvgmvu7x22reumug5wwtwcef4qoatu2f2kovbqwsicukjn22jgi',
#        'RA_UK_Hercules_XV208_001F': 'bafybeigiq6b2uws7niheqs7ak2yoed7pz3tyjym3nzarmys4lk47okv3ki',
#        'RA_UK_Hercules_XV208b_100F': 'bafybeicl2wn3pgl57l6bxwinrosgy2jdynaxvfdhtn446he7a772wjihue',


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


def open_meteor2(path="../data/rvs/meteor-gate.zarr"):
    ds = (
        xr.open_dataset(path, engine="zarr")
        .sel(time=slice("1974-08-10", "1974-09-30"))
        .squeeze()
    )
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
