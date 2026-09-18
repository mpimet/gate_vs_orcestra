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
        "dropsondes": "bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q",
        "halo": "bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",
        "meteor3": "bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44",
        "RA_NOAA_DC6": "bafybeib74rgi6pvz5pbwmcprniqdq6kfooz4i5jaegkexyswd6zf24xbjq",
        "RA_NCAR_Queen_Air": "bafybeibgrr5bvphu5sduwbllzm44psbonfwhxfcvh6igfwe6cc4rygcbim",
        "RA_NOAA_DC6_39_Charlie": "bafybeicrygz3p34b7xvtqwo2av5fbcy2w2zmosyjp2b46tkyjmuwbtawzy",
        "RA_NCAR_Elektra": "bafybeibrqy7wygkxel5ismmldaju4kmuqy6sy3zjxurdrjhejelwpizs6e",
        "RA_NCAR_Sabreliner": "bafybeid4sg7r64sasuljvprjzpjoeifff65xwrhbikwayyfelt74nfbyb4",
        "RA_NOAA_C130": "bafybeidjo25ixufydls2kydqtzdgo7qo5rt66ghv7i2nlp5mnb2sicf7ja",
        "RA_DC7_1S": "bafybeiezefd7zugmlhuupaj2ltzoxkbybelzlzxf4dsbdmr6eltvep7et4",
        "RA_UK_Hercules_XV208_100F": "bafybeiezrs252fciepksm2wby3vciwoz5ierdlxqijgrf3cer6yiqpc3wy",
        "RA_NASA_Convair_990": "bafybeieykyhcnns4f52z6lqg62jrp47imfjrfqu7zefcjyszrkh56qacqu",
        "dallas":["bafybeiac5k7cnzp56lqivzp2iy4hs2ztjpohtruoygbihkantdqrbdtavu",
                  "bafybeib3te2yt7pkodkllivlw5ox4aaifh5yi56fzsbt3lfbuc55zqyefm",
                  "bafybeiflavnoniy5rx7xvdxsu6bayrarlqdtjqd7uecfzemjz6pnnbsk3y",
                  "bafybeibgtnnadhmcdrvy4izdsgncmzqhl5bztcpeutb2i7o6ufg5dkhatq"],
        "faye":["bafybeihahfpqsu7bvg2t5r7go6do6aggb5g6pz4p7oc5uhq6frihqldcnm",
                "bafybeic5elbr47dj2mzxuwe6r7xojyt2fmu4kmjraqgrfzch34ygutwcvu"],
        "gilliss":["bafybeia6v5oamtynxnqq2hrtq27t4364xjs2kf2cbjoowza47uyw4gh32e",
                   "bafybeibersvcslljz6oxpgalen2vwtn72g6k363xoqiv2wg3myugu4e5ku",
                   "bafybeid7kfvlsfium55srzebk5jwjx3566nzoviafei5uuchacfsmm3hxy",
                   "bafybeiayrpmw6e7zd5w3vyzkjniihgqdjvz7ku33gckt5iz3wz3h3mrbwi"],
        "researcher":["bafybeidzpd5upvnylpqr247u6btltu3nueukwys7marxv2k24ntopj5iou",
                      "bafybeih6t6dfjazjidoeutnlpn3yt6phfysddno3izyc4wqrl7gpkx36eu",
                      "bafybeiclwv6miedtvnvml3w5y3oh7amr55pkpjndszp6m34kytyuclprxe",
                      "bafybeibf5gpvvzcesmhw6fu2ssxyiixc3lb7gxozp4kl7n5scl6lgzbqc4"],
        "meteor-gate":"bafybeia245wg2a3r3oet7qpuwiignzzsfg4t23w3w7pnqfgvoa4jip3pam",
        "planet":"bafybeifqht6q7hhaj3uwd5fpfwe7j624yzhobksigrvpj7oxuwljftabpq",
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
