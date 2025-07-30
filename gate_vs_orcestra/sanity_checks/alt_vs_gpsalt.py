# %%
import fsspec
import xarray as xr
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
import data_utils as data


# %%
def fsglob(pattern):
    schema = pattern.split(":")[0]
    fs = fsspec.filesystem(schema)
    return fs.glob(pattern)


def fsls(path):
    schema = path.split(":")[0]
    fs = fsspec.filesystem(schema)
    return fs.ls(path, detail=False)


root = data.get_cids()["orcestra"]

l2_path = f"ipfs://{root}/HALO/dropsondes/Level_2"
l2_flights = fsls(l2_path)

# %%

sondelist = []
for flight in l2_flights[1:]:
    sondes = [f"ipfs://{path}" for path in fsls(f"ipfs://{flight}")]
    for sonde_file in sondes:
        sondelist.append(xr.open_dataset(sonde_file, engine="zarr"))


# %%
def remove_outlier(ds, var, perc=0.99):
    thres = ds[var].quantile(perc)
    return ds.where(np.abs(ds[var]) <= thres)


# %%
datalist = []
for sonde in sondelist:
    ds = (
        sonde.sortby("time")
        .reset_coords()
        .interpolate_na("time", limit=100)
        .dropna("time", how="all", subset=["p"])
        .swap_dims({"time": "p"})
        .sortby("p")
    )
    datalist.append(
        ds.groupby_bins("p", bins=np.linspace(10000, 101000, 1000))
        .mean()
        .rename({"p_bins": "p"})
    )

# %%
ds = xr.concat(datalist, dim="sonde")
ds = remove_outlier(ds, "gpsalt", perc=0.99)
ds = remove_outlier(ds, "alt", perc=0.99)
# %%


fig, ax = plt.subplots()
(ds.gpsalt - ds.alt).mean("sonde").sel(p=slice(20000, 100000)).rolling(
    p=20
).mean().plot(y="p", label="BEACH")
ax.invert_yaxis()
ax.set_xlabel("Difference between GPS altitude and altitude")
ax.set_ylabel("Pressure / Pa")
ax.legend()
ax.set_title("")
sns.despine(offset=10)
