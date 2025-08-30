# %%
# plot sst and air temperatures from ships

import glob
import xarray as xr
import utilities.preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt

# %%
files = sorted(glob.glob(f"{'/Users/m219063/work/data/orcestra/gate/meteor'}/*.nc"))
files = (files[0], files[3], files[4], files[5], files[2])
meteor2 = (
    xr.open_mfdataset(files)
    .sel(time=slice("1974-08-10", "1974-09-30"))
    .pipe(pre.sel_gate_A, item_var="time", lon_var="longitude", lat_var="latitude")
    .squeeze()
)
meteor3 = (
    xr.open_dataset(
        "ipfs://bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44",
        engine="zarr",
    )
    .sel(time=slice("2024-08-10", "2024-09-30"))
    .pipe(pre.sel_gate_A, item_var="time", lon_var="lon", lat_var="lat")
    .reset_coords("lat")
    .reset_coords("lon")
)

files = sorted(glob.glob(f"{'/Users/m219063/work/data/orcestra/gate/ssts'}/*.nc"))
gate_ships = (
    xr.open_mfdataset(files)
    .sel(time=slice("1974-08-10", "1974-09-30"))
    .stack(points=("time", "latitude", "longitude"))
    .pipe(pre.sel_gate_A, item_var="points", lon_var="longitude", lat_var="latitude")
)

# %%
meteor2.temperature.plot.scatter(alpha=0.5)
meteor2.sst.plot.scatter(alpha=0.5)

sns.despine(offset=10)
plt.show()

# %%
meteor3_sst = (meteor3.sst_extern_port + meteor3.sst_extern_board) / 2.0
meteor3_sst.plot.scatter(s=20)
plt.show()

# %%
datasets = {
    "gate_gridded": {"data": gate_ships.sst, "color": "orangered"},
    "meteor-gate": {"data": meteor2.sst, "color": "fuchsia"},
    "meteor-orchestra": {"data": meteor3_sst, "color": "navy"},
}
for key, dx in datasets.items():
    if dx["data"].mean() < 200:
        dx["data"] = dx["data"] + 273.15
    label = f"{key} ({dx['data'].quantile(0.5).values:.2f}K)"
    dx["data"].plot.hist(
        bins=20, alpha=0.3333, density=True, color=dx["color"], label=label
    )
    plt.gca().axvline(dx["data"].quantile(0.5).values, color=dx["color"], ls="dotted")
    print(
        f"{key:20s}: [{dx['data'].quantile(0.1).values:.2f}, {dx['data'].quantile(0.25).values:.2f}, {dx['data'].quantile(0.5).values:.2f}, {dx['data'].quantile(0.75).values:.2f}, {dx['data'].quantile(0.9).values:.2f}]"
    )
sns.despine(offset=10)
plt.legend()
plt.show()

# %%
meteor3_tair = (meteor3.t_air_port + meteor3.t_air_board) / 2.0
print(meteor3_tair.quantile(0.5).values)
# %%
