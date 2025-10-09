# %%
# - plot sst and air temperatures from ships
import glob
import xarray as xr
import utilities.preprocessing as pre
import utilities.data_utils as dus
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# - process or load ship data
reprocess_ships = False
ships = {
    "dallas": "DALLAS",
    "faye": "FAYE",
    "gilliss": "JAMES_M_GILLISS",
    "researcher": "RESEARCHER",
    "meteor-gate": "METEOR",
    "planet": "PLANET",
}
datasets = {}
for ship, path in ships.items():
    fname = f"../data/rvs/{ship}.zarr"
    if reprocess_ships:
        files = sorted(
            glob.glob(f"/Users/m219063/work/data/orcestra/GATE_v3/DSHIP/{path}/*.nc")
        )
        xs = []
        for file in files:
            xs.append(xr.open_dataset(file))
        ds = xr.concat(xs, dim="time").drop_duplicates(dim="time").sortby("time")
        ds.to_zarr(fname, mode="w")
    datasets[ship] = xr.open_dataset(fname, engine="zarr")

datasets["meteor"] = xr.open_dataset(
    "ipfs://bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44",
    engine="zarr",
)
# %%
# - calculate sst median values and Ts offsets
cids = dus.get_cids()
ships = {
    "gate": dus.open_meteor2(path="../data/rvs/meteor-gate.zarr").pipe(
        pre.sel_gate_A, item_var="time", lon_var="lon", lat_var="lat"
    ),
    "orcestra": dus.open_meteor3(cids["meteor3"]).pipe(
        pre.sel_gate_A, item_var="time", lon_var="lon", lat_var="lat"
    ),
}
for campaign, ds in ships.items():
    print(
        f"{campaign}: SST = {ds['sst'].quantile(0.5).values:.3f}K, Temp. Air = {ds['ta'].quantile(0.5).values:.3F}K"
    )

# %%
# - quick plots of time-series
ships["gate"].sst.plot.scatter(alpha=0.5)
ships["gate"].ta.plot.scatter(alpha=0.5)

sns.despine(offset=10)
plt.show()

# %%

ships["orcestra"].sst.plot.scatter(alpha=0.5)
ships["orcestra"].ta.plot.scatter(alpha=0.5)

sns.despine(offset=10)
plt.show()

# %%
# - plot histograms of gate vs orcestra sst
datasets = {
    #   "gate_gridded": {"data": gate_ships.sst, "color": "orangered"},
    "meteor-gate": {"data": ships["gate"].sst, "color": "fuchsia"},
    "meteor-orchestra": {"data": ships["orcestra"].sst, "color": "navy"},
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
# - plot histograms of gate vs orcestra sst
tristan_chord = {
    "GATE": {"data": ships["gate"].sst, "color": "navy"},
    "ORCESTRA": {"data": ships["orcestra"].sst, "color": "orangered"},
}

cw = 190 / 25.4  # A4 Column width with 1cm margins
fig, ax = plt.subplots(1, 1, figsize=(cw / 2, cw / 2.5))

for key, dx in tristan_chord.items():
    if dx["data"].mean() < 200:
        dx["data"] = dx["data"] + 273.15
    label = f"{key}"  # ({dx['data'].quantile(0.5).values:.2f}K)"
    dx["data"].plot.hist(
        ax=ax, bins=20, alpha=0.3333, density=True, color=dx["color"], label=label
    )
    print(
        f"{key:20s}: [{dx['data'].quantile(0.1).values:.2f}, {dx['data'].quantile(0.25).values:.2f}, {dx['data'].quantile(0.5).values:.2f}, {dx['data'].quantile(0.75).values:.2f}, {dx['data'].quantile(0.9).values:.2f}]"
    )

ax.set_xticks([300.15, 301.25])
ax.set_yticks([1, 2])
ax.set_xlabel("Temperature / K")
ax.set_ylabel("probability density")
sns.despine(offset=10)
plt.legend()
plt.savefig("plots/tristan.pdf", bbox_inches="tight")
plt.show()
