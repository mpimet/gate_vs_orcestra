# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

import easygems.healpix as egh
import intake
from scipy import signal
import cartopy.crs as ccrs
import cmocean
import cartopy.feature as cf

import utilities.data_utils as data

from utilities.settings_and_colors import colors

# %%# %%
reanalysis = data.open_reanalysis(chunks={}, zoom=7)
# %%

era = reanalysis["ERA5"]
extent = egh.get_extent_mask(era, extent=[-27, -20, 5, 12])
era = era.sel(time=era.time.dt.month.isin([8, 9])).where(extent)
# %%


def ocean(ds):
    return ds.ocean_fraction_surface == 1


cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")

icon = cat.ICON["ngc3028"](zoom=7, chunks="auto").to_dask()
sea = icon.assign_coords(cell=icon.cell).where(icon.ocean_fraction_surface > 0.9)


# %%
era2tocean = (
    era["2t"].where(~np.isnan(sea.ocean_fraction_surface)).groupby("time.year").mean()
)
merra2tocean = reanalysis["MERRA2"]["t2m"].where(~np.isnan(sea.ocean_fraction_surface))
merra2tocean = (
    merra2tocean.sel(time=merra2tocean.time.dt.month.isin([8, 9]))
    .where(extent)
    .dropna(dim="time", how="all")
    .groupby("time.year")
    .mean()
)
jra2tocean = reanalysis["JRA3Q"]["mean2t"].where(~np.isnan(sea.ocean_fraction_surface))
jra2tocean = (
    jra2tocean.sel(time=jra2tocean.time.dt.month.isin([8, 9]))
    .where(extent)
    .isel(time=slice(1, None))
    .groupby("time.year")
    .mean()
)

# %%
best = xr.open_dataset("/work/mh0066/m301046/Data/BEST/Global_TAVG_Gridded_1deg.nc")


# %%
def get_useful_times(ds):
    years = ds.time.astype(int)
    months = np.ceil((best.time - best.time.astype(int)) * 12).astype(int)

    return ds.assign(
        time=[
            np.datetime64(f"{year}-{month:02d}-01")
            for year, month in zip(years.values, months.values)
        ]
    )


best_data = get_useful_times(best).sel(latitude=slice(5, 12), longitude=slice(-27, -20))

# %%

best_summer = (
    best_data.sel(time=best_data.time.dt.month.isin([8, 9])).groupby("time.year").mean()
)
# %%
detrended_era = signal.detrend(
    era2tocean.mean("cell").sel(year=slice(1974, None)).values,
)
detrended_merra = signal.detrend(
    merra2tocean.mean("cell").values,
)
detrended_jra = signal.detrend(
    jra2tocean.mean("cell").sel(year=slice(1974, None)).values,
)
detrended_best = signal.detrend(
    best_summer.temperature.mean(dim=["latitude", "longitude"])
    .sel(year=slice(1974, None))
    .values
)
# %%
std = np.nanstd(
    np.concatenate([detrended_jra, detrended_merra, detrended_era, detrended_best])
)

# %%
plt.style.use("./gate.mplstyle")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    era2tocean.year.sel(year=slice(1974, None)),
    detrended_era,
    label="ERA5",
    color=colors["era"],
)

ax.plot(merra2tocean.year, detrended_merra, label="MERRA2", color=colors["merra"])

ax.plot(
    jra2tocean.sel(year=slice(1974, None)).year,
    detrended_jra,
    label="JRA3Q",
    color=colors["jra"],
)

ax.plot(
    best_summer.sel(year=slice(1974, None)).year,
    detrended_best,
    label="BEST",
    color=colors["best"],
)

ax.fill_between(
    np.array([1974, 2025]),
    [-std, -std],
    [std, std],
    color="gray",
    alpha=0.2,
)


ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
ax.legend()
ax.set_xlabel("Year")
ax.set_ylabel("August/September mean anomaly / K")
ax.set_title("Detrended 2m/sfc air temperature for lon: [-27, -20] and lat: [5, 12]")
sns.despine(offset=10)
xticks = ax.get_xticks()
xticks = np.append(xticks, [1974, 2024])
ax.set_xticks(xticks)
yticks = np.concatenate([ax.get_yticks(), [-std, std]])
ax.set_yticks(
    [
        tick
        for tick in np.where(np.abs(np.abs(yticks) - 0.2) > 1e-3, yticks, np.nan)
        if not np.isnan(tick)
    ],
    labels=[
        f"{tick:.2f}"
        for tick in np.where(np.abs(np.abs(yticks) - 0.2) > 1e-3, yticks, np.nan)
        if not np.isnan(tick)
    ],
)
ax.set_xlim(1973, 2025)
fig.savefig(
    "images/sfc_temperature_anomaly.pdf",
)

# %%

projection = ccrs.Robinson(central_longitude=10)
fig, ax = plt.subplots(
    figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
)
ax.set_extent([-67, -10, -2, 22], crs=ccrs.PlateCarree())

egh.healpix_show(jra2tocean.mean("year"), ax=ax, cmap=cmocean.cm.thermal)
ax.add_feature(cf.COASTLINE, linewidth=0.8)
ax.add_feature(cf.BORDERS, linewidth=0.4)
