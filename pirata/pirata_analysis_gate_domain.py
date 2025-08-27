# %%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sn
import random
import numpy as np
from scipy.stats import linregress
import matplotlib.ticker as mticker
from utilities.settings_and_colors import colors
import utilities.preprocessing as pp


# %%
reanalysis_sfc = pp.preprocess_sfc_temperatures()

# %%
best_test = (
    reanalysis_sfc["BEST"].temperature
    + reanalysis_sfc["BEST"].climatology.sel(year=slice(1961, 1990)).mean()
    + 273.15
)
reanalysis_ds = xr.merge(
    [
        xr.merge(
            [
                reanalysis_sfc[name]
                .sel(year=slice(1974, None))
                .mean("cell")
                .rename(name)
                for name in ["JRA3Q", "ERA5", "MERRA2"]
            ]
        ),
        best_test.mean(["latitude", "longitude"])
        .sel(year=slice(1974, None))
        .rename("BEST"),
    ]
)
reanalysis_ds = reanalysis_ds.assign(
    mean=(
        ("year"),
        np.nanmean([reanalysis_ds[name] for name in reanalysis_ds.keys()], axis=0),
    ),
    std=(
        ("year"),
        np.nanstd([reanalysis_ds[name] for name in reanalysis_ds.keys()], axis=0),
    ),
)

# %%
path = "pirata/pirata_data/"

lon_west = 23
lats_north = ["4", "12"]
temp_res = "dy"

T0 = 273.15
# %% Loading and reformatting the data

var_help = {
    "AT": {"key": "t_air", "depth": -3.0},
}

vars = list(var_help.keys())

for varname in vars:
    q_varname = f"Q{varname}"
    var_help[q_varname] = {
        "key": f"q_{var_help[varname]['key']}",
        "depth": var_help[varname]["depth"],
    }
    s_varname = f"S{varname}"
    var_help[s_varname] = {
        "key": f"s_{var_help[varname]['key']}",
        "depth": var_help[varname]["depth"],
    }

ds_all = {}

for lat_north in lats_north:
    filename = f"{path}*{lat_north}n{lon_west}w_{temp_res}.cdf"
    ds = xr.open_mfdataset(filename)
    ds_all[lat_north] = ds

ds_formatted = xr.combine_by_coords(
    [ds_all[lat_north] for lat_north in lats_north],
    compat="broadcast_equals",
    combine_attrs="drop_conflicts",
)

for var in ds_formatted.data_vars:
    var_new = var.split("_")[0]
    ds_formatted = ds_formatted.rename({var: var_new})

    if var_new in var_help:
        selected_depth = var_help[var_new]["depth"]
        ds_formatted[var_new] = ds_formatted[var_new].sel(depth=selected_depth)
        ds_formatted[var_new].attrs["sel_depth_m"] = float(selected_depth)
        ds_formatted = ds_formatted.rename({var_new: var_help[var_new]["key"]})

    else:
        print(f"Need to add {var_new} to var_help.")

if "depth" in ds_formatted.coords:
    ds_formatted = ds_formatted.drop_vars("depth")

ds_formatted = ds_formatted.compute()
ds_formatted = ds_formatted.where(ds_formatted.q_t_air.isin([1, 2]), drop=True)
# %%
residuals_dict = {}

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

Tmax = {}
Tmin = {}
for sel_lat in [4, 12]:
    sel_t_air = ds_formatted["t_air"].sel(lon=337, lat=sel_lat)
    sel_t_air = sel_t_air.where(sel_t_air["time.month"].isin([8, 9]), drop=True)
    grouped = sel_t_air.groupby("time.year")

    sel_t_air_num = grouped.count().compute()
    sel_t_air_mean = grouped.mean().compute()

    # Only keep years with at least 60 data points
    sel_t_air_mean_filtered = (
        sel_t_air_mean.where(sel_t_air_num >= 60, drop=True).dropna(
            dim="year", how="any"
        )
        + T0
    )

    print(
        f"{sel_lat}°N: {len(sel_t_air_mean_filtered.values)} years of data from {int(sel_t_air_mean_filtered.year[0].values)} to {int(sel_t_air_mean_filtered.year[-1].values)}"
    )

    years = sel_t_air_mean_filtered["year"].values
    values = sel_t_air_mean_filtered.values
    Tmax[sel_lat] = values.max()
    Tmin[sel_lat] = values.min()

    slope, intercept, r_value, _, _ = linregress(years, values)
    fit_line = slope * years + intercept
    r_squared = r_value**2

    ax[0].scatter(
        years,
        values,
        s=20,
        marker="o",
        color=colors["pirata" + str(sel_lat)],
        label=f"{sel_lat}°N (fit: K/dec={slope * 10:.2f}, $R^2$={r_squared:.2f})",
    )
    ax[0].plot(
        years,
        fit_line,
        linestyle="--",
        color=colors["pirata" + str(sel_lat)],
        linewidth=1.5,
    )

    residuals_dict[sel_lat] = values - fit_line

year_slice = slice(1974, None)

slope, intercept, r_value, _, _ = linregress(
    reanalysis_ds.year.sel(year=year_slice).values,
    reanalysis_ds["BEST"].sel(year=year_slice).values,
)
reanalysis_ds["BEST"].plot(
    ax=ax[0],
    marker="x",
    linestyle="",
    color=colors["merra2"],
    label=f"BEST:(fit: K/dec={slope * 10:.2f}, $R^2$={r_squared:.2f})",
)
ax[0].plot(
    reanalysis_ds.year.sel(year=year_slice).values,
    slope * reanalysis_ds.year.sel(year=year_slice).values + intercept,
    linestyle="--",
    color=colors["merra2"],
    linewidth=1.5,
)
print(f"slope for {year_slice}: {slope}")
slope, intercept, r_value, _, _ = linregress(
    reanalysis_ds.year.values, reanalysis_ds["BEST"].values
)
print("slope whole period", slope)
rean_resid = reanalysis_ds["BEST"] - (slope * reanalysis_ds.year.values + intercept)
"""
slopes = {}
for name in ["JRA3Q", "ERA5", "MERRA2", "BEST"]:
    #rean_ds.plot(ax=ax[0], marker="x", linestyle="", color=colors[name.lower()])
    rean_ds = reanalysis_ds[name].dropna(dim="year", how="any")
    slope, intercept, r_value, _, _ = linregress(rean_ds.year.sel(year=year_slice).values, rean_ds.sel(year=year_slice).values)
    ax[0].plot(
        rean_ds.year.sel(year=year_slice).values,
        slope * rean_ds.year.sel(year=year_slice).values + intercept,
        linestyle="--",
        color=colors[name.lower()],
        linewidth=1.5,
        label=name
    )
    slopes[name] = slope
"""
ax[0].set_xticks(np.arange(1974, 2025, 6))
ax[0].spines["bottom"].set_bounds(1974, 2024)
ax[0].spines["left"].set_bounds(Tmin[4], Tmax[12])
ax[0].set_yticks(np.arange(299, 302, 1))
ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
ax[0].set_xlabel("year")
ax[0].set_ylabel(r"$T_{3\,\mathrm{m}}$ / K")
ax[0].legend()
ax[0].tick_params(axis="x", rotation=0)
sn.despine(offset=10, ax=ax[0])
# Residual histogram
residuals_all = np.concatenate([residuals_dict[4], residuals_dict[12]])
bins = np.arange(-0.8, 0.9, 0.1)
for sel_lat in [4, 12]:
    ax[1].hist(
        residuals_dict[sel_lat],
        bins=bins,
        alpha=0.4,
        density=True,
        label=f"{sel_lat}°N",
        color=colors["pirata" + str(sel_lat)],
    )
ax[1].hist(
    rean_resid,
    bins=bins,
    alpha=0.4,
    label="reanalysis",
    density=True,
    color=colors["merra2"],
)
ax[1].set_xlim(-0.82, 0.82)
ax[1].spines["bottom"].set_bounds(-0.8, 0.8)
ax[1].axvline(0, linestyle=":", color="k")
ax[1].set_ylabel("density")
ax[1].set_xlabel(r"residual $T_{3\,\mathrm{m}}$ / K")
sn.despine(ax=ax[1])

plt.tight_layout()
plt.savefig("pirata_linear_fit.pdf", bbox_inches="tight", dpi=300)

# %% Difference between two randomly selected years (without replacement)

plt.figure(figsize=(5, 5))
for i_sel, sel_lat in enumerate([4, 12]):
    sel_t_air = ds_formatted["t_air"].sel(lon=337, lat=sel_lat)
    sel_t_air = sel_t_air.where(sel_t_air["time.month"].isin([8, 9]), drop=True)
    grouped = sel_t_air.groupby("time.year")

    sel_t_air_num = grouped.count().compute()
    sel_t_air_mean = grouped.mean().compute()

    n_samples = int(1e3)
    sampled_pairs = [
        random.sample(list(sel_t_air_mean.dropna(dim="year", how="any").values), 2)
        for _ in range(n_samples)
    ]
    sampled_pair_diffs = np.array([np.abs(pair[0] - pair[1]) for pair in sampled_pairs])

    plt.hist(
        sampled_pair_diffs,
        bins=np.arange(0, 1.5, 0.01),
        density=True,
        cumulative=True,
        color=colors["pirata" + str(sel_lat)],
        histtype="step",
        label=f"{sel_lat}°N",
    )
    sn.despine()
    plt.xlim(xmin=0)

    print(len(np.where(sampled_pair_diffs < 1.5)[0]) / len(sampled_pair_diffs))

plt.legend()
plt.ylabel("CDF")
plt.xlabel(r"$|\Delta T_{3\,\mathrm{m}}|$ between two years (°C)")
# plt.savefig("pirata_pairwise_comparison.pdf", bbox_inches="tight", dpi=300)
# %%
