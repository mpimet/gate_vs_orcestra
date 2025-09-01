# %%
import matplotlib.pyplot as plt
import seaborn as sn
import random
import numpy as np
from scipy.stats import linregress
import matplotlib.ticker as mticker
from utilities.settings_and_colors import colors
import utilities.preprocessing as pp

# %%
# - Loading and reformatting the data

tsrf_anal = pp.get_tsrf_berkeley(
    src="/work/mh0066/m301046/Data/BEST/Global_TAVG_Gridded_1deg.nc"
)

# %%
pirata = pp.get_pirata()

# %%
# - Paper Figure
T0 = 273.15
residuals_dict = {}

cw = 190 / 25.4  # A4 Column width with 1cm margins
print(cw)
fig, ax = plt.subplots(1, 2, figsize=(cw, cw / 2))

Tmax = {}
Tmin = {}
for sel_lat in [4, 12]:
    sel_t_air = pirata["t_air"].sel(lon=337, lat=sel_lat)
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
        label=f"{sel_lat}°N (K/dec={slope * 10:.2f}, $R^2$={r_squared:.2f})",
    )

    residuals_dict[sel_lat] = values - fit_line

year_slice = slice(1974, None)

slope, intercept, r_value, _, _ = linregress(
    tsrf_anal.year.sel(year=year_slice).values,
    tsrf_anal.sel(year=year_slice).values,
)
tsrf_anal.plot(
    ax=ax[0],
    marker="x",
    linestyle="",
    color=colors["merra2"],
    label=f"Berkeley:(K/dec={slope * 10:.2f}, $R^2$={r_squared:.2f})",
)
ax[0].plot(
    tsrf_anal.year.sel(year=year_slice).values,
    slope * tsrf_anal.year.sel(year=year_slice).values + intercept,
    linestyle="--",
    color=colors["merra2"],
    linewidth=1.5,
)

year_slice = slice(2006, None)
slope, intercept, r_value, _, _ = linregress(
    tsrf_anal.year.sel(year=year_slice).values,
    tsrf_anal.sel(year=year_slice).values,
)

print(f"slope for {year_slice}: {slope}")
slope, intercept, r_value, _, _ = linregress(tsrf_anal.year.values, tsrf_anal.values)
print("slope whole period", slope)
rean_resid = tsrf_anal - (slope * tsrf_anal.year.values + intercept)


offset = 300.26 - 301.25
meteor2 = np.asarray([299.85, 300.05, 300.15, 300.35, 300.55]) + offset
meteor3 = np.asarray([300.75, 300.95, 301.25, 301.35, 301.55]) + offset

ax[0].plot(
    [1974, 1974],
    [meteor2[1], meteor2[3]],
    lw=2.5,
    c=colors["gate"],
    label=f"Meteor {meteor2[2]:.2f} K",
)

ax[0].plot([1974, 1974], [meteor2[0], meteor2[4]], lw=0.5, c=colors["gate"])

ax[0].plot(
    [2024, 2024],
    [meteor3[1], meteor3[3]],
    lw=2.5,
    c=colors["orcestra"],
    label=f"Meteor {meteor3[2]:.2f} K",
)

ax[0].plot([2024, 2024], [meteor3[0], meteor3[4]], lw=0.5, c=colors["orcestra"])

ax[0].set_xticks(np.arange(1974, 2025, 10))
ax[0].spines["bottom"].set_bounds(1974, 2024)
ax[0].spines["left"].set_bounds(Tmin[4], Tmax[12])
ax[0].set_yticks(np.arange(299, 302, 1))
ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
ax[0].set_xlabel("year")
ax[0].set_ylabel(r"$T_{3\,\mathrm{m}}$ / K")
ax[0].tick_params(axis="x", rotation=0)
ax[0].plot([1973.8, 1974.1], [meteor2[2], meteor2[2]], lw=3, c="w")
ax[0].plot([2023.9, 2024.1], [meteor3[2], meteor3[2]], lw=3, c="w")
ax[0].legend(fontsize=8)

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
plt.savefig("plots/pirata_linear_fit.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
# -Difference between two randomly selected years (without replacement)

plt.figure(figsize=(5, 5))
for i_sel, sel_lat in enumerate([4, 12]):
    sel_t_air = pirata["t_air"].sel(lon=337, lat=sel_lat)
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
plt.show()
plt.savefig("plots/pirata_pairwise_comparison.pdf", bbox_inches="tight", dpi=300)
