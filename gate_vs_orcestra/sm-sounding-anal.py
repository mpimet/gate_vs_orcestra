# %%
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import Normalize

from moist_thermodynamics import functions as mt
from moist_thermodynamics import constants
from moist_thermodynamics import utilities as utils
from moist_thermodynamics import saturation_vapor_pressures as svp

import utilities.data_utils as dus
import utilities.preprocessing as dpp
from utilities.settings_and_colors import colors


def adiabat(
    P, Tsfc=301.0, RHsfc=0.8, Psfc=100500.0, Tmin=200.0, thx=mt.theta_l, integrate=False
) -> xr.Dataset:
    """creates a sounding from a moit adiabat

    Cacluates the moist adiabate based either on an integration or a specified
    isentrope with pressure as the vertical coordinate.

    Args:
        P: pressure
        Tsfc: starting (value at P.max()) temperature
        qsfc: starting (value at P.max()) specific humidity
        Tmin: minimum temperature of adiabat
        thx: function to calculate isentrope if integrate = False
        integrate: determines if explicit integration will be used.
    """

    qsfc = mt.partial_pressure_to_specific_humidity(mt.es_default(Tsfc) * RHsfc, Psfc)

    TPq = xr.Dataset(
        data_vars={
            "T": (
                ("levels",),
                utils.moist_adiabat_with_ice(
                    P, Tx=Tsfc, qx=qsfc, Tmin=Tmin, thx=thx, integrate=integrate
                ),
                {"units": "K", "standard_name": "air_temperature", "symbol": "$T$"},
            ),
            "P": (
                ("levels",),
                P,
                {"units": "Pa", "standard_name": "air_pressure", "symbol": "$P$"},
            ),
            "q": (
                ("levels",),
                qsfc * np.ones(len(P)),
                {"units": "1", "standard_name": "specific_humidity", "symbol": "$q$"},
            ),
        },
    )
    TPq = TPq.assign(
        altitude=xr.DataArray(
            mt.pressure_altitude(TPq.P, TPq.T, qv=TPq.q).values,
            dims=("levels"),
            attrs={
                "units": "m",
                "standard_name": "altitude",
                "description": "hydrostatic altitude given the datasets temperature and pressure",
            },
        )
    )
    TPq = TPq.assign(
        theta=(
            TPq.T.dims,
            mt.theta(TPq.T, TPq.P).values,
            {
                "units": "K",
                "standard_name": "air_potential_teimerature",
                "symbol": "$\theta$",
            },
        )
    )
    TPq = TPq.assign(
        P0=xr.DataArray(
            constants.P0, attrs={"units": "Pa", "standards_name": "referenece_pressure"}
        )
    )

    return TPq


fpath = "/Users/m219063/work/data/orcestra/gate/aircraft"

cpd = constants.cpd
Rd = constants.Rd
Rv = constants.Rv
P0 = constants.P0
T_min = 220.0
T_max = 260.0

cw = 190 / 25.4
sns.set_context(context="paper")

es = svp.liq_wagner_pruss
P = np.arange(101275, 10000, -100)

cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])
rapsodi = dus.open_radiosondes(cids["radiosondes"])
gate = dus.open_gate(cids["gate"])

rs = dpp.sel_percusion_E(dus.open_radiosondes(cids["radiosondes"]))
gs = dpp.sel_percusion_E(dus.open_gate(cids["gate"]))
bs = dpp.sel_percusion_E(dus.open_dropsondes(cids["dropsondes"]))
# %%
# -- lookup adiabatic values
Tsfc = np.arange(290, 305, 0.1)
adiabats = []
Psfc = gs.p.mean(dim="sonde")[0].squeeze().values
RHsfc = gs.rh.mean(dim="sonde")[0].squeeze().values
for T in Tsfc:
    adiabats.append(adiabat(P, T, Psfc=Psfc, RHsfc=RHsfc, integrate=False))
dx = xr.concat(adiabats, dim=xr.DataArray(Tsfc, dims="Tsfc"))
dx["P"] = dx.P[0, :]
dx["q"] = dx.q[:, 0]

qv = mt.saturation_partition(dx.P, es(dx.T), dx.q)
dx["T_rho"] = dx.T * (1.0 - dx.q + qv * Rv / Rd)

lookup_adiabat = dx.set_coords("P").swap_dims({"levels": "P"}).set_index(P="P")

spline80 = RectBivariateSpline(
    lookup_adiabat.Tsfc, lookup_adiabat.P[::-1], lookup_adiabat.T_rho[:, ::-1]
)

adiabats = []
Psfc = rs.p.mean(dim="sonde")[0].squeeze().values
RHsfc = rs.rh.mean(dim="sonde")[0].squeeze().values
for T in Tsfc:
    adiabats.append(adiabat(P, T, Psfc=Psfc, RHsfc=RHsfc, integrate=False))
dx = xr.concat(adiabats, dim=xr.DataArray(Tsfc, dims="Tsfc"))
dx["P"] = dx.P[0, :]
dx["q"] = dx.q[:, 0]

qv = mt.saturation_partition(dx.P, es(dx.T), dx.q)
dx["T_rho"] = dx.T * (1.0 - dx.q + qv * Rv / Rd)

lookup_adiabat = dx.set_coords("P").swap_dims({"levels": "P"}).set_index(P="P")

spline82 = RectBivariateSpline(
    lookup_adiabat.Tsfc, lookup_adiabat.P[::-1], lookup_adiabat.T_rho[:, ::-1]
)

spline = spline80


def find_x(y_q, f_t, spline=spline):
    return brentq(lambda x: spline(x, y_q, grid=False) - f_t, 290, 310)


find_x_vec = np.vectorize(find_x)

# %%
# - get halo data
halo = (
    xr.open_dataset(
        "ipfs://bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",
        engine="zarr",
    )
    .rename_vars(
        {
            "IRS_LAT": "latitude",
            "IRS_LON": "longitude",
            "IRS_ALT": "altitude",
        }
    )
    .set_coords(({"latitude", "longitude", "altitude"}))
).pipe(dpp.sel_percusion_E, item_var="TIME", lon_var="longitude", lat_var="latitude")

hal = (
    halo.swap_dims({"TIME": "altitude"})
    .sortby("altitude")
    .sel(altitude=slice(14000, None))
)
halo.RELHUM.where(halo.RELHUM < 70, drop=True) / 100.0
halo_tk = halo.TS.where(halo.RELHUM < 70, drop=True)

# %%
# -- get and adjust gate aircrat data
ra_dsd = {}
for ra, ds in (dus.open_gate_ras(dus.get_cids())).items():
    ds = (
        ds.sel(time=slice("1974-08-10", None))
        .squeeze()
        .pipe(dpp.sel_percusion_E, item_var="time", lon_var="lon", lat_var="lat")
    )
    mask = ds.ta < T_max
    p_vals = ds.p.where(mask.compute(), drop=True).values
    ta_vals = ds.ta.where(mask.compute(), drop=True).values
    if ds.ta.min() < T_max:
        unique_days = pd.to_datetime(ds.time.values).normalize().unique()
        ra_dsd[ra] = {}
        ra_dsd[ra]["flights"] = []
        ra_dsd[ra]["ds"] = ds
        for day in sorted(unique_days):
            dx = ds.sel(time=f"{day:%Y-%m-%d}")
            if dx.ta.min() < T_max:
                ra_dsd[ra]["flights"].append(f"{day:%Y-%m-%d}")
        print(f"\n{ra}: {len(ra_dsd[ra]['flights'])} flights")
        print(
            f"  Min lat {ds.lat.min().values:.1f} degN, min temperature {ta_vals.min():.1f} K"
        )
        print(
            f"  Mean lat {ds.lat.mean().values:.1f} degN, {p_vals.max() / 100:.1f} hPa - {p_vals.min() / 100:.1f} hPa"
        )
        ftime = pd.Timestamp(ds.time.min().values.item())
        print(f"  first flight: {pd.Timestamp(ds.time.min().values.item()):%Y-%m-%d}")
        print(f"  last flight: {pd.Timestamp(ds.time.max().values.item()):%Y-%m-%d}")

# %%
# -- plot flight time versus latitude
for ra in ra_dsd.keys():
    da = ra_dsd[ra]["ds"].lat.groupby("time.hour").mean()
    da.plot.scatter(label=ra, s=50)
plt.legend()
sns.despine(offset=10)

# %%
# -- get moist adiabats
xx = np.asarray([])
for ra in ra_dsd.keys():
    ds = ra_dsd[ra]["ds"]
    X1 = []
    X2 = []
    for flight in ra_dsd[ra]["flights"]:
        dx = ds.sel(time=f"{flight}")
        mask = dx.ta < T_max
        p_vals = dx.p.where(mask.compute(), drop=True).values
        ta_vals = dx.ta.where(mask.compute(), drop=True).values
        try:
            xx = np.concatenate([find_x_vec(p_vals, ta_vals, spline=spline80), xx])
            X1.append(np.mean(xx))
            X2.append(np.mean(p_vals))
        except Exception as e:
            print(f"Error: {e}")
    ra_dsd[ra]["T_sfc"] = np.asarray(X1)
    ra_dsd[ra]["P"] = np.asarray(X2)

# %%
# -- compare to individual soundings
cmap = plt.get_cmap("twilight")
norm = Normalize(vmin=0, vmax=24)  # hours in a day

ncols = 3
H1 = 11
H2 = 18
for ra in ra_dsd.keys():
    nrows = int(np.ceil(len(ra_dsd[ra]["flights"]) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(cw, cw / 6 * nrows))
    ds = ra_dsd[ra]
    nax = 0
    for flight in ra_dsd[ra]["flights"]:
        nrow = int(nax / ncols)
        ncol = nax % ncols
        if len(ra_dsd[ra]["flights"]) > len(axs):
            ax = axs[nrow, ncol]
        else:
            ax = axs[ncol]
        dx = ra_dsd[ra]["ds"].sel(time=f"{flight}")
        mask = dx.ta < T_max
        p_vals = dx.p.where(mask.compute(), drop=True).values
        ta_vals = dx.ta.where(mask.compute(), drop=True).values
        if len(p_vals) > 2:
            for sonde in gs.swap_dims({"sonde": "launch_time"}).sel(
                launch_time=f"{flight}"
            )["launch_time"]:
                ts = pd.Timestamp(sonde.values)
                hour = ts.hour + ts.minute / 60 + ts.second / 3600
                if hour > H2 or hour < H1:
                    gx = gs.swap_dims({"sonde": "launch_time"}).sel(
                        launch_time=sonde.values
                    )
                cb = ax.scatter(gx.p, gx.ta, s=2.0, alpha=0.5, color=cmap(norm(hour)))
            for x, y in zip(p_vals, ta_vals):
                ax.plot([x, x], [y - 0.25, y + 0.25], color="k", lw=2)
            ax.set_xlim(p_vals.min() - 1000, p_vals.max() + 1000)
            ax.set_ylim(ta_vals.min() - 1, ta_vals.max() + 1)
        else:
            ax.axis("off")  # hides ticks, labels, AND the box/spines
        nax += 1
        ax.set_title(flight)

    fig.tight_layout()
    fig.savefig(f"plots/{ra}_night.pdf")
    plt.show()

# %%
# -- histogram of temperature differences with mean sounding

ticks = []
for ra in ra_dsd.keys():
    ds = ra_dsd[ra]
    tdiff = np.asarray([])
    for flight in ra_dsd[ra]["flights"]:
        dx = ra_dsd[ra]["ds"].sel(time=f"{flight}")
        mask = dx.ta < T_max
        p_vals = dx.p.where(mask.compute(), drop=True).values
        if len(p_vals) > 2:
            gx = (
                gs.swap_dims({"sonde": "launch_time"})
                .sel(launch_time=slice(f"{flight}T{H1}:00", f"{flight}T{H2}:00"))
                .mean(dim="launch_time")
            )
            # gx = gs.swap_dims({"sonde":"launch_time"}).sel(launch_time=f"{flight}").mean(dim='launch_time')
            try:
                gx_bar_ta = (
                    gx.where(~np.isnan(gx.p), drop=True)
                    .set_coords("p")
                    .swap_dims({"altitude": "p"})
                    .interp(p=p_vals)
                    .ta
                )
                tdiff = np.concatenate(
                    [tdiff, dx.ta.where(mask.compute(), drop=True).values - gx_bar_ta]
                )
            except Exception as e:
                print(f"Error: {e} -- no valid values for {ra} on {flight}")

    plt.hist(tdiff, alpha=0.3, label=ra[0 : ra.find("_M")] + f" {H1}-{H2} UTC", bins=30)
    ticks.append(np.median(tdiff))

sns.despine(offset=10)
plt.gca().set_xlabel("$\\Delta T$ / K")
plt.gca().set_ylabel("samples")
plt.gca().set_xticks(ticks, minor=True)
plt.legend()

# %%
# -- Halo sounding aircraft difference
ds = halo
tdiff = np.asarray([])
ticks = []

halo_days = pd.to_datetime(halo.TIME.values).normalize().unique()
meteor_days = pd.to_datetime(rs.launch_time.values).normalize().unique()
intersection = [x for x in halo_days if x in meteor_days]

for flight in intersection:
    day = f"{flight:%Y-%m-%d}"
    dx = halo.sel(TIME=f"{day}")
    mask = dx.TS < T_max
    p_vals = dx.PS.where(mask.compute(), drop=True).values
    if len(p_vals) > 2:
        rx = (
            rs.swap_dims({"sonde": "launch_time"})
            .sel(launch_time=slice(f"{flight}T{H1}:00", f"{flight}T{H2}:00"))
            .mean(dim="launch_time")
        )
        # rx = rs.swap_dims({"sonde":"launch_time"}).sel(launch_time=f"{day}").mean(dim='launch_time')
        rx_bar_ta = (
            rx.where(~np.isnan(rx.p), drop=True)
            .set_coords("p")
            .swap_dims({"altitude": "p"})
            .interp(p=p_vals * 100)
            .ta
        )
        tdiff = np.concatenate(
            [tdiff, dx.TS.where(mask.compute(), drop=True).values - rx_bar_ta]
        )
plt.hist(
    tdiff, alpha=0.3, label=f"HALO {H1}-{H2} UTC", bins=30, color=colors["rapsodi"]
)
ticks.append(np.median(tdiff))

sns.despine(offset=10)
plt.gca().set_xlabel("$\\Delta T$ / K")
plt.gca().set_ylabel("samples")
plt.gca().set_xticks(ticks, minor=True)
plt.legend()

# %%
# -- histogram of moist adiabats
ticks = []
X1 = np.asarray([])
for ra in ra_dsd.keys():
    ds = ra_dsd[ra]
    for flight in ra_dsd[ra]["flights"]:
        dx = ra_dsd[ra]["ds"].sel(time=f"{flight}")
        mask = dx.ta < T_max
        p_vals = dx.p.where(mask.compute(), drop=True).values
        ta_vals = dx.ta.where(mask.compute(), drop=True).values
        try:
            X1 = np.concatenate([X1, find_x_vec(p_vals, ta_vals, spline=spline80)])
        except Exception as e:
            print(f"Error: {e}")

sns.kdeplot(X1, label="GATE Aircraft", color=colors["gate"], ax=ax)
ticks.append(np.median(X1))

ticks = []
X2 = np.asarray([])
for flight in intersection:
    day = f"{flight:%Y-%m-%d}"
    dx = halo.sel(TIME=f"{day}")
    mask = dx.TS < T_max
    p_vals = dx.PS.where(mask.compute(), drop=True).values * 100.0
    ta_vals = dx.TS.where(mask.compute(), drop=True).values
    if len(p_vals) > 2:
        try:
            X2 = np.concatenate([X2, find_x_vec(p_vals, ta_vals, spline=spline82)])
        except Exception as e:
            print(f"Error: {e}")
ticks.append(np.median(X1))
ticks.append(np.median(X2))

# %%
# -- moist adiabatic analysis of sounding datasets
datasets = {
    "gate": {"sondes": gs, "color": colors["gate"]},
    "beach": {"sondes": bs, "color": colors["beach"]},
    "rapsodi": {"sondes": rs, "color": colors["rapsodi"]},
    "gate-day": {"sondes": gs, "color": colors["gate"]},
    "rapsodi-day": {"sondes": rs, "color": colors["rapsodi"]},
}

for dataset in datasets.keys():
    datasets[dataset]["exclude"] = []
    sondes = datasets[dataset]["sondes"]

    if dataset.find("day") != -1:
        mask = (sondes.launch_time.dt.hour >= 10) & (sondes.launch_time.dt.hour < 20)
        sondes = sondes.where(mask, drop=True)
    T1 = []
    T2 = []
    datasets[dataset]["tma"] = np.asarray([])
    datasets[dataset]["ta"] = np.asarray([])

    spline = spline82
    if dataset.find("gate") != -1:
        spline = spline80
        print(f"computing {dataset} with spline 80")

    for sonde in np.arange(0, sondes["sonde"].size, 1):
        xs = sondes.sel(sonde=sonde)

        mask = (xs.ta > T_min) & (xs.ta < T_max) & (xs.p > 10000.0)
        p_vals = xs.p.where(mask, drop=True).values
        ta_vals = xs.ta.where(mask, drop=True).values

        try:
            xx = find_x_vec(p_vals, ta_vals, spline=spline)
            T1.append(np.mean(xx))
            T2.append(np.std(xx))
            datasets[dataset]["tma"] = np.concatenate([datasets[dataset]["tma"], xx])
            datasets[dataset]["ta"] = np.concatenate([datasets[dataset]["ta"], ta_vals])
        except Exception as e:
            print(
                f"Error: {e} -- excluding sonde {sonde}, {len(p_vals)}, {len(ta_vals)}"
            )
            datasets[dataset]["exclude"].append(sonde)

    datasets[dataset]["Tsfc_avg"] = np.asarray(T1)
    datasets[dataset]["Tsfc_std"] = np.asarray(T2)

print(
    f"mean: {datasets['gate']['Tsfc_avg'].mean():.2f}K, std {datasets['gate']['Tsfc_std'].mean():.2f}K"
)
# %%
# -- plotting moist adiabatic analysis
fig, axs = plt.subplots(2, 1, figsize=(cw / 2, cw / 1.25))

for dataset in datasets.keys():
    ls = "solid"
    if dataset.find("day") != -1:
        ls = "dashed"
    sns.kdeplot(
        datasets[dataset]["Tsfc_avg"],
        ax=axs[0],
        label=f"{dataset}",
        color=datasets[dataset]["color"],
        lw=2.0,
        ls=ls,
    )
    print(
        f"{dataset}: excluded {len(datasets[dataset]['exclude'])} of {datasets[dataset]['sondes']['sonde'].size} sondes, median {np.median(datasets[dataset]['Tsfc_avg']):.2f}"
    )

tick1 = np.median(datasets["gate"]["Tsfc_avg"]).round(2)
tick2 = np.median(datasets["rapsodi"]["Tsfc_avg"]).round(2)
tick3 = np.median(datasets["beach"]["Tsfc_avg"]).round(2)
print(np.std(datasets["rapsodi"]["Tsfc_avg"]).round(4))
print(np.std(datasets["gate"]["Tsfc_avg"]).round(4))
axs[0].set_xticks([tick1, tick2])
axs[0].set_xticks([tick3], minor=True)
axs[0].set_yticks([0, 1, 2])
axs[0].set_ylim(0, 2.1)
axs[0].set_xlim(297.5, 301.5)
axs[0].legend(ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0), loc="upper right")

sns.kdeplot(X1, ax=axs[1], label="GATE, Aircraft", color=colors["gate"], lw=2)
sns.kdeplot(X2, ax=axs[1], label="ORCESTRA, HALO", color=colors["orcestra"], lw=2)

ticks = [np.median(X1).round(2), np.median(X2).round(2)]
print(f"Temperature difference of moist adiabats {np.median(X1) - np.median(X2):.2f} K")

axs[1].set_xlabel("$T_*$ / K")

axs[1].set_ylim(0.0, 1.4)
axs[1].set_xticks(ticks)
axs[1].set_ylim(0, 1.1)
axs[1].set_yticks([0, 0.5, 1])
axs[1].set_xlim(297, 302)

sns.despine(offset=10)
axs[1].legend(ncol=1, fontsize=8, bbox_to_anchor=(1.05, 1.0), loc="upper right")

sns.despine(offset=10)
fig.tight_layout()
plt.savefig("plots/moist-adiabats.pdf")

# %%
# -- moist adiabatic analysis as scatter plot

fig, ax = plt.subplots(1, 1, figsize=(cw / 2, cw / 2))

for dataset in ["gate", "rapsodi", "beach"]:
    try:
        color = colors[dataset]
        if dataset == "gate":
            alpha = 0.005
        else:
            alpha = 0.01
        plt.scatter(
            datasets[dataset]["tma"],
            datasets[dataset]["ta"],
            label=dataset,
            edgecolors="none",
            facecolors=color,
            alpha=alpha,
            s=2,
        )
    except Exception as e:
        print(f"Error: {e} -- not plotting {dataset}")

ax.hlines([220, 260], xmin=296, xmax=302, color="k", ls=":")
ax.set_xlabel("$T_*$ / K")
ax.set_ylabel("$T$ / K")
ax.set_xlim(296.5, 301.5)
ax.invert_yaxis()

sns.despine(offset=10)
fig.tight_layout()

fig.savefig("plots/fits-scatter.png")

# %%
# -- document timing of radiosondes to address reviewer question
nbins = 8
hour = gs.launch_time.dt.hour
hour.plot.hist(bins=nbins, range=(0, 23), alpha=0.5, label="gate rs")

hour = rs.launch_time.dt.hour
hour.plot.hist(bins=nbins, range=(0, 23), alpha=0.5, label="orcestra-rapsodi")

hour = bs.launch_time.dt.hour
hour.plot.hist(bins=nbins, range=(0, 23), alpha=0.5, label="orcestra-beach")

sns.despine(offset=10)
plt.legend()
plt.gca().set_xlim(0, 23.5)
plt.gca().set_xticks([0, 6, 12, 18])

# %%
fig, ax = plt.subplots(1, 1, figsize=(cw / 1.5, cw / 2.5))
times = bs.launch_time
pressures = halo.sel(TIME=times, method="nearest").PS * 100
nbins = 50
ticks = []
for dp in [1000, 2000, 3000]:
    tx = bs.ta.where(bs.p < pressures + dp).max(dim="altitude")
    tx.plot.hist(
        ax=ax,
        bins=nbins,
        range=(203, 228),
        alpha=0.5,
        label=f"Dropsonde at $P_\\mathrm{{HALO}}$+{dp / 100:.0f} hPA",
    )
    ticks.append(tx.mean().round(2))
tx = halo.sel(TIME=times, method="nearest").TS
tx.plot.hist(ax=ax, bins=nbins, range=(203, 228), alpha=0.5, label="HALO")
ticks.append(tx.mean().round(2))
sns.despine(offset=10)
ax.set_xlim(207, 223)
ax.set_xticks(ticks)
ax.set_xlabel("$T$ / hPa")
plt.legend(fontsize=8)
fig.tight_layout()

fig.savefig("plots/flight-dropsonde-temperature.pdf")

# %%
