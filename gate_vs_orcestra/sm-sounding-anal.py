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


def find_x(y_q, f_t):
    return brentq(lambda x: spline(x, y_q, grid=False) - f_t, 290, 310)


find_x_vec = np.vectorize(find_x)

fpath = "/Users/m219063/work/data/orcestra/gate/aircraft"
raf_names = [
    "NCAR_SABRE_MEANS",
    "NASA_CONVAIR_990_MEANS",
    "UKHERCULES_XV208a",
    "NOAA_US-C130_MEANS",
    "UKHERCULES_XV208b",
    "NCAR_ELECTRA_MEANS",
    "DC-7_CEV",
    "NCAR_QUEEN_AIR_MEANS",
    "NOAA_DC-6_MEANS",
]

cpd = constants.cpd
Rd = constants.Rd
Rv = constants.Rv
P0 = constants.P0
T_min = 210.0
T_max = 260.0


es = svp.liq_wagner_pruss
P = np.arange(100500, 10000, -100)

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
for T in Tsfc:
    adiabats.append(adiabat(P, T, RHsfc=0.8, integrate=False))
dx = xr.concat(adiabats, dim=xr.DataArray(Tsfc, dims="Tsfc"))
dx["P"] = dx.P[0, :]
dx["q"] = dx.q[:, 0]

qv = mt.saturation_partition(dx.P, es(dx.T), dx.q)
dx["T_rho"] = dx.T * (1.0 - dx.q + qv * Rv / Rd)

lookup_adiabat = dx.set_coords("P").swap_dims({"levels": "P"}).set_index(P="P")

spline = RectBivariateSpline(
    lookup_adiabat.Tsfc, lookup_adiabat.P[::-1], lookup_adiabat.T_rho[:, ::-1]
)

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
# -- get gate aircraft data
rafs = {}
for raf in raf_names:
    xx = xr.open_mfdataset(f"{fpath}/{raf}/*.nc", combine="by_coords")
    xx = xx.squeeze().sel(time=slice("1974-08-10", None))
    xx = (
        xx.set_coords("lat")
        .sortby("lat")
        .swap_dims({"time": "lat"})
        .sel(lat=slice(4.5, 12.5))
    )
    xx = (
        xx.set_coords("lon")
        .sortby("lon")
        .swap_dims({"lat": "lon"})
        .sel(lon=slice(-34, -20))
    )
    rafs[raf] = xx.swap_dims({"lon": "time"})

# %%
# -- inspect aircraft data
T_max = 260.0
raf_xx = {}
for raf, ds in rafs.items():
    print(f"\n{raf}")
    print(
        f"  Min lat {ds.lat.min().values:.1f} degN, min temperature {ds.ta.min().values:.1f} K"
    )
    ftime = pd.Timestamp(ds.time.min().values.item())
    print(f"  first flight: {pd.Timestamp(ds.time.min().values.item()):%Y-%m-%d}")
    print(f"  last flight: {pd.Timestamp(ds.time.max().values.item()):%Y-%m-%d}")
    if ds.ta.min() < T_max:
        unique_days = pd.to_datetime(ds.time.values).normalize().unique()
        raf_xx[raf] = {}
        raf_xx[raf]["flights"] = []
        raf_xx[raf]["ds"] = ds
        for day in sorted(unique_days):
            dx = ds.sel(time=f"{day:%Y-%m-%d}")
            if dx.ta.min() < T_max:
                raf_xx[raf]["flights"].append(f"{day:%Y-%m-%d}")

for key in raf_xx.keys():
    print(key, len(raf_xx[key]["flights"]))

# %%
# -- plot flight time versus latitude
for raf in raf_xx.keys():
    da = raf_xx[raf]["ds"].lat.groupby("time.hour").mean()
    da.plot.scatter(label=raf, s=50)
plt.legend()
sns.despine(offset=10)

# %%
# -- get moist adiabats
xx = np.asarray([])
for raf in raf_xx.keys():
    ds = raf_xx[raf]["ds"]
    X1 = []
    X2 = []
    for flight in raf_xx[raf]["flights"]:
        dx = ds.sel(time=f"{flight}")
        mask = (dx.ta > T_min) & (dx.ta < T_max)
        p_vals = dx.p.where(mask.compute(), drop=True).values
        ta_vals = dx.ta.where(mask.compute(), drop=True).values
        try:
            xx = np.concatenate([find_x_vec(p_vals, ta_vals), xx])
            X1.append(np.mean(xx))
            X2.append(np.mean(p_vals))
        except:
            pass
    print(f"{raf}", len(X1), len(X2))
    raf_xx[raf]["T_sfc"] = np.asarray(X1)
    raf_xx[raf]["P"] = np.asarray(X2)

# %%
# -- compare to individual soundings
cw = 190 / 25.4
sns.set_context(context="paper")
cmap = plt.get_cmap("twilight")
norm = Normalize(vmin=0, vmax=24)  # hours in a day

ncols = 3
H1 = 11
H2 = 18
for raf in raf_xx.keys():
    nrows = int(np.ceil(len(raf_xx[raf]["flights"]) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(cw, cw / 6 * nrows))
    ds = raf_xx[raf]
    nax = 0
    for flight in raf_xx[raf]["flights"]:
        nrow = int(nax / ncols)
        ncol = nax % ncols
        if len(raf_xx[raf]["flights"]) > len(axs):
            ax = axs[nrow, ncol]
        else:
            ax = axs[ncol]
        dx = raf_xx[raf]["ds"].sel(time=f"{flight}")
        mask = (dx.ta > T_min) & (dx.ta < T_max)
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
    fig.savefig(f"plots/{raf}_night.pdf")
    plt.show()

# %%
# -- compare to individual soundings
cw = 190 / 25.4
sns.set_context(context="paper")


ticks = []
for raf in raf_xx.keys():
    ds = raf_xx[raf]
    tdiff = np.asarray([])
    for flight in raf_xx[raf]["flights"]:
        dx = raf_xx[raf]["ds"].sel(time=f"{flight}")
        mask = (dx.ta > T_min) & (dx.ta < T_max)
        p_vals = dx.p.where(mask.compute(), drop=True).values
        if len(p_vals) > 2:
            gx = (
                gs.swap_dims({"sonde": "launch_time"})
                .sel(launch_time=slice(f"{flight}T{H1}:00", f"{flight}T{H2}:00"))
                .mean(dim="launch_time")
            )
            # gx = gs.swap_dims({"sonde":"launch_time"}).sel(launch_time=f"{flight}").mean(dim='launch_time')
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
    plt.hist(
        tdiff, alpha=0.3, label=raf[0 : raf.find("_M")] + f" {H1}-{H2} UTC", bins=30
    )
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
    mask = (dx.TS > T_min) & (dx.TS < T_max)
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
for raf in raf_xx.keys():
    ds = raf_xx[raf]
    for flight in raf_xx[raf]["flights"]:
        dx = raf_xx[raf]["ds"].sel(time=f"{flight}")
        mask = (dx.ta > T_min) & (dx.ta < T_max)
        p_vals = dx.p.where(mask.compute(), drop=True).values
        ta_vals = dx.ta.where(mask.compute(), drop=True).values
        try:
            X1 = np.concatenate([X1, find_x_vec(p_vals, ta_vals)])
        except:
            pass
sns.kdeplot(X1, label="GATE Aircraft", color=colors["gate"], ax=ax)
ticks.append(np.median(X1))

ticks = []
X2 = np.asarray([])
for flight in intersection:
    day = f"{flight:%Y-%m-%d}"
    dx = halo.sel(TIME=f"{day}")
    mask = (dx.TS > 190) & (dx.TS < 400)
    p_vals = dx.PS.where(mask.compute(), drop=True).values * 100.0
    ta_vals = dx.TS.where(mask.compute(), drop=True).values
    if len(p_vals) > 2:
        try:
            X2 = np.concatenate([X2, find_x_vec(p_vals, ta_vals)])
        except:
            pass
ticks.append(np.median(X1))
ticks.append(np.median(X2))

cw = 190 / 25.4
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(cw / 2, cw / 3))

sns.kdeplot(X1, label="GATE, Aircraft", color=colors["gate"])
sns.kdeplot(X2, label="ORCESTRA, HALO", color=colors["orcestra"])

ticks = [np.median(X1), np.median(X2)]
print(f"Temperature difference of moist adiabats {np.median(X1) - np.median(X2):.2f} K")

ax.set_xlabel("$T_\\mathrm{sfc}$ / K")
ax.set_ylabel("$p(T_\\mathrm{sfc})$")
ax.set_xticks(ticks, minor=True)

sns.despine(offset=10)
plt.legend()
plt.show()

# %%
# -- moist adiabatic analysis of sounding datasets
datasets = {
    "gate": {"sondes": gs, "color": colors["gate"]},
    "beach": {"sondes": bs, "color": colors["beach"]},
    "rapsodi": {"sondes": rs, "color": colors["rapsodi"]},
    "gate-day": {"sondes": gs, "color": colors["gate"]},
    "rapsodi-day": {"sondes": rs, "color": colors["rapsodi"]},
}

T_min = 225.0
T_max = 260.0

for dataset in datasets.keys():
    datasets[dataset]["exclude"] = []
    sondes = datasets[dataset]["sondes"]

    if dataset.find("day") != -1:
        mask = (sondes.launch_time.dt.hour >= 10) & (xs.launch_time.dt.hour < 20)
        sondes = sondes.where(mask, drop=True)
    T1 = []
    T2 = []
    datasets[dataset]["tma"] = np.asarray([])
    datasets[dataset]["ta"] = np.asarray([])

    for sonde in np.arange(0, sondes["sonde"].size, 1):
        xs = sondes.sel(sonde=sonde)

        mask = (xs.ta > T_min) & (xs.ta < T_max)
        p_vals = xs.p.where(mask, drop=True).values
        ta_vals = xs.ta.where(mask, drop=True).values

        try:
            xx = find_x_vec(p_vals, ta_vals)
            T1.append(np.mean(xx))
            T2.append(np.std(xx))
            datasets[dataset]["tma"] = np.concatenate([datasets[dataset]["tma"], xx])
            datasets[dataset]["ta"] = np.concatenate([datasets[dataset]["ta"], ta_vals])

        except:
            datasets[dataset]["exclude"].append(sonde)

    datasets[dataset]["Tsfc_avg"] = np.asarray(T1)
    datasets[dataset]["Tsfc_std"] = np.asarray(T2)

# %%
# -- plotting moist adiabatic analysis
for dataset in datasets.keys():
    ls = "solid"
    if dataset.find("day") != -1:
        ls = "dashed"
    sns.kdeplot(
        datasets[dataset]["Tsfc_avg"],
        label=f"{dataset}",
        color=datasets[dataset]["color"],
        lw=2.0,
        ls=ls,
    )
    print(
        f"{dataset}: excluded {len(datasets[dataset]['exclude'])} of {datasets[dataset]['sondes']['sonde'].size} sondes, median {np.median(datasets[dataset]['Tsfc_avg']):.2f}"
    )

plt.legend()
plt.gca().set_xlabel("Moist Adiabat / K")
plt.gca().set_xlim(297.5, 301.5)
tick1 = np.median(datasets["gate"]["Tsfc_avg"]).round(2)
tick2 = np.median(datasets["rapsodi"]["Tsfc_avg"]).round(2)
plt.gca().set_xticks([tick1, tick2])
tick3 = np.median(datasets["beach"]["Tsfc_avg"]).round(2)
plt.gca().set_xticks([tick3], minor=True)

sns.despine(offset=10)

# %%
# -- moist adiabatic analysis as scatter plot
for dataset in datasets.keys():
    try:
        color = colors[dataset]
        alpha = 0.01
        plt.scatter(
            datasets[dataset]["tma"],
            datasets[dataset]["ta"],
            label=dataset,
            color=color,
            alpha=alpha,
            s=3,
        )
    except:
        print(f"not plotting {dataset}")

plt.gca().set_xlabel("Moist Adiabat / K")
plt.gca().set_ylabel("Temperature / K")
plt.gca().set_xlim(296.5, 301.5)
sns.despine(offset=10)
