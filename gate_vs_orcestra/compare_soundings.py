#  %%
# -Define some functions for plotting and evaluating atmospheric soundings
#
import utilities.data_utils as dus
import utilities.preprocessing as dpp
import utilities.modify_ds as md
from utilities.settings_and_colors import colors

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from moist_thermodynamics import saturation_vapor_pressures as svp
from moist_thermodynamics import functions as mtf
from moist_thermodynamics import constants as mtc
import utilities.thermo as thermo

import seaborn as sns

es = svp.liq_wagner_pruss
# %%
# - load data
#
cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])

rapsodi = dus.open_radiosondes(cids["radiosondes"])
gate = dus.open_gate(cids["gate"])

# %%
# - localize data into different domains
#
sondes = {
    "gate": dpp.sel_percusion_E(gate),
    "rapsodi": dpp.sel_percusion_E(
        rapsodi.assign(
            n2=xr.apply_ufunc(
                mtf.brunt_vaisala_frequency,
                rapsodi.theta,
                rapsodi.q,
                rapsodi.altitude,
                input_core_dims=[["altitude"], ["altitude"], ["altitude"]],
                output_core_dims=[["altitude"]],
                vectorize=True,
            )
        )
    ),
    "beach": dpp.sel_percusion_E(
        beach.assign(
            n2=xr.apply_ufunc(
                mtf.brunt_vaisala_frequency,
                beach.theta,
                beach.q,
                beach.altitude,
                input_core_dims=[["altitude"], ["altitude"], ["altitude"]],
                output_core_dims=[["altitude"]],
                vectorize=True,
            )
        )
    ),
}
sonde_means = {
    key: sondes[key].mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
    for key in sondes.keys()
}


# %%
# - get aircraft data
halo = dus.open_halo(cids["halo"]).pipe(
    dpp.sel_percusion_E, item_var="TIME", lon_var="longitude", lat_var="latitude"
)

hal = (
    halo.swap_dims({"TIME": "altitude"})
    .sortby("altitude")
    .sel(altitude=slice(14000, None))
)
halo_rh = halo.RELHUM.where(halo.RELHUM < 70, drop=True) / 100.0
halo_tk = halo.TS.where(halo.RELHUM < 70, drop=True)
x = hal.swap_dims({"altitude": "TIME"})
zbar = x.altitude.mean().values
# %%
# - parameters for soundings
#
sfc_vals = {key: md.get_surface(ds, es) for key, ds in sondes.items()}
for key, ds in sondes.items():
    print(
        f"{key:10s}: RH = {sfc_vals[key]['RH']:.3f}, T = {sfc_vals[key]['T']:.2f} K, P = {sfc_vals[key]['P'] / 100:.1f} hPa"
    )

# %%
# - zero-degree isotherms
#
print("Height of 0˚ isotherm:")
zp_ticks = {}
for key, ds in sondes.items():
    Tx = (ds.ta - mtc.T0) ** 2
    z_T0 = Tx.idxmin(dim="altitude")
    if ds.title[0] == "G":
        z_T0_gate = z_T0
    print(
        f"  {key}: {z_T0.median().values:.2f}m ({(z_T0.quantile(0.9) - z_T0.quantile(0.1)).values / 2:.2f} m)"
    )
    zp_ticks[key] = z_T0.median().values


# %%
# - cold point
#
keys = ["rapsodi", "gate"]
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 2), sharey=True)

print("Height and temerature of cold point:")
cp_ticks = {}
for key in keys:
    ds = sondes[key]
    mask = (
        ds["ta"].sel(altitude=slice(18000, None)).count(dim="altitude") > 1
    ).compute()
    dsx = ds.isel(sonde=mask)

    z_cp = dsx.ta.idxmin(dim="altitude")
    t_cp = dsx.ta.min(dim="altitude")
    print(f"  {key}: n={np.sum(mask).values} sondes")
    print(
        f"   z at cold point: {z_cp.median().values:.2f} m ({(z_cp.quantile(0.9) - z_cp.quantile(0.1)).values / 2:.2f} m)"
    )
    print(
        f"   T at cold point: {t_cp.median().values:.2f} K ({(t_cp.quantile(0.9) - t_cp.quantile(0.1)).values / 2:.2f} K)"
    )

    z_cp.plot.hist(
        bins=50, density=True, xlim=(14000, 19000), color=colors[key], alpha=0.5
    )

    cp_ticks[key] = z_cp.median().values
ax.set_xlabel("z / m")
sns.despine(offset=10)
# %%

Tx = sonde_means["rapsodi"].ta
RHice = svp.ice_wagner_etal(Tx) / svp.liq_wagner_pruss(Tx)
# %%
# - convective top


ct_ticks = {}
print("Height and temerature of convective top:")
for key, ds in sonde_means.items():
    z_ct = (
        ds.n2.sel(altitude=slice(5e3, 13e3))
        .coarsen(altitude=2, boundary="trim")
        .mean()
        .idxmin(dim="altitude")
    )
    ct_ticks[key] = z_ct.values
    print(f" {key}: z at convective top: {z_ct.values:.2f} m ")

# %%
# - create theoretical soundings
#
Px = 100900.0
P = np.arange(Px, 4000.0, -500)

# %%
# - create theoretical soundings from fits to upper atmosphere
#
T_sig = np.sqrt(0.2896**2 + 0.427**2)
sfc_est = {
    "gate": {
        "T": 298.54,  # 0.1116,
        "RH": 0.805,
    },
    "orcestra": {
        "T": 299.74,  # 0.0819,
        "RH": 0.820,
    },
}
for key in sfc_est.keys():
    sfc_est[key]["q"] = mtf.partial_pressure_to_specific_humidity(
        svp.es_default(sfc_est[key]["T"]) * sfc_est[key]["RH"], Px
    )

Px = 101275.0
P = np.arange(Px, 4000.0, -500)
adiabat_fits = {
    key: thermo.make_sounding_from_adiabat(P, sfc_est[key]["T"], sfc_est[key]["q"])
    for key in sfc_est.keys()
}


deltafits = {
    "orcestra": adiabat_fits["orcestra"].Trho
    - adiabat_fits["gate"].Trho.interp(altitude=adiabat_fits["orcestra"].altitude)
}
for datakey, newkey in [("orcestra", "pls"), ("gate", "mns")]:
    new_T = sfc_est[datakey]["T"] + T_sig
    adiabat_fits[newkey] = thermo.make_sounding_from_adiabat(
        P,
        new_T,
        mtf.partial_pressure_to_specific_humidity(
            svp.es_default(new_T) * sfc_est[datakey]["RH"], Px
        ),
    )
deltafits["pls"] = adiabat_fits["pls"].Trho - adiabat_fits["gate"].Trho.interp(
    altitude=adiabat_fits["pls"].altitude
)
deltafits["mns"] = adiabat_fits["orcestra"].Trho - adiabat_fits["mns"].Trho.interp(
    altitude=adiabat_fits["orcestra"].altitude
)

deltafits["mns"].plot()
deltafits["pls"].plot()
# %%
# - plot profiles
#
cw = 190 / 25.4
sns.set_context(context="paper")
fig, ax = plt.subplots(
    1,
    3,
    figsize=(cw, cw / 2),
    sharey=True,
)
fig.subplots_adjust(wspace=-0.2)

ylim = (0, 23000)
rlim = (0, 1)
dlim = (295, 375)
#
# temperature profiles and adiabats
kwargs = {"ax": ax[0], "y": "altitude", "ylim": ylim, "xlim": (185, 200)}
for key, label in [
    ("rapsodi", "ORCESTRA-RS"),
    ("gate", "GATE-RS"),
    ("beach", "ORCESTRA-DS"),
]:
    sonde_means[key].ta.plot(c=colors[key], ls="-", label=label, **kwargs)


for key, ls in [("orcestra", "--"), ("gate", ":")]:
    adiabat_fits[key].Trho.sel(altitude=slice(0, 15300)).plot(c="grey", ls=ls, **kwargs)


ax[0].annotate(
    "$z_\\mathrm{cp}$",
    xy=(230, (cp_ticks["rapsodi"] + cp_ticks["gate"]) / 2),
    fontsize=8,
)

ax[0].set_xlim(190, 305)
ax[0].set_xlabel("$T$ / K")
ax[0].set_ylabel("z / km")
ax[0].set_xticks(
    [
        np.round(sonde_means["rapsodi"].ta.min().values, 1),
        250,
        np.round(sonde_means["rapsodi"].ta.max().values, 1),
    ]
)
ax[0].set_yticks(np.arange(0, 22000, 3000))
ax[0].set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21])
sns.despine(ax=ax[0], offset={"bottom": 0, "left": 10})

kwargs = {"ax": ax[1], "y": "altitude", "ylim": ylim, "xlim": (0, 0.015)}

for key, label in [
    ("rapsodi", "ORCESTRA-RS"),
    ("gate", "GATE-RS"),
    ("beach", "ORCESTRA-DS"),
]:
    sonde_means[key].sel(altitude=slice(20, 15300)).n2.plot(
        c=colors[key], ls="-", label=label, **kwargs
    )

for key, ls in [("orcestra", "--"), ("gate", ":")]:
    mtf.brunt_vaisala_frequency(
        adiabat_fits[key].theta_rho,
        adiabat_fits[key].q,
        adiabat_fits[key].altitude,
    ).sel(altitude=slice(0, 15300)).plot(c="grey", ls=ls, label=f"{key}-fit", **kwargs)

for x in ["rapsodi", "gate"]:
    print(ct_ticks[x])
    ax[1].axhline(ct_ticks[x], xmin=0.5, xmax=0.8, lw=1, ls=":", c=colors[x])

ax[1].annotate(
    "$z_1$",
    xy=(0.0135, (ct_ticks["rapsodi"] + ct_ticks["gate"]) / 2),
    fontsize=8,
)

ax[1].set_xticks(
    [
        np.round(
            sonde_means["rapsodi"].n2.sel(altitude=slice(9000, None)).min().values, 3
        ),
        np.round(
            sonde_means["rapsodi"].n2.sel(altitude=slice(200, 5000)).max().values, 3
        ),
    ]
)
ax[1].set_xlim(0.005, 0.015)
ax[1].set_xlabel("$N$ / s")
ax[1].set_ylabel("")
ax[1].legend(ncol=2, fontsize=8, loc="best")
sns.despine(ax=ax[1], offset={"bottom": 0, "left": 10})

rs_tdiff = sonde_means["rapsodi"].ta - sonde_means["gate"].ta
ds_tdiff = sonde_means["beach"].ta - sonde_means["gate"].ta
rs_tdiff.plot(
    ax=ax[2], y="altitude", ylim=ylim, label="ORCESTRA-RS", color=colors["rapsodi"]
)
ds_tdiff.sel(altitude=slice(None, 14200)).plot(
    ax=ax[2], y="altitude", ylim=ylim, label="ORCESTRA-DS", color=colors["beach"]
)

altslice = slice(0, 15300)
deltafits["orcestra"].sel(altitude=altslice).where(
    deltafits["orcestra"] > 0.1, drop=True
).plot(ax=ax[2], y="altitude", color="k", ls="-", lw=1)

ax[2].fill_betweenx(
    deltafits["mns"].sel(altitude=altslice).altitude[:-1],
    deltafits["mns"].sel(altitude=altslice)[:-1],
    deltafits["pls"].sel(altitude=altslice),
    color=colors["orcestra"],
    alpha=0.2,
)
ax[2].axvline(0, color="grey", lw=0.5, ls="--")
ax[2].plot([-1, -2.5], [21.0e3, 23e3], color="grey", lw=0.5, ls="-")
ax[2].annotate("RCE", xy=(-2.8, 21200), color="k", fontsize=8)
ax[2].annotate(
    "$z_0$",
    xy=(-2, z_T0.quantile(0.5)),
    fontsize=8,
)

ax[2].set_xlabel("$\\Delta T$ / K")
ax[2].set_ylabel(None)
ax[2].set_xticks(
    [
        np.round(rs_tdiff.min().values, 2),
        np.round(rs_tdiff.sel(altitude=slice(23000, 24500)).mean().values, 2),
        np.round(ds_tdiff.sel(altitude=slice(None, 400)).mean().values, 2),
        np.round(rs_tdiff.max().values, 2),
    ]
)
ax[2].set_xticks([0], minor=True)

ax[2].set_yticks(np.arange(0, 21500, 3000))
ax[2].set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21])
ax[2].set_yticks([z_T0_gate.quantile(0.5), z_T0.quantile(0.5)], minor=True)
for x in ["rapsodi", "gate"]:
    ax[0].axhline(cp_ticks[x], xmin=0.05, xmax=0.3, lw=1, ls=":", c=colors[x])
    ax[2].hlines(zp_ticks[x], xmin=-1.0, xmax=3, lw=1, ls=":", color=colors[x])

sns.despine(ax=ax[2], offset=0)
fig.tight_layout()
fig.savefig("plots/sounding-new.pdf")
# %%

# %%
# - write diagnostics

gx = (
    sonde_means["gate"]
    .sel(altitude=slice(0, 12000))
    .set_coords("p")
    .swap_dims({"altitude": "p"})
)
pgrid = np.asarray([100000, 70000, 25000])
for key in ["rapsodi", "beach"]:
    rx = (
        sonde_means[key]
        .sel(altitude=slice(0, 12000))
        .set_coords("p")
        .swap_dims({"altitude": "p"})
    )
    arr = (rx.interp(p=pgrid) - gx.interp(p=pgrid))["ta"]
    print(
        f"Amplificaiton factors\n {key}: {arr.values[2] / arr.values[1]:.2f}, and {arr.values[2] / arr.values[0]:.2f}"
    )

gp = (
    adiabat_fits["gate"]
    .sel(altitude=slice(0, 12000))
    .set_coords("P")
    .swap_dims({"altitude": "P"})
)
rp = (
    adiabat_fits["orcestra"]
    .sel(altitude=slice(0, 12000))
    .set_coords("P")
    .swap_dims({"altitude": "P"})
)
arrp = (rp.interp(P=pgrid) - gp.interp(P=pgrid))["T"]
print(
    f"Adiabat fits: {arrp.values[2] / arrp.values[1]:.2f}, and {arrp.values[2] / arrp.values[0]:.2f}"
)
gp.interp(P=pgrid).altitude.values


# %%
# - look at distributions
#
kwargs = {
    "bins": 300,
    "density": True,
    "alpha": 0.4,
}
thetabinrange = (294, 304)
rhbinrange = (0.5, 1.1)
pbinrange = (100000, 102000)
sns.set_context("paper")
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for name, ds in [("gate", gate), ("rapsodi", rapsodi), ("beach", beach)]:
    ds["theta"].sel(altitude=slice(0, 300)).plot.hist(
        ax=ax[0], range=thetabinrange, color=colors[name], **kwargs
    )
    ds["rh"].sel(altitude=slice(10, 100)).plot.hist(
        ax=ax[1], range=rhbinrange, color=colors[name], **kwargs
    )
    ds["p"].sel(altitude=slice(10, 50)).plot.hist(
        ax=ax[2], range=pbinrange, color=colors[name], **kwargs
    )


ax[0].set_xlabel("$\\theta_\\mathrm{bl}$ / K")
ax[0].set_xlim(thetabinrange)
ax[0].set_xticks(
    [
        np.round(ds["theta"].sel(altitude=slice(0, 500)).median().values, 2)
        for ds in [gate, rapsodi]
    ]
)


ax[1].set_xlabel("RH$_\\mathrm{sfc}$ / $-$")
ax[1].set_xlim(rhbinrange)
ax[1].set_xticks(
    [
        0.75,
        np.round(
            np.mean(
                [
                    ds["rh"].sel(altitude=slice(0, 50)).median().values
                    for ds in [gate, rapsodi]
                ]
            ),
            2,
        ),
        0.9,
    ]
)

ax[2].set_xlabel("$p_\\mathrm{sfc}$ / $-$")
ax[2].set_xlim(pbinrange)
ax[2].set_xticks(
    [
        100000,
        np.round(
            np.mean(
                [
                    ds["p"].sel(altitude=slice(10, 50)).median().values
                    for ds in [gate, rapsodi]
                ]
            ),
            2,
        ),
        102000,
    ]
)


sns.despine(offset=10)
fig.tight_layout()

# %% additonal analysis
# -- plot difference depending on domain
#

sondes_gateA = {
    key: dpp.sel_gate_A(ds)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
    for key, ds in sondes.items()
}


sns.set_context("paper")
fig, ax = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

for key in ["beach", "rapsodi"]:
    difftheta = sondes_gateA[key].theta - sonde_means[key].theta
    difftheta.plot(ax=ax[0], y="altitude", ylim=ylim, label=key, c=colors[key])
    print(
        f"diff {key} theta: {difftheta.sel(altitude=slice(0, 14000)).mean().values:.2f}"
    )
    diffu = sondes_gateA[key].u - sonde_means[key].u
    diffu.plot(ax=ax[1], y="altitude", ylim=ylim, label=key, c=colors[key])
    print(f"diff {key} u: {diffu.sel(altitude=slice(0, 14000)).mean().values:.2f}")

ax[0].set_ylim(0, 14300)
ax[0].set_xlim(-1, 1)
ax[0].set_xlabel("$\\Delta \\theta$ / K")
ax[1].set_xlabel("$\\Delta u$ / ms$^{-1}$")
ax[0].set_ylabel("z / km")
ax[0].set_xticks(
    [
        (sondes_gateA[key].theta - sonde_means[key].theta)
        .sel(altitude=slice(0, 14000))
        .mean()
        .values
        for key in ["beach", "rapsodi"]
    ],
    minor=True,
)
ax[0].set_yticks(np.arange(0, 12300, 3000))
ax[0].set_yticklabels([0, 3, 6, 9, 12])
ax[1].set_ylabel(None)
plt.legend()

sns.despine(offset=10)
# %%
# - plot theta profiles
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharey=True)

ylim = (0, 15000)
tlim = (296, 356)
nlim = (0, 0.02)
rlim = (0, 1)
for key in ["rapsodi", "beach", "gate"]:
    sonde_means[key].theta.plot(
        ax=ax, y="altitude", ylim=ylim, xlim=tlim, label=key, color=colors[key]
    )

ax.set_xlabel("$\\theta$ / K")

ax.legend(fontsize=10)
fig.tight_layout()
sns.despine(offset=10)
plt.savefig("plots/theta.pdf")
# %%
# - plot zonal and meridional winds
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
for key in ["rapsodi", "beach", "gate"]:
    sonde_means[key].u.plot(ax=ax[0], y="altitude", label=key, color=colors[key])
    sonde_means[key].v.plot(ax=ax[1], y="altitude", label=key, color=colors[key])


ax[0].axvline(x=0.0, ls=":", lw=1)
ax[1].axvline(x=0.0, ls=":", lw=1)

ax[0].set_xlabel("$u$ / ms$^{-1}$")
ax[0].set_ylabel("z / km")
ax[1].set_xlabel("$v$ / ms$^{-1}$")
ax[1].set_ylabel(None)
ax[0].set_xlim(-23, 10)

ax[0].set_yticks(np.arange(0, 18300, 3000))
ax[0].set_yticklabels([0, 3, 6, 9, 12, 15, 18])

z0 = (sonde_means["beach"].u ** 2).idxmin(dim="altitude")
z1 = (sonde_means["beach"].u).idxmin(dim="altitude")
ax[0].set_yticks([z0, z1], minor=True)
ax[0].set_xticks(
    [
        np.round(sonde_means["rapsodi"].u.min().values, 1),
        0,
        np.round(sonde_means["rapsodi"].u.max().values, 1),
    ]
)

ax[1].set_xticks(
    [0, np.round(sonde_means["beach"].v.sel(altitude=slice(0, 5000)).max().values, 1)]
)
ax[1].set_xlim(-3, 7)
ax[0].set_ylim(0, 18000)
sns.despine(offset=10)
plt.savefig("plots/zonal-wind.pdf")
