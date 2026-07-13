# %%
import os

os.environ["PAMTRA_DATADIR"] = "/Users/m219063/work/pamtra/pamtra_data"
import seaborn as sns
import pyPamtra
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import utilities.data_utils as dus
import utilities.preprocessing as pp
import utilities.settings_and_colors as set

# %%
ch2 = 53.74
ch3 = 54.96
ch4 = 57.94


cids = dus.get_cids()
old_pe = np.array([[-34.0, 13.5], [-20.0, 13.5], [-20.0, 3.5], [-34.0, 3.5]])

# %%
import os

os.environ["PAMTRA_DATADIR"] = "/Users/m219063/work/pamtra/pamtra_data"
import seaborn as sns
import pyPamtra
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %%
ch2 = 53.74
ch3 = 54.96
ch4 = 57.94
gate_A = np.array(
    [
        [-27.0, 6.5],
        [-23.5, 5.0],
        [-20.0, 6.5],
        [-20.0, 10.5],
        [-23.5, 12.0],
        [-27.0, 10.5],
    ]
)

percusion_E = np.array([[-34.0, 13.5], [-20.0, 13.5], [-20.0, 3.5], [-34.0, 3.5]])

def open_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "sounding": "sonde_id",
                "flight_time": "bin_average_time",
                "flight_lat": "latitude",
                "flight_lon": "longitude",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "latitude", "longitude", "bin_average_time", "sonde_id"])
        .drop_dims(["nv"])
        .swap_dims({"launch_time": "sonde"})
    )


def open_gate(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.set_coords(["launch_lat", "launch_lon", "launch_time"])
        .swap_dims({"sonde": "launch_time"})
        .sel(launch_time=slice("1974-08-10", "1974-09-30"))
        .swap_dims({"launch_time": "sonde"})
    )


def get_cids():
    orcestra_main = "QmPNVTb5fcN59XUi2dtUZknPx5HNnknBC2x4n7dtxuLdwi"
    return {
        "gate": "QmWFfuLW7VSqEFrAwaJr1zY9CzWqF4hC22yqgXELmY133K",
        "orcestra": orcestra_main,
        "radiosondes": f"{orcestra_main}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr",
        "dropsondes": f"{orcestra_main}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    }


def sel_sub_domain(
    ds, polygon, item_var="sonde", lon_var="launch_lon", lat_var="launch_lat"
):
    """
    select points from dataset that lie within the polygon
    """
    from matplotlib.path import Path

    points = np.column_stack([ds[lon_var].values, ds[lat_var].values])
    inside = Path(polygon).contains_points(points)
    return ds.sel(**{item_var: inside})


def sel_gate_A(ds, **kwargs):
    """
    select points from dataset that lie within the gate_A polygon
    """
    return sel_sub_domain(ds, gate_A, **kwargs)


def sel_percusion_E(ds, **kwargs):
    """
    select points from dataset that lie within the percusion_E polygon
    """
    return sel_sub_domain(ds, percusion_E, **kwargs)


# %%

cids = get_cids()
rs = (
    open_radiosondes(cids["radiosondes"])
    .pipe(sel_percusion_E)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)
gs = (
    open_gate(cids["gate"])
    .pipe(sel_percusion_E)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)

hamp = (
    xr.open_dataset(
        "ipfs://bafybeicbj76n3hi52pxtcyzu5in7efk36fk7lavauishclybrsbvlrpq3e",
        engine="zarr",
    ).set_coords(({"lat", "lon", "time"}))
    .pipe(sel_percusion_E, item_var="time", lon_var="lon", lat_var="lat")
)
hamp = (hamp
        .where(np.abs(hamp.plane_pitch) < 3, drop=True)
        .where(np.abs(hamp.plane_roll)  < 3, drop=True)
        .where(hamp.plane_altitude > 13800, drop=True)
        .where(hamp.plane_altitude < 14000, drop=True)
        )

halo_TB = (hamp
           .sel(frequency=53.75)
           .where(np.abs(hamp.plane_roll) < 5, drop=True)
).TBs

halo_alt   = hamp.plane_altitude.mean(dim="time")
halo_nadir = 180 - hamp.plane_pitch.mean(dim="time")
print (f'HALO PERCUSSION E means: nadir {halo_nadir:.2f}, altitude {halo_alt:.2f}')
# %%

# %%
sondes = {
    "orcestra": {"sounding": rs, "tsfc": 301.2},
    "gate": {"sounding": gs, "tsfc": 297.5},
}


pam = pyPamtra.pyPamtra()
pam.df.addHydrometeor(
    (
        "ice",
        -99.0,
        -1,
        917.0,
        130.0,
        3.0,
        0.684,
        2.0,
        3,
        1,
        "mono_cosmo_ice",
        -99.0,
        -99.0,
        -99.0,
        -99.0,
        -99.0,
        -99.0,
        "mie-sphere",
        "heymsfield10_particles",
        0.0,
    )
)
pam.df.nhydro
pam.nmlSet["active"] = False

freqs = np.sort(np.concatenate((np.arange(40, 80, 0.25),np.asarray([ch2,ch3,ch4]))))

nlevs = 240
tb = {}
for key, ds in sondes.items():
    print(f"Running PAMTRA for {key} dataset")
    x = ds["sounding"]
    pamData = dict()
    pamData["temp"] = (
        x.ta[:nlevs].interpolate_na(dim="altitude", method="akima").values 
    )
    pamData["relhum"] = (
        x.rh[:nlevs].interpolate_na(dim="altitude", method="akima").values
    )
    pamData["hgt"] = (
        x.altitude[:nlevs].interpolate_na(dim="altitude", method="akima").values
    )
    pamData["press"] = x.p[:nlevs].interpolate_na(dim="altitude", method="akima").values

    pamData["obs_height"] = np.asarray([halo_alt, 400000.0])
    pamData["lat"] = np.asarray([8.5])
    pamData["lon"] = np.asarray([-23.5])
    pamData["groundtemp"] = [ds["tsfc"]]
    pamData["wind10u"] = [rs.u[0]]
    pamData["wind10v"] = [rs.v[0]]
    pamData["sfc_type"] = [0]

    pam.createProfile(**pamData)
    pam.p["sfc_type"] = np.zeros(pam._shape2D)
    pam.p["sfc_model"] = np.zeros(pam._shape2D)
    pam.p["sfc_refl"] = np.chararray(pam._shape2D)
    pam.p["sfc_refl"][pam.p["sfc_type"] == 0] = "F"
    pam.runPamtra(freqs=freqs)
    tb[key] = pam.r["tb"]

pamtra_tb = xr.DataArray(
    data=np.asarray(
        [
            np.asarray(tb["gate"])[0, 0, :, :, :, :],
            np.asarray(tb["orcestra"])[0, 0, :, :, :, :],
        ]
    ),
    dims=["campaign", "altitude", "angle", "frequency", "polarization"],
    coords={
        "campaign": ["gate", "orcestra"],
        "frequency": freqs,
        "angle": pam.r["angles_deg"],
        "altitude": pam.p["obs_height"][0, 0, :],
        "polarization": ["horizontal", "vertical"],
    },
)
# %%
# %%
sns.set_context(context="paper")
fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

x = pamtra_tb.sel(altitude=400000.0).sel(polarization="horizontal").interp(angle=[180])

x.sel(campaign="orcestra").plot(color="navy", ax=ax[0])
x.sel(campaign="gate").plot(color="orangered", ax=ax[0])
ax[0].axhline(265.5, lw=0.75, ls=":", c="k")

x.diff(dim="campaign").plot(color="teal", ax=ax[1])

ax[0].set_ylabel("$T_\mathrm{b}$ / K")
ax[0].legend(["ORCESTRA", "GATE"], loc="upper right", fontsize=8)
ax[0].set_xlim(40, 80)
ax[1].set_ylabel("$\\Delta T_\\mathrm{b}$ / K")
ax[1].set_xlabel("frequency / GHz")
ax[0].set_xlabel(None)
ax[1].set_xticks([40, 50, 60, 70, 80])
ax[1].set_xticks([ch2], minor=True)
ax[1].set_xlim(40, 80)
ax[1].set_yticks(
    [0, np.round(x.diff(dim="campaign").sel(frequency=ch2).squeeze().values, 2), -2]
)
ax[0].set_title(None)
ax[1].set_title(None)

sns.despine(offset=10)
# %%
alt = 400000.0
ang = 180
y = pamtra_tb.sel(altitude=alt).sel(polarization="horizontal").diff(dim="campaign").interp(angle=ang)
MT = np.round(y.sel(frequency=ch2).squeeze().values, 2)
TP = np.round(y.sel(frequency=ch3).squeeze().values, 2)
LS = np.round(y.sel(frequency=ch4).squeeze().values, 2)


LT = 1.538 * MT - 0.548 * TP + 0.01 * LS
T24 = 1.1 * MT - 0.1 * LS
print(f"Channel 2 {ch2:.2f} GHz: {MT:.2f} K, Channel 3 {ch3:.2f} GHz: {TP:.2f} K, Channel 4 {ch4:.2f} GHz: {LS:.2f} K")
print (f"Bulk temperature change, UAHv6 {LT:.2f} K, and Fu {T24:.2f} K")
print (f"Bulk temperature trends, UAHv6 {LT/5:.2f} K/dec, and Fu {T24/5:.2f} K/dec")





# %%
def open_old_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "sounding": "sonde_id",
                "flight_time": "bin_average_time",
                "flight_lat": "latitude",
                "flight_lon": "longitude",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "latitude", "longitude", "bin_average_time", "sonde_id"])
        .drop_dims(["nv"])
        .swap_dims({"launch_time": "sonde"})
    )
orcestra_main = "QmPNVTb5fcN59XUi2dtUZknPx5HNnknBC2x4n7dtxuLdwi"
rs_old_cid = f"{orcestra_main}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr"
#%%

pe_east = old_pe#pp.sel_percusion_E

rs = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pp.sel_sub_domain, polygon=pe_east)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)

rs_old = open_old_radiosondes(rs_old_cid).pipe(pp.sel_sub_domain, polygon=pe_east).mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
rs_old_interp = open_old_radiosondes(rs_old_cid).pipe(pp.sel_sub_domain, polygon=pe_east).pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
gs = (
    dus.open_gate(cids["gate"])
    .pipe(pp.sel_sub_domain, polygon=pe_east)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)
gate_old_cid =  "QmWFfuLW7VSqEFrAwaJr1zY9CzWqF4hC22yqgXELmY133K"#"QmWZryTDTZu68MBzoRDQRcUJzKdCrP2C4VZfZw1sZWMJJc"
gate_old = (
    dus.open_gate(gate_old_cid)
    .pipe(pp.sel_sub_domain, polygon=pe_east)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()

)
hamp = dus.open_hamp().pipe(
    pp.sel_sub_domain, polygon=pe_east, item_var="time", lon_var="lon", lat_var="lat"
)

halo_TB = (hamp.sel(frequency=53.75).where(np.abs(hamp.plane_roll) < 5, drop=True)).TBs

halo_alt = hamp.plane_altitude.mean(dim="time")
halo_nadir = 180 - hamp.plane_pitch.mean(dim="time")
print(f"HALO PERCUSSION E means: nadir {halo_nadir:.2f}, altitude {halo_alt:.2f}")


# %%

tsfc_gate = 301.2#set.sfc_est["gate"]["T"]
tsfc_orc = 297.5#set.sfc_est["orcestra"]["T"]
sondes = {
    "orcestra": {"sounding": rs, "tsfc": tsfc_orc},
    "orcestra_old": {"sounding": rs_old_interp, "tsfc": tsfc_orc},
    "orcestra_old_int": {"sounding": rs_old, "tsfc": tsfc_orc},
    "gate": {"sounding": gs, "tsfc": tsfc_gate},
    "gate_old": {"sounding": gate_old, "tsfc": tsfc_gate},
}


pam = pyPamtra.pyPamtra()
pam.df.addHydrometeor(
    (
        "ice",
        -99.0,
        -1,
        917.0,
        130.0,
        3.0,
        0.684,
        2.0,
        3,
        1,
        "mono_cosmo_ice",
        -99.0,
        -99.0,
        -99.0,
        -99.0,
        -99.0,
        -99.0,
        "mie-sphere",
        "heymsfield10_particles",
        0.0,
    )
)
pam.df.nhydro
pam.nmlSet["active"] = False

freqs = np.sort(np.concatenate((np.arange(40, 80, 0.25), np.asarray([ch2, ch3, ch4]))))

nlevs = 240
tb = {}
for key, ds in sondes.items():
    print(f"Running PAMTRA for {key} dataset")
    x = ds["sounding"]
    pamData = dict()
    pamData["temp"] = x.ta[:nlevs].interpolate_na(dim="altitude", method="akima").values
    pamData["relhum"] = (
        x.rh[:nlevs].interpolate_na(dim="altitude", method="akima").values
    )
    pamData["hgt"] = (
        x.altitude[:nlevs].interpolate_na(dim="altitude", method="akima").values
    )
    pamData["press"] = x.p[:nlevs].interpolate_na(dim="altitude", method="akima").values

    pamData["obs_height"] = np.asarray([halo_alt, 400000.0])
    pamData["lat"] = np.asarray([8.5])
    pamData["lon"] = np.asarray([-23.5])
    pamData["groundtemp"] = [ds["tsfc"]]
    pamData["wind10u"] = [rs.u[0]]
    pamData["wind10v"] = [rs.v[0]]
    pamData["sfc_type"] = [0]

    pam.createProfile(**pamData)
    pam.p["sfc_type"] = np.zeros(pam._shape2D)
    pam.p["sfc_model"] = np.zeros(pam._shape2D)
    pam.p["sfc_refl"] = np.chararray(pam._shape2D)
    pam.p["sfc_refl"][pam.p["sfc_type"] == 0] = "F"
    pam.runPamtra(freqs=freqs)
    tb[key] = pam.r["tb"]

pamtra_tb = xr.DataArray(
    data=np.asarray(
        [
            np.asarray(tb["gate"])[0, 0, :, :, :, :],
            np.asarray(tb["gate_old"])[0, 0, :, :, :, :],
            np.asarray(tb["orcestra"])[0, 0, :, :, :, :],
            np.asarray(tb["orcestra_old"])[0, 0, :, :, :, :],
            np.asarray(tb["orcestra_old_int"])[0, 0, :, :, :, :],
        ]
    ),
    dims=["campaign", "altitude", "angle", "frequency", "polarization"],
    coords={
        "campaign": ["gate","gate_old", "orcestra", "orcestra_old", "orcestra_old_int"],
        "frequency": freqs,
        "angle": pam.r["angles_deg"],
        "altitude": pam.p["obs_height"][0, 0, :],
        "polarization": ["horizontal", "vertical"],
    },
)

# %%
orc_key = "orcestra_old"
gate_key = "gate_old"
sns.set_context(context="paper")
fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

x = pamtra_tb.sel(altitude=400000.0).sel(polarization="horizontal").interp(angle=[180]).sel(campaign=[ gate_key, orc_key])

x.sel(campaign=orc_key).plot(color="navy", ax=ax[0])
x.sel(campaign=gate_key).plot(color="orangered", ax=ax[0])
ax[0].axhline(265.5, lw=0.75, ls=":", c="k")

x.diff(dim="campaign").plot(color="teal", ax=ax[1])

ax[0].set_ylabel("$T_\mathrm{b}$ / K")
ax[0].legend(["ORCESTRA", "GATE"], loc="upper right", fontsize=8)
ax[0].set_xlim(40, 80)
ax[1].set_ylabel("$\\Delta T_\\mathrm{b}$ / K")
ax[1].set_xlabel("frequency / GHz")
ax[0].set_xlabel(None)
ax[1].set_xticks([40, 50, 60, 70, 80])
ax[1].set_xticks([ch2], minor=True)
ax[1].set_xlim(40, 80)
ax[1].set_yticks(
    [0, np.round(x.diff(dim="campaign").sel(frequency=ch2).squeeze().values, 2), -2]
)
ax[0].set_title(None)
ax[1].set_title(None)

sns.despine(offset=10)

# %%
ang = 180
for key in ["orcestra", "orcestra_old"]:
    print(key)
    for gate_key in ["gate", "gate_old"]:
        print(gate_key)
        for alt in pamtra_tb.altitude.values:
            print("altitude", alt)

            y = (
                pamtra_tb.sel(altitude=alt)
                .sel(polarization="horizontal", campaign=[gate_key, key])
                .diff(dim="campaign")
                .interp(angle=ang)
            )
            MT = np.round(y.sel(frequency=ch2).squeeze().values, 2)
            TP = np.round(y.sel(frequency=ch3).squeeze().values, 2)
            LS = np.round(y.sel(frequency=ch4).squeeze().values, 2)
            LT = 1.538 * MT - 0.548 * TP + 0.01 * LS
            T24 = 1.1 * MT - 0.1 * LS
            print(
                f"Channel 2 {ch2:.2f} GHz: {MT:.2f} K, Channel 3 {ch3:.2f} GHz: {TP:.2f} K, Channel 4 {ch4:.2f} GHz: {LS:.2f} K"
            )
            print(f"Bulk temperature change, UAHv6 {LT:.2f} K, and Fu {T24:.2f} K")
            print(
                f"Bulk temperature trends, UAHv6 {LT / 5:.2f} K/dec, and Fu {T24 / 5:.2f} K/dec"
            )
