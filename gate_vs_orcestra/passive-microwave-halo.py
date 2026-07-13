# %%
import os

os.environ["PAMTRA_DATADIR"] = "/Users/m219063/work/pamtra/pamtra_data"
import seaborn as sns
import pyPamtra
import matplotlib.pyplot as plt
import utilities.preprocessing as pp
import numpy as np
import xarray as xr
import utilities.settings_and_colors as set

import utilities.data_utils as dus

# %%
ch2 = 53.74
ch3 = 54.96
ch4 = 57.94

pe_region = set.percusion_E


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


new_cids = dus.get_cids()
old_gate = (
    dus.open_gate("QmWFfuLW7VSqEFrAwaJr1zY9CzWqF4hC22yqgXELmY133K")
    .pipe(pp.sel_sub_domain, polygon=pe_region)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)
gate = (
    dus.open_gate(new_cids["gate"])
    .pipe(pp.sel_sub_domain, polygon=pe_region)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)
rs = (
    dus.open_radiosondes(new_cids["radiosondes"])
    .pipe(pp.sel_sub_domain, polygon=pe_region)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)
old_rs_cid = "QmPNVTb5fcN59XUi2dtUZknPx5HNnknBC2x4n7dtxuLdwi/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr"
old_rs = (
    open_old_radiosondes(old_rs_cid)
    .pipe(pp.sel_sub_domain, polygon=pe_region)
    .mean(dim="sonde")
    .coarsen(altitude=10, boundary="trim")
    .mean()
)
# %%

hamp = dus.open_hamp().pipe(
    pp.sel_sub_domain, polygon=pe_region, item_var="time", lon_var="lon", lat_var="lat"
)

halo_TB = (hamp.sel(frequency=53.75).where(np.abs(hamp.plane_roll) < 5, drop=True)).TBs

halo_alt = hamp.plane_altitude.mean(dim="time")
halo_nadir = 180 - hamp.plane_pitch.mean(dim="time")
print(f"HALO PERCUSSION E means: nadir {halo_nadir:.2f}, altitude {halo_alt:.2f}")
# %%

# %%
sondes = {
    "orcestra": {"sounding": rs, "tsfc": set.sfc_est["orcestra"]["T"]},
    "orcestra_old": {"sounding": old_rs, "tsfc": set.sfc_est["orcestra"]["T"]},
    "gate_old": {"sounding": old_gate, "tsfc": set.sfc_est["gate"]["T"]},
    "gate": {"sounding": gate, "tsfc": set.sfc_est["gate"]["T"]},
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
# %%
pamtra_tb = xr.DataArray(
    data=np.asarray(
        [
            np.asarray(tb["gate"])[0, 0, :, :, :, :],
            np.asarray(tb["gate_old"])[0, 0, :, :, :, :],
            np.asarray(tb["orcestra"])[0, 0, :, :, :, :],
            np.asarray(tb["orcestra_old"])[0, 0, :, :, :, :],
        ]
    ),
    dims=["campaign", "altitude", "angle", "frequency", "polarization"],
    coords={
        "campaign": ["gate", "gate_old", "orcestra", "orcestra_old"],
        "frequency": freqs,
        "angle": pam.r["angles_deg"],
        "altitude": pam.p["obs_height"][0, 0, :],
        "polarization": ["horizontal", "vertical"],
    },
)
# %%
gate_key = "gate_old"
orc_key = "orcestra_old"
alt = 400000.0
ang = 180
sns.set_context(context="paper")
fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

x = (
    pamtra_tb.sel(altitude=alt)
    .sel(polarization="horizontal", campaign=[gate_key, orc_key])
    .interp(angle=[ang])
)

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

for gate_key in ["gate", "gate_old"]:
    print(f"Results for {gate_key}")
    for orc_key in ["orcestra", "orcestra_old"]:
        print(f"Comparing with {orc_key}")
        x = (
            pamtra_tb.sel(altitude=alt)
            .sel(polarization="horizontal", campaign=[gate_key, orc_key])
            .interp(angle=[ang])
        )

        y = x.diff(dim="campaign")
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
