# %%
import xarray as xr
import fsspec
import hashlib
import utilities.data_utils as dus
import numpy as np
import tqdm
from importlib import metadata


from pydropsonde.processor import Sonde
from pydropsonde.pipeline import sondes_to_gridded

pydropsonde_version = metadata.version("pydropsonde")

# %%
cids = dus.get_cids()

path_to_rs_l1 = f"ipfs://{cids['orcestra']}/products/Radiosondes/Level_1/RAPSODI_RS_METEOR_ORCESTRA_level1"
rapsodi_met_files = dus.fsglob(path_to_rs_l1 + "/*.nc")
rapsodi_inmg_files = dus.fsglob(
    f"ipfs://{cids['orcestra']}/products/Radiosondes/Level_1/RAPSODI_RS_INMG_ORCESTRA_level1/*.nc"
)
rapsodi_bco_files = dus.fsglob(
    f"ipfs://{cids['orcestra']}/products/Radiosondes/Level_1/RAPSODI_RS_BCO_ORCESTRA_level1/*.nc"
)
rapsodi_l1_files = rapsodi_met_files + rapsodi_inmg_files + rapsodi_bco_files


# %%
def get_ascent_flag(da):
    if "ascent" in str(da.values):
        return 1
    elif "descent" in str(da.values):
        return 0
    else:
        return np.nan


def open_dataset(file):
    fs = fsspec.filesystem("ipfs")
    ds = (
        xr.open_dataset(fs.open(file))
        .squeeze("sounding")
        .reset_coords(["flight_time", "p", "sounding", "lat", "lon", "p"])
        .rename(
            {
                "alt": "gpsalt",
                "height_ptu": "alt",
                "flight_time": "time",
                "platform": "platform_id",
                "sounding": "vaisala_serial_id",
            }
        )
        .swap_dims({"level": "time"})
        .drop_vars(["level", "dz", "dp", "mr"])
    )
    ds = ds.assign(
        sonde_id=hashlib.sha256(
            str(ds.vaisala_serial_id.values).encode("ascii")
        ).hexdigest()[:16],
        ascent_flag=get_ascent_flag(ds.vaisala_serial_id),
        launch_lat=ds.lat.interpolate_na(
            "time", method="nearest", fill_value="extrapolate"
        )
        .isel(time=0)
        .values,
        launch_lon=ds.lon.interpolate_na(
            "time", method="nearest", fill_value="extrapolate"
        )
        .isel(time=0)
        .values,
    )

    return ds  # .chunk(chunks={"time": -1})


# %%


def iterate_method_over_sondes(sondes, function, **kwargs):
    new_sondes = []
    for sonde in tqdm.tqdm(sondes):
        sonde = function(sonde, **kwargs)

        if sonde is not None:
            new_sondes.append(sonde)
    return new_sondes


# %%

sondes = []
for f in rapsodi_l1_files:
    ds = open_dataset(f)
    nonzero_count = np.count_nonzero(~np.isnan(ds.p.values))
    _, time_idc = np.unique(ds.time, return_index=True)
    ds = ds.isel(time=time_idc)

    sonde = Sonde(_serial_id=ds.sonde_id.values, _launch_time=ds.launch_time.values)
    sonde.add_l2_ds(l2_ds=ds)
    sonde.alt_dim = "gpsalt"
    sonde.global_attrs = {"global": ds.attrs}
    sondes.append(sonde)


# %%

new_sondes = iterate_method_over_sondes(sondes.copy(), Sonde.create_interim_l3)

new_sondes = iterate_method_over_sondes(new_sondes, Sonde.set_alt_dim, alt_dim="gpsalt")
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.replace_alt_dim)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_q_to_l2_ds)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_theta_to_l2_ds)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_wind_components)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.interpolate_alt_dim)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.remove_non_mono_incr_alt)
new_sondes = iterate_method_over_sondes(
    new_sondes, Sonde.swap_alt_dimension, dropna=False
)
new_sondes = iterate_method_over_sondes(
    new_sondes,
    Sonde.interpolate_variables_to_common_grid,
    interp_stop=30000,
    method="linear_interpolate",
)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.drop_empty)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.recalc_q_and_ta)

new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_iwv, qc_var=None)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_wind)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.update_history_l3)
# %%

gridded = sondes_to_gridded(new_sondes, config=None)
# gridded.add_history_to_gridded()
gridded.add_dim_names()
gridded.concat_sondes()

gridded.global_attrs["l3"] = {
    "title": "RAPSODI radiosonde dataset (Level 2)",
}

gridded.history = ""

gridded.l3_filename = "RAPSODI_l2.zarr"
gridded.write_l3(l3_dir="/Users/helene/Documents/Data/GATE/")
