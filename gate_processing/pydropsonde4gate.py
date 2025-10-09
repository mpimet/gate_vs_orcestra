# %%
import xarray as xr
import hashlib
import numpy as np
import glob
import tqdm

from utilities.gate_process import mask_unphysical, mask_outliers

from pydropsonde.processor import Sonde
from pydropsonde.pipeline import sondes_to_gridded
from pydropsonde.circles import Circle

from importlib import metadata

pydropsonde_version = metadata.version("pydropsonde")


# %%
path_to_rs_l1 = "/Users/helene/Documents/Data/GATE/gate-sondes/"
gate_l1 = glob.glob(path_to_rs_l1 + "/*.nc")
gate_l1
# %%


def open_dataset(file):
    ds = (
        xr.open_dataset(file)
        .squeeze("time")
        .rename(
            {
                "time": "launch_time",
                "flight_time": "time",
                "ua": "u",
                "va": "v",
                "hus": "q",
                "plev": "p",
            }
        )
    )
    ds = (
        ds.assign(
            sonde_id=(
                "sonde",
                [
                    hashlib.sha256(
                        str(file).split("/")[-1].encode("ascii")
                    ).hexdigest()[:6]
                ],
                {"long_name": "sonde identifier"},
            ),
            launch_lat=(
                "sonde",
                [float(ds.attrs["launch_end_position"].split()[1])],
                {
                    "long_name": "latitude of drop end position",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            launch_lon=(
                "sonde",
                [float(ds.attrs["launch_end_position"].split()[0])],
                {
                    "long_name": "longitude of drop end position",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            platform_id=(
                "sonde",
                [ds.attrs["platform"]],
                {"long_name": "platform identifier"},
            ),
            gpsalt=(
                "level",
                ds.alt.values,
                {"long_name": "altitude", "standard_name": "altitude", "units": "m"},
            ),
            p=(
                "level",
                ds.p.values * 100,
                {"long_name": "pressure", "standard_name": "pressure", "units": "Pa"},
            ),
            q=(
                "level",
                ds.q.values / 1000,
                {
                    "long_name": "specific humidity",
                    "standard_name": "specific_humidity",
                    "units": "kg kg-1",
                },
            ),
            ta=(
                "level",
                ds.ta.values + 273.15,
                {
                    "long_name": "air temperature",
                    "standard_name": "air_temperature",
                    "units": "K",
                },
            ),
            time=(
                "level",
                ds.time.interpolate_na(
                    "level", method="linear", fill_value="extrapolate"
                ).values,
            ),
            ascent_flag=(
                "sonde",
                [
                    int(
                        ds.alt.interpolate_na("level", fill_value="extrapolate")
                        .diff("time")
                        .values[0]
                        > 0
                    )
                ],
                {"long_name": "ascent flag, 1 for ascent, 0 for descent"},
            ),
            vaisala_serial_id=("sonde", [file]),
        )
        .squeeze("sonde")
        .swap_dims({"level": "time"})
    )
    ds = ds.assign()
    if ds.sizes["time"] == 0:
        return None
    return ds


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
for f in gate_l1:
    ds = open_dataset(f)
    nonzero_count = np.count_nonzero(~np.isnan(ds.p.values))
    _, time_idc = np.unique(ds.time, return_index=True)
    ds = ds.isel(time=time_idc)

    sonde = Sonde(_serial_id=ds.sonde_id.values, _launch_time=ds.launch_time.values)
    sonde.add_l2_ds(l2_ds=ds)
    sonde.alt_dim = "gpsalt"
    sonde.global_attrs = {"global": ""}
    sondes.append(sonde)


# %%

new_sondes = iterate_method_over_sondes(sondes.copy(), Sonde.create_interim_l3)

new_sondes = iterate_method_over_sondes(new_sondes, Sonde.set_alt_dim, alt_dim="gpsalt")
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.replace_alt_dim)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_rh_to_l2_ds)
new_sondes = iterate_method_over_sondes(new_sondes, Sonde.add_theta_to_l2_ds)
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

gridded.global_attrs = {
    "global": {
        "creator_name": "Bjorn Stevens",
        "creator_email": "bjorn.stevens@mpimet.mpg.de",
        "license": "CC-BY-4.0",
        "institution": "Max Planck Institute for Meteorologie",
        "source": "radiosonde",
        "references": "https://github.com/hgloeckner/gate_vs_orcestra, https://github.com/atmdrops/pydropsonde",
    },
    "l3": {
        "title": "GATE radiosonde dataset (Level 2)",
        "summary": "GATE ship radiosondes (gridded to same altitude)",
        "processing_level": "2",
    },
}
gridded.history = ""

gridded.l3_filename = "GATE_l2.zarr"
gridded.write_l3(l3_dir="/Users/helene/Documents/Data/GATE/")
# %%
circle = Circle(
    circle_ds=mask_unphysical(gridded.concat_sonde_ds).drop_vars(["interpolated_time"]),
    clon=0.0,
    clat=0.0,
    crad=0.0,
    flight_id="",
    platform_id="",
    segment_id="",
    alt_dim="altitude",
    sonde_dim="sonde",
)
circle.interpolate_na_sondes(method="linear", max_gap=1500)
circle.extrapolate_na_sondes(max_alt=300)
# %%
gate_final = (
    circle.circle_ds.pipe(mask_outliers)
    .sel(
        altitude=slice(-0, 25000),
    )
    .where(circle.circle_ds.launch_time.dt.month.isin([8, 9]), drop=True)
)

# %%
gridded.global_attrs["l3"] = {
    "title": "GATE radiosonde dataset (Level 3)",
    "summary": "GATE ship radiosondes (filled and cleaned) subsetted to ORCESTRA time of year and lower 25 km",
    "processing_level": "3",
}

gridded.history = (
    f"Processed Gate level 2 radiosonde data with pydropsonde version {pydropsonde_version}",
)

gridded.l3_filename = "GATE_l3.zarr"
gridded.concat_sonde_ds = gate_final
gridded.write_l3(l3_dir="/Users/helene/Documents/Data/GATE/")
