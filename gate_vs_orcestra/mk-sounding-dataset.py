#%%
import glob
import os
import xarray as xr
import numpy as np

# IMPORTANT FOR DETERMINISTIC CIDs
import numcodecs
numcodecs.blosc.set_nthreads(1)  

dz = 10
alt10 = xr.DataArray(np.arange(0,31000,dz),
                     dims=('altitude',),
                     attrs={"units" : "m", "long_name" : "altitude in 10 m increments"})
zthresh = 1500

xlb = {"ta":180, "rh":0.00, "ua":-50, "va":-50, "p":1000}
xub = {"ta":305, "rh":1.02, "ua": 50, "va": 50, "p":102500}
xsg = {"ta":3,   "rh":3,   "ua":3  , "va":3  , "p":3}

def clean_ds(ds:xr.Dataset,zthresh=zthresh)->xr.Dataset:
    """ 
    Clean the dataset by removing levels where altitude is not increasing
    over an interval less than zthresh (default 1500 m). To form the Mask 
    using diff it is necessary to pad it at the first so that it aligns with
    the original dataset. 
    """
    mask = xr.concat([
        xr.DataArray(np.ones((1,1),dtype=bool),dims=('time','level')),
        (ds.alt.diff(dim='level') > 0)],
        dim='level')
    try:
        if (ds.where(mask,drop=True).alt.diff(dim='level').max().values < zthresh) :
            ds = ds.where(mask,drop=True)
    finally:
        return ds


def bounds(da:xr.Dataset, quantile = True, sig = 3, dim = 'time')->xr.DataArray:
    """ 
    Create data bounds using quantiles or standard deviations to help identify
    and flag possible outliers. 
    """
    from scipy import stats
    if (quantile) :
        xq = stats.norm.sf(sig)
        x1 = da.quantile(xq,dim=dim)
        x2 = da.quantile(1.- xq,dim=dim)
    else:
        dx = da.std(dim=dim) * sig
        x1 = da.mean(dim=dim) - dx
        x2 = x1 + 2*dx
    return x1, x2


def flag_outliers(ds:xr.Dataset, xlb, xub, xsg)->xr.Dataset:
    """
    flag outliers by setting main variable to nan and writing outlier value to a new
    variable defined with _out.  Outliers are those points outside of xsg of a set of 
    control data
    """
    xx = (ds.assign_coords(dc = ds.monotonicity).swap_dims({'time':'dc'})
        .sel(dc="passed").swap_dims({'dc':'time'}))
    mask = xx.platforms != "METEOR"
    cntrl = xx.isel(time=mask.compute())
    print ('Control data from following platforms:')
    summarize_platforms(cntrl)

    for fld in ["ta","rh","ua","va","p"]:
        x = cntrl[fld].where((cntrl[fld] > xlb[fld]) & (cntrl[fld] < xub[fld]))
        x1, x2 = bounds(x,sig=xsg[fld])
        x = ds[fld].copy(deep=True)
        ds[fld]      = x.where((x>x1) & (x<x2))
        ds[fld+'_out'] = x.where((x<x1) | (x>x2))
        ds[fld+"_out"].attrs = {"long_name" : f"value of {fld}, flagged as outlier"}

    ds["ta_out"].attrs['units'] = "K"
    ds["rh_out"].attrs['units'] = "-"
    ds["ua_out"].attrs['units'] = "m/s"
    ds["va_out"].attrs['units'] = "m/s"
    ds["p_out"].attrs['units']  = "Pa"

    return ds


def process_gate(fdir):
    """
    Process the GATE sounding data from the specified directory.  This involves
    flagging files with non-increasing altitudes, interpolating to radio-sonde
    altitudes, and restructuring the dataset to follow naming and unit conventions.
    The function also adds auxiliary variables and flags outliers based on specified
    thresholds.
    """
    import moist_thermodynamics.constants as constants
    import moist_thermodynamics.saturation_vapor_pressures as svp
    from moist_thermodynamics.functions import partial_pressure_to_specific_humidity as p2q

    Rd  = constants.Rd
    Rv  = constants.Rv
    cpd = constants.cpd
    P0  = constants.P0
    es  = svp.liq_wagner_pruss
    #        
    # Process files.  Usually failure implies a lack of monotonicity in alt
    # coordinate, which we try to repair in a second pass.
    #
    x=[]
    nc_fname=[]
    nbeg = len(glob.glob(f"{fdir}/*.nc"))
    for stage in ['passed','repaired']:
        nc_files = sorted(glob.glob(f"{fdir}/*.nc"))
        for f in nc_files:
            if (stage == 'passed'):
                ds = xr.open_dataset(f)
            else:
                ds = xr.open_dataset(f).pipe(clean_ds,zthresh=1500)
            ds.attrs["monotonicity"] = stage
            try:
                x.append(ds
                    .assign_coords(alt=ds.alt.mean(dim='time'))
                    .swap_dims({'level':'alt'})
                    .interp(alt=alt10))
                nc_fname.append(f)
                os.system(f"rm {f}")
            except Exception as e:
                print(f"File not included in {stage} due to exception {e}")
    nend = len(glob.glob(f"{fdir}/*.nc"))
    #        
    # restructure to follow name and unit conventison
    #
    sondes = xr.concat(x, dim="time")
    sondes["ta"]      = sondes["ta"] + constants.T0 
    sondes.ta.attrs   = {'units':"K"}
    sondes["plev"]    = sondes["plev"] * 100.
    sondes.plev.attrs = {'units':"Pa"}
    sondes["hus"]     = sondes["hus"] / 1000.
    sondes.hus.attrs  = {'units':"kg/kg"}
    sondes["hus_err"] = sondes["hus_err"] / 1000.
    sondes.hus_err.attrs = {'units':"kg/kg"}

    sondes = sondes.rename({"hus" : "q", "hus_err" : "q_err", "plev" : "p", "alt" : "altitude"})

    sondes["aux_longitude"] = xr.DataArray(
        [float(y.attrs["launch_start_position"][:9]) for y in x],
        dims=('time',),
        attrs={"units" : "degrees North", "long_name" : "longitude from launch_start_position"}
        )
    sondes["aux_latitude"] = xr.DataArray(
        [float(y.attrs["launch_start_position"][10:]) for y in x],
        dims=('time',),
        attrs={"units" : "degrees North", "long_name" : "latitude from launch_end_position"}
        )
    sondes["longitude"] = xr.DataArray(
        [float(y.attrs["launch_end_position"][:9]) for y in x],
        dims=('time',),
        attrs={"units" : "degrees North", "long_name" : "longitude from launch_end_position"}
        )
    sondes["latitude"] = xr.DataArray(
        [float(y.attrs["launch_end_position"][10:]) for y in x],
        dims=('time',),
        attrs={"units" : "degrees North", "long_name" : "latitude from launch start position"}
        )
    sondes["platforms"] = xr.DataArray(
        [y.attrs["platform"] for y in x],
        dims=('time',),
        attrs={"long_name" : "Name of Research Vessel from which sonde was launched"}
        )
    sondes["src"] = xr.DataArray(
        [y[y.rfind('/')+1:] for y in nc_fname],
        dims=('time',),
        attrs={"long_name" : "netcdf file name of source data"}
        )
    sondes["monotonicity"] = xr.DataArray(
        [y.attrs["monotonicity"] for y in x],
        dims=('time',),
        attrs={"long_name" : "classification of data continuity"}
        )
    
    sondes = sondes.sortby('time')
    #        
    # define RH for outlier detection
    #
    sondes['rh'] = sondes.q * Rv / (Rd + (Rv-Rd)*sondes.q)  * sondes.p / es(sondes.ta) 
    sondes["rh"].attrs = {"units" : "-", "long_name" : "relative humidity"}
    #        
    # flag outliers
    #
    sondes = flag_outliers(sondes, xlb, xub, xsg)    #        
    #        
    # restrict auxillary variable definition to 'data that passes first montonicity check'
    #
    sondes['theta']       = (sondes.ta) * (P0/sondes.p)**(Rd/cpd)
    sondes["theta"].attrs = {"units" : "K", "long_name" : "potential temperature"}
    sondes["q"]           = p2q( sondes.rh * es( sondes.ta), sondes.p )
    #        
    # update global attributes
    #
    del sondes.attrs["monotonicity"]
    del sondes.attrs["platform"]
    del sondes.attrs["launch_start_position"]
    del sondes.attrs["launch_end_position"]
    sondes.attrs["creator_name"]  = "Bjorn Stevens"
    sondes.attrs["creator_email"] = "bjorn.stevens@mpimet.mpg.de"
    sondes.attrs["title"]         = "GATE phase 2 and 3 ship-soundings"
    sondes.attrs["license"]       = "CC-BY-4.0"
    sondes.attrs["summary"]       = f"GATE ship-soundings: processed {nbeg-nend} of {nbeg} files. "    \
    f"Processing conditional on files having a monotonic coordinate, or non-monotonic gaps smaller "    \
    f"than {zthresh} m which were then filled. Processed data is interpolated to {dz} m grid, with unit " \
    f"conversion, supplemental variables, outliers parsed and meta data provided for traceability"

    return sondes


def summarize_platforms(gate:xr.Dataset):
    """
    Print a summary of the platforms in the gate dataset.
    """
    unique_platforms = np.unique(gate.platforms.values)
    print(f"Platforms in GATE dataset: {len(unique_platforms)}")
    for platform in unique_platforms:
        n = np.sum(gate.platforms.values == platform)
        print(f"{platform:10s} : {n:5d} sondes")
    return


# %%
# - load or create gate sounding data
#
src   = '/Users/m219063/work/data/orcestra/gate/sondes/'
fname = "/Users/m219063/data/gate-radiosondes.zarr"
if os.path.isdir(fname):
    gate  = xr.open_zarr(fname)
else:
    gate  = process_gate(src)
    gate.to_zarr(fname)

unique_platforms = np.unique(gate.platforms.values)
summarize_platforms(gate)
# %%
