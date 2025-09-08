# %%

import xarray as xr
import matplotlib.pyplot as plt

# %%

# https://zenodo.org/records/7051674
ds = xr.open_dataset("../data/CircBrazil_Sonne_soundings_level2_v1.0.0.nc")

# %%

ds_orcestra = ds.where(
    (ds.lat > 4.5) & (ds.lat < 12.5) & (ds.lon > -34) & (ds.lon < -20), drop=True
)

lats = ds_orcestra.lat.isel(alt=0)
lons = ds_orcestra.lon.isel(alt=0)
plt.scatter(lons, lats)

num_orcestra_east_sondes = len(lats)
num_orcestra_east_sondes_august = len(
    ds_orcestra.where(ds_orcestra.launch_time.dt.month == 8, drop=True).sounding
)

print(
    f"""Find a total of {num_orcestra_east_sondes} radiosonde launches 
      in the ORCESTRA East domain. Of these {num_orcestra_east_sondes_august} 
      were launched in August 2021."""
)
# %%
