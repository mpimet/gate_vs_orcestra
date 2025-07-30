import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob

#    'Accent':  A qualitative colormap with 8 distinct colors.
#    'Dark2':   A qualitative colormap with 8 distinct colors.
#    'Paired':  A qualitative colormap with 12 distinct colors.
#    'Pastel1': A qualitative colormap with 9 distinct colors.
#    'Pastel2': A qualitative colormap with 8 distinct colors.
#    'Set1':    A qualitative colormap with 9 distinct colors.
#    'Set2':    A qualitative colormap with 8 distinct colors.
#    'Set3':    A qualitative colormap with 12 distinct colors.
#    'tab10':   A qualitative colormap with 10 distinct colors.
#    'tab20':   A qualitative colormap with 20 distinct colors.
#    'tab20b':  A qualitative colormap with 20 distinct colors.
#    'tab20c':  A qualitative colormap with 20 distinct colors.

meteor = glob.glob('/Users/m300083/Projekte/GATE/GATE_Radiosonde_Data/3.00.02.104-3.31.02.101_19740601-19740930/RADIOSONDE/METEOR/8db9d2/*.nc')
gate_array_v1 = glob.glob('/Users/m300083/Projekte/GATE/GATE_Radiosonde_Data/3.31.02.101-3.33.02.101_19740601-19740930_v1/*.nc')
gate_array_v2 = glob.glob('/Users/m300083/Projekte/GATE/GATE_Radiosonde_Data/3.31.02.101-3.33.02.101_19740601-19740930_v2/*.nc')

test = glob.glob('/Users/m300083/Projekte/GATE/GATE_Radiosonde_Data/test/*.nc')

files = meteor + gate_array_v2

datasets = [xr.open_dataset(file) for file in files]

positions = [ds.attrs['launch_end_position'].split() for ds in datasets]
platforms = [ds.attrs['platform'] for ds in datasets]

positions = [(float(lon), float(lat)) for lon, lat in positions]

fig = plt.figure(figsize=(10, 10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-40, 0, 0, 40])
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

LONGITUDE_FORMATTER = LongitudeFormatter(zero_direction_label=True)
LATITUDE_FORMATTER = LatitudeFormatter()

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

colors = plt.cm.tab10(np.linspace(0, 1, len(set(platforms))))

unique_platforms = set(platforms)
for i, platform in enumerate(unique_platforms):
    platform_positions = [pos for pos, plat in zip(positions, platforms) if plat == platform]
    lons, lats = zip(*platform_positions)
    ax.scatter(lons, lats, label=platform, color=colors[i], s=5)

ax.legend(loc='upper left')
fig.savefig('map.png', dpi=300)
plt.show()
