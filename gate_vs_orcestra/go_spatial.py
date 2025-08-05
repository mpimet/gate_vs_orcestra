# %%
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import utilities.data_utils as dus
import utilities.preprocessing as dpp
from utilities.settings_and_colors import colors, gate_B, gate_A, percusion_E

# %%
# - load gate sounding data
#
cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])
rapsodi = dus.open_radiosondes(cids["radiosondes"])
gate = dus.open_gate(cids["gate"])

gs_PE = dpp.sel_percusion_E(gate)
rs_PE = dpp.sel_percusion_E(rapsodi)
bs_PE = dpp.sel_percusion_E(beach)

gs_GA = dpp.sel_gate_A(gs_PE)
rs_GA = dpp.sel_gate_A(rs_PE)
bs_GA = dpp.sel_gate_A(bs_PE)
# %%
# - plot geographic distribution of soundings
#
sns.set_context(context="paper")
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": ccrs.PlateCarree()})

ngso = len(gs_PE.sonde)
nrso = len(rs_PE.sonde)
nbso = len(bs_PE.sonde)
ngs = len(gs_GA.sonde)
nrs = len(rs_GA.sonde)
nbs = len(bs_GA.sonde)

kwargs = {"transform": ccrs.PlateCarree(), "marker": "o"}
ax.scatter(
    gs_PE.launch_lon,
    gs_PE.launch_lat,
    s=1,
    color=colors["gate"],
    alpha=0.5,
    label=f"GATE (n={ngso})",
    **kwargs,
)
ax.scatter(
    rs_PE.launch_lon,
    rs_PE.launch_lat,
    s=1,
    color=colors["rapsodi"],
    alpha=0.5,
    label=f"ORCESTRA (Meteor, n={nrso})",
    **kwargs,
)
ax.scatter(
    bs_PE.launch_lon,
    bs_PE.launch_lat,
    s=1,
    color=colors["beach"],
    alpha=0.5,
    label=f"ORCESTRA (HALO, n={nbso})",
    **kwargs,
)
ax.scatter(
    gs_GA.launch_lon,
    gs_GA.launch_lat,
    s=4,
    color=colors["gate"],
    alpha=0.5 / 3,
    label=f"GATE (n={ngs})",
    **kwargs,
)
ax.scatter(
    rs_GA.launch_lon,
    rs_GA.launch_lat,
    s=4,
    color=colors["rapsodi"],
    alpha=0.5,
    label=f"ORCESTRA (Meteor, n={nrs})",
    **kwargs,
)
ax.scatter(
    bs_GA.launch_lon,
    bs_GA.launch_lat,
    s=4,
    color=colors["beach"],
    alpha=0.5,
    label=f"ORCESTRA (HALO, n={nbs})",
    **kwargs,
)

tr = rs_GA.launch_time.mean().dt.strftime("%m-%d").values
td = bs_GA.launch_time.mean().dt.strftime("%m-%d").values
tg = gs_GA.launch_time.mean().dt.strftime("%m-%d").values

print(
    f"RAPSODI: {rs_GA.launch_lon.mean().values:.2f}, {rs_GA.launch_lat.mean().values:.2f}, {tr}"
)
print(
    f"BEACH:   {bs_GA.launch_lon.mean().values:.2f}, {bs_GA.launch_lat.mean().values:.2f}, {td}"
)
print(
    f"GATE:    {gs_GA.launch_lon.mean().values:.2f}, {gs_GA.launch_lat.mean().values:.2f}, {tg}"
)
#
# Add map features
#
ax.coastlines(color="black")
ax.add_feature(cfeature.LAND, facecolor="lightgray")

ax.set_xlim(-37.0, -13.0)
ax.set_ylim(2.0, 18.0)

gl = ax.gridlines(
    draw_labels=True, linewidth=1, color="none", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([])
gl.ylocator = mticker.FixedLocator([])

ax.set_xlabel("longitude / $^\\circ$W")
ax.set_ylabel("latitude / $^\\circ$N")

gate_color = "k"
ax.plot(gate_B[:, 0], gate_B[:, 1], color="k", lw=1, label="GATE B Array")
ax.plot(
    gate_A[:3, 0], gate_A[:3, 1], color="k", lw=1, ls="dashed", label="GATE A/B Array"
)
ax.plot(gate_A[3:, 0], gate_A[3:, 1], color="k", lw=1, ls="dashed")
ax.plot(
    percusion_E[:2, 0],
    percusion_E[:2, 1],
    color="k",
    lw=1,
    ls="dotted",
    label="ORCESTRA East",
)
ax.plot(percusion_E[2:, 0], percusion_E[2:, 1], color="k", lw=1, ls="dotted")

h_g1 = mlines.Line2D(
    [],
    [],
    color=colors["gate"],
    marker="o",
    linestyle="None",
    markersize=3,
    alpha=0.3,
    label=f"GATE ($n=$ {ngs})",
)
h_g2 = mlines.Line2D(
    [],
    [],
    color=colors["gate"],
    marker="o",
    linestyle="None",
    markersize=1,
    alpha=0.75,
    label=f"GATE ($n=$ {ngso})",
)
h_r1 = mlines.Line2D(
    [],
    [],
    color=colors["rapsodi"],
    marker="o",
    linestyle="None",
    markersize=3,
    alpha=0.9,
    label=f"METEOR ($n=${nrs})",
)
h_r2 = mlines.Line2D(
    [],
    [],
    color=colors["rapsodi"],
    marker="o",
    linestyle="None",
    markersize=1,
    alpha=0.9,
    label=f"METEOR ($n=${nrso})",
)
h_b1 = mlines.Line2D(
    [],
    [],
    color=colors["beach"],
    marker="o",
    linestyle="None",
    markersize=3,
    alpha=0.9,
    label=f"HALO ($n=${nbs})",
)
h_b2 = mlines.Line2D(
    [],
    [],
    color=colors["beach"],
    marker="o",
    linestyle="None",
    markersize=1,
    alpha=0.9,
    label=f"HALO ($n=${nbso})",
)

ax.legend(
    ncol=3, loc="upper left", handles=[h_r1, h_r2, h_b1, h_b2, h_g1, h_g2], fontsize=6
)

xticks = [-34, -27.0, -23.5, -20]
yticks = [3.5, 8.5, 13.5]
ax.set_xticks(xticks)
ax.set_xlabel("longitude / $^\\circ$W")
ax.set_yticks(yticks)
ax.set_ylabel("latitude / $^\\circ$N")

ax.annotate(
    "GATE B array",
    xy=gate_B[4],
    xytext=(-21.8, 12),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="k"),
)

# ax.annotate(
#     "Dakar",
#     xy=(-17.467686,14.716677),
#     xytext=(-20.5,14),
#     fontsize=8,
#     arrowprops=dict(arrowstyle="->", color='k')
# )

ax.annotate(
    "Praia",
    xy=(-23.51254, 14.93152),
    xytext=(-22.5, 14.2),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="k"),
)

ax.annotate(
    "GATE A/B array",
    xy=(-20.5, 5.5),
    fontsize=8,
)
ax.annotate(
    "ORCESTRA East",
    xy=(-35.5, 13.7),
    fontsize=8,
)

plt.savefig("plots/gate-orcestra-sondes.pdf")

# %%
