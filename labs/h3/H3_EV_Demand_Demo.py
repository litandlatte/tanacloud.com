"""
H3 for Data Science - Detecting EV Charging to See a Competitor's Demand (ABC Corp)

Auto-generated from H3_EV_Demand_Demo.ipynb - edit the notebook, not this file.
Run as a script or cell-by-cell in VS Code via the # %% markers. Interactive folium
maps only display inside a notebook.

Data is downloaded from the data/ folder of this repository at run time.

Requires: pip install h3 folium pandas numpy scikit-learn
"""

# %% [markdown]
# # H3 for Data Science — EV Charging Demand Prediction (ABC Corp)
#
# This notebook has **two sections**:
#
# 1. **H3 Fundamentals** — how H3 divides the Earth into hexagons, what resolutions (zoom levels) mean, the parent/child hierarchy, and neighbours — with live maps.
# 2. **Business demo** — using H3 to predict EV recharging demand across Europe and recommend where ABC Corp should build new stations, versus the traditional haversine cross-join approach.
#
# It is designed to run top-to-bottom on **Google Colab**. Just run the install cell first.

# %%
# Colab: install the libraries this notebook needs.
!pip -q install h3 folium scikit-learn

# %%
import time

import h3
import numpy as np
import pandas as pd
import folium

print('h3 version:', h3.__version__)

# %% [markdown]
# ## H3 in one picture — resolutions across Chennai
#
# The **Chennai city area is highlighted in yellow**. Use the **layer control (top-right of the
# map)** to overlay H3 hexagons at resolutions 6–8 and watch the same city go from a handful of
# big hexagons to over a thousand small ones — the quickest way to *feel* what an H3 "resolution"
# means, and it renders in a fraction of a second.

# %%
from folium import FeatureGroup, LayerControl

# Chennai city area (approximate bounding box)
CHENNAI_BOX = [(13.25, 80.10), (13.25, 80.32), (12.82, 80.32), (12.82, 80.10)]
chennai_poly = h3.LatLngPoly(CHENNAI_BOX)

# 3 resolutions - light enough to render instantly during a live demo
res_colors = {6: '#d73027', 7: '#fc8d59', 8: '#3186cc'}

m = folium.Map(location=[13.03, 80.21], zoom_start=11, tiles='cartodbpositron')

# Highlight Chennai in a distinct colour (always visible)
folium.Polygon([[la, lo] for la, lo in CHENNAI_BOX], color='#000000', weight=2.5,
               fill=True, fill_color='#ffd400', fill_opacity=0.20,
               tooltip='Chennai city area').add_to(m)

# One toggleable H3 layer per resolution
for res, color in res_colors.items():
    edge_km = h3.average_hexagon_edge_length(res, unit='km')
    n = 0
    fg = FeatureGroup(name=f"H3 res {res}  (~{edge_km:.2f} km edge)", show=(res == 7))
    for c in h3.polygon_to_cells(chennai_poly, res):
        boundary = [[la, lo] for la, lo in h3.cell_to_boundary(c)]
        folium.Polygon(boundary, color=color, weight=0.8, fill=True,
                       fill_color=color, fill_opacity=0.15).add_to(fg)
        n += 1
    fg.add_to(m)
    print(f"res {res}: {n} hexagons cover Chennai (~{edge_km:.2f} km edge)")

LayerControl(collapsed=False).add_to(m)
print("Toggle resolutions (res 6-8) with the layer control in the top-right of the map.")
print(m)

# %% [markdown]
# # Section 1 — H3 Fundamentals: How the Earth Becomes Hexagons
#
# **H3** is Uber's open-source **Hexagonal Hierarchical Geospatial Indexing System**.
# It answers a simple question: *given a latitude/longitude, which "cell" of the world does it fall into?* — in a way that is fast, consistent, and multi-scale.
#
# How it is built:
# - Start with an **icosahedron** (a 20-faced shape) wrapped around the globe.
# - Project it down to **122 base cells** — **110 hexagons + 12 pentagons** (the 12 pentagons are unavoidable when you tile a sphere with hexagons).
# - Each cell is recursively subdivided into finer cells across **16 resolutions** (0 = coarsest, 15 = finest). Each finer level splits a cell into ~7 children (an *aperture-7* hierarchy).
# - Any point maps to exactly **one cell ID** per resolution, and cells **nest** neatly inside their parents.
#
# Think of resolution like a **map zoom level**: low resolution = big hexagons covering regions; high resolution = tiny hexagons covering a street corner.

# %% [markdown]
# ## 1.1 The 16 resolutions (zoom levels)
#
# The table below shows, for every resolution, how many cells cover the **whole Earth**, the **average hexagon edge length**, the **average hexagon area**, and an **Indian-context yardstick** for that area. Notice how each step multiplies the number of cells by ~7 and shrinks the area accordingly.
#
# The last column is the one to read aloud: 4.3 million km2 and 0.9 m2 are both just numbers until one of them is the country you live in and the other is the table you eat at.

# %%
# The 16 H3 resolutions, with an Indian-context yardstick for each hexagon's average area.
# INDIA_SCALE is a hand-written lookup, not a calculation: a bare km2 figure means nothing to an
# audience, so each resolution is pinned to something they can picture. Coarse cells map onto
# states and cities; fine cells use everyday Indian land units.
#   1 acre  = 4,047 m2      1 ground = 2,400 sq ft = 223 m2      1 cent = 435.6 sq ft = 40.5 m2
INDIA_SCALE = {
    0:  '1.3x the whole of India',
    1:  '1.8x Rajasthan, the largest state',
    2:  'about the size of Bihar',
    3:  'about the size of Tripura',
    4:  'about the size of Delhi (NCT)',
    5:  '~60% of Chennai city (GCC area)',
    6:  'about one Chennai corporation zone',
    7:  'a locality such as T. Nagar or Adyar',
    8:  '182 acres\u2014a third of the IIT Madras campus',
    9:  '26 acres\u2014a large gated community',
    10: '3.7 acres\u2014a school campus',
    11: '0.5 acre\u2014about 10 house plots',
    12: '307 m2\u2014a 1.4-ground house plot',
    13: '44 m2\u2014a 1BHK flat',
    14: '6 m2\u2014a small bathroom',
    15: '0.9 m2\u2014a dining table',
}

# Values are kept at FULL precision here and formatted only for display below. Rounding first
# would flatten the fine end - res 15's true 0.0000009 km2 rounds away to 0.000001 at 6 decimals.
rows = []
for r in range(16):
    rows.append(dict(
        resolution=r,
        num_cells_on_earth=h3.get_num_cells(r),
        avg_edge_km=h3.average_hexagon_edge_length(r, unit='km'),
        avg_area_km2=h3.average_hexagon_area(r, unit='km^2'),
        india_scale=INDIA_SCALE[r],
    ))
res_df = pd.DataFrame(rows)

# Plain decimals, not scientific notation. The areas span 4.36 million km2 down to under a
# millionth of one, and pandas' default '4.357449e+06' hides exactly the point this table makes.
res_df.style.format({
    'num_cells_on_earth': '{:,}',
    'avg_edge_km': '{:,.4f}',
    'avg_area_km2': '{:,.8f}',
}).hide(axis='index')

# %% [markdown]
# **How to read it:**
# - **Resolution 0** — 122 huge cells; each spans ~1,000+ km. One cell is **1.3x India**. Good for continents.
# - **Resolution 5** (~9.9 km edge) — city / metro scale; **~60% of Chennai city**.
# - **Resolution 8** (~0.5 km edge) — neighbourhood scale, **~180 acres** (what we use for EV catchments).
# - **Resolution 9** (~0.2 km edge) — a few city blocks, **~26 acres**.
# - **Resolution 15** — sub-metre; **0.9 m2**, smaller than a car.
#
# Choosing a resolution is a **modelling decision**: it sets the spatial scale of your analysis. We tune it explicitly in Section 2.

# %% [markdown]
# ## 1.2 A single point at many resolutions
#
# The same location gets a **different cell ID at each resolution** — finer resolutions give longer, more specific IDs. Here is a point in Amsterdam encoded from resolution 0 to 10.

# %%
lat, lon = 52.3676, 4.9041  # Amsterdam
for r in range(0, 11):
    cell = h3.latlng_to_cell(lat, lon, r)
    print(f'res {r:>2} -> {cell}')

# %% [markdown]
# ## 1.3 The hierarchy: parents and children
#
# Because cells nest, you can move **up** (coarser, `cell_to_parent`) or **down** (finer, `cell_to_children`) the hierarchy. This is what makes H3 great for **multi-scale aggregation** — roll fine cells up into coarse ones for free.

# %%
cell8 = h3.latlng_to_cell(lat, lon, 8)
print('cell at res 8      :', cell8)
print('its parent (res 7) :', h3.cell_to_parent(cell8, 7))
print('its parent (res 5) :', h3.cell_to_parent(cell8, 5))
print('children at res 9  :', len(h3.cell_to_children(cell8, 9)), '(~7 per level)')
print('children at res 10 :', len(h3.cell_to_children(cell8, 10)), '(~49 = 7x7)')

# %% [markdown]
# ## 1.4 Neighbours and `grid_disk`
#
# Every hexagon has **6 neighbours**, all at (roughly) the **same centre-to-centre distance** — this is the big advantage of hexagons over squares (squares have 8 neighbours at two different distances). `grid_disk(cell, k)` returns a cell plus every cell within `k` rings — a ready-made **catchment area**.

# %%
print('immediate neighbours (k=1):', h3.grid_disk(cell8, 1))
print()
for k in range(0, 4):
    print(f'grid_disk k={k} -> {len(h3.grid_disk(cell8, k))} cells')

# %% [markdown]
# ## 1.5 Seeing the hexagons on a map
#
# The helper below draws H3 cells on an interactive Folium map (renders inline in Colab).

# %%
def draw_cells(cells, color='#3186cc', zoom=9, m=None, center=None,
               fill_opacity=0.25, weight=1.5, label=''):
    cells = list(cells)
    if center is None:
        center = h3.cell_to_latlng(cells[0])
    if m is None:
        m = folium.Map(location=list(center), zoom_start=zoom, tiles='cartodbpositron')
    for c in cells:
        boundary = [[la, lo] for la, lo in h3.cell_to_boundary(c)]
        folium.Polygon(boundary, color=color, weight=weight, fill=True,
                       fill_color=color, fill_opacity=fill_opacity,
                       tooltip=f'{label}{c}').add_to(m)
    return m

# One resolution-8 hexagon over Amsterdam
print(draw_cells([h3.latlng_to_cell(lat, lon, 8)], zoom=13).to_string(index=False))

# %% [markdown]
# ### The nesting hierarchy, visualised
# The same point sits inside progressively smaller hexagons as resolution increases. Each finer hexagon nests inside the coarser one.

# %%
m = folium.Map(location=[lat, lon], zoom_start=8, tiles='cartodbpositron')
palette = {4: '#d73027', 5: '#fc8d59', 6: '#fee08b', 7: '#91cf60', 8: '#1a9850'}
for r, col in palette.items():
    draw_cells([h3.latlng_to_cell(lat, lon, r)], color=col, m=m,
               fill_opacity=0.12, label=f'res {r}: ')
folium.Marker([lat, lon], tooltip='Amsterdam').add_to(m)
print(m)

# %% [markdown]
# ### A catchment: `grid_disk` around a cell
# The red cell is the centre; the blue ring is everything within 2 rings — the kind of neighbourhood we treat as a station's catchment in Section 2.

# %%
center_cell = h3.latlng_to_cell(lat, lon, 8)
disk = h3.grid_disk(center_cell, 2)
m = draw_cells(disk, color='#3186cc', zoom=12)
draw_cells([center_cell], color='#e31a1c', m=m, fill_opacity=0.6)
print(m)

# %% [markdown]
# ## 1.6 Why hexagons (and why this matters for us)
#
# - **Uniform adjacency** — 6 equidistant neighbours make "nearness" consistent in every direction (no edge-vs-diagonal ambiguity of square grids).
# - **Multi-scale** — the parent/child hierarchy lets us aggregate at any zoom for free.
# - **Just an ID** — a cell is a short string, so binning and joining are plain **groupby / hash-join** operations that scale in Spark, BigQuery, Snowflake, Polars.
#
# These three properties are exactly what turn the expensive geospatial join in Section 2 into a cheap, scalable one.

# %% [markdown]
# # Section 2 — Business Demo: Did this car charge at this station?
#
# **ABC Corporation** runs ~20% of Europe's EV charging stations and wants 50% by 2030. To invest
# well it needs to know **where charging demand actually is** — including at its competitors' sites.
#
# ABC knows exactly what happens at **its own 80 stations**: it has the billing records. It knows
# **nothing** about the 250 competitor stations. But it does see Europe-wide telematics — every EV's
# position, speed, heading and state of charge.
#
# > **The question:** a car stopped somewhere for 45 minutes. Was it *charging at that station*, or
# > just parked near it?
#
# Answer that reliably and you can replay it against **every competitor station on the continent**
# and finally see their demand.
#
# ### Why this is a geospatial compute problem
#
# To ask "did this stop happen at that station?" you have to **pair every stop with every station** —
# a cross join. That is `stops × stations`, and it explodes exactly the way the Section 1 examples
# suggested. We will build it twice: once with a **haversine cross join**, once with **H3 as a
# blocking key**, and measure the difference.

# %% [markdown]
# ## 2.1 The data — four files
#
# | File | Grain | Role |
# |---|---|---|
# | `ev_pings.csv` | one telematics ping | **Raw signal** — position, speed, heading, state of charge |
# | `ev_vehicles.csv` | one vehicle | **Fleet master** — engine type, battery size, max charge rate |
# | `ev_stations.csv` | one charging station | ABC's 80 plus 250 competitor sites |
# | `ev_charging_sessions.csv` | one charging session | **ABC's internal records — the LABELS** |
#
# The sessions file is what makes supervised learning possible, and its limit is the whole problem:
# **it covers ABC's own chargers only.** Competitor charging happens in the telematics and leaves no
# trace in ABC's systems.
#
# The fleet master matters more than it looks. Charging time is physics — *energy needed ÷ the slower
# of the car's rate and the station's* — so a 105 kWh car gaining 40% sits far longer than a 9 kWh
# PHEV doing the same. Without knowing the car, a long dwell is ambiguous.
#
# > ⚠️ Every row is synthetic. Real city coordinates, invented everything else — see `data/README.md`.

# %%
DATA = "https://raw.githubusercontent.com/litandlatte/tanacloud.com/H3/labs/h3/data/"

pings    = pd.read_csv(DATA + "ev_pings.csv", parse_dates=["ts_utc"])
vehicles = pd.read_csv(DATA + "ev_vehicles.csv")
stations = pd.read_csv(DATA + "ev_stations.csv")
sessions = pd.read_csv(DATA + "ev_charging_sessions.csv",
                       parse_dates=["start_utc", "end_utc"])

abc        = stations[stations.is_abc].reset_index(drop=True)
competitor = stations[~stations.is_abc].reset_index(drop=True)

print(f"pings    : {len(pings):>7,}  ({pings.vehicle_id.nunique():,} vehicles, "
      f"{pings.heading_deg.isna().mean():.0%} missing heading)")
print(f"vehicles : {len(vehicles):>7,}  "
      f"({(vehicles.engine_type=='BEV').sum():,} BEV, "
      f"{(vehicles.engine_type=='PHEV').sum():,} PHEV)")
print(f"stations : {len(stations):>7,}  (ABC {len(abc)}, competitor {len(competitor)})")
print(f"sessions : {len(sessions):>7,}  <- ABC's own chargers ONLY")
print()
print("ABC knows this much about its own network:")
print(sessions.head(3).to_string(index=False))
print("\n...and nothing at all about the other 250 stations.")

# %% [markdown]
# ## 2.2 From pings to **stops**
#
# A single ping is not an event. Charging is something that happens over a *run of consecutive
# pings in the same place* — which is exactly the "how long did it sit there" signal we need.
#
# So collapse consecutive parked pings into one **stop**. A stop ends when the car moves off,
# jumps more than 200 m, or the feed goes quiet for over half an hour.
#
# Each stop carries the three things that will turn out to matter:
# **how long it lasted**, **how much charge it gained**, and **which way the car was facing.**

# %%
LOW_SPEED_KMH = 5.0     # <= this is "parked"
GAP_MIN       = 30      # a longer silence starts a new stop
JUMP_KM       = 0.2     # so does moving more than 200 m


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; broadcasts over numpy arrays."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


p = pings.sort_values(["vehicle_id", "ts_utc"]).reset_index(drop=True)
parked = p.speed_kmh <= LOW_SPEED_KMH

# A new stop begins wherever the run of parked pings is broken.
step_km = haversine_km(p.lat.shift().to_numpy(), p.lon.shift().to_numpy(),
                       p.lat.to_numpy(), p.lon.to_numpy())
new_stop = ((~parked)
            | (p.vehicle_id != p.vehicle_id.shift())
            | ((p.ts_utc - p.ts_utc.shift()).dt.total_seconds() / 60 > GAP_MIN)
            | (pd.Series(step_km) > JUMP_KM))
p["stop_id"] = new_stop.cumsum()

# Heading is a compass bearing, so it has to be averaged as a direction, not a number:
# the mean of 350 and 10 degrees is 0, not 180.
sp = p[parked].copy()
sp["hsin"] = np.sin(np.radians(sp.heading_deg))
sp["hcos"] = np.cos(np.radians(sp.heading_deg))

stops = sp.groupby("stop_id").agg(
    vehicle_id=("vehicle_id", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
    start=("ts_utc", "min"), end=("ts_utc", "max"), n_pings=("ping_id", "size"),
    soc_start=("soc_pct", "first"), soc_end=("soc_pct", "last"),
    hsin=("hsin", "mean"), hcos=("hcos", "mean")).reset_index()

stops = stops[stops.n_pings >= 2].reset_index(drop=True)       # a stop needs a run
stops["dwell_min"]   = (stops.end - stops.start).dt.total_seconds() / 60
stops["soc_delta"]   = stops.soc_end - stops.soc_start
stops["heading_deg"] = np.degrees(np.arctan2(stops.hsin, stops.hcos)) % 360

print(f"{len(pings):,} pings  ->  {len(stops):,} stops "
      f"({stops.n_pings.mean():.1f} pings each, median dwell {stops.dwell_min.median():.0f} min)")
stops[["stop_id", "vehicle_id", "lat", "lon", "n_pings", "dwell_min", "soc_delta",
       "heading_deg"]].head()

# %% [markdown]
# ## 2.3 The labels
#
# A stop is **positive** for a station if ABC's records show a session for that vehicle, at that
# station, overlapping that stop in time.
#
# One session must map to exactly **one** stop — the one it overlaps most. Allowing a session to
# claim several nearby stops silently manufactures false positives, and the damage is invisible
# until you plot the distances and find "charging" stops kilometres from the charger.

# %%
m = sessions.merge(stops[["stop_id", "vehicle_id", "start", "end"]], on="vehicle_id")

# Overlap in minutes between the session window and the stop window.
lo = np.maximum(m.start_utc.values.astype("datetime64[s]"),
                m.start.values.astype("datetime64[s]"))
hi = np.minimum(m.end_utc.values.astype("datetime64[s]"),
                m.end.values.astype("datetime64[s]"))
m["overlap_min"] = (hi - lo).astype("timedelta64[s]").astype(float) / 60

m = m[m.overlap_min > 0]
best = m.loc[m.groupby("session_id").overlap_min.idxmax()]      # one session -> one stop
truth = set(zip(best.stop_id, best.station_id))

print(f"{len(sessions):,} sessions -> {len(truth):,} labelled (stop, station) pairs")
print(f"on {best.stop_id.nunique():,} distinct stops "
      f"({best.stop_id.nunique() / len(stops):.1%} of all stops)")

# Sanity check worth doing every time: a labelled charging stop should be AT its station.
chk = best.merge(stations[["station_id", "lat", "lon"]], on="station_id") \
          .merge(stops[["stop_id", "lat", "lon"]], on="stop_id", suffixes=("_st", "_stop"))
d_m = haversine_km(chk.lat_stop, chk.lon_stop, chk.lat_st, chk.lon_st) * 1000
print(f"\nlabel sanity - distance from stop to its own station: "
      f"median {np.median(d_m):.0f} m, 99th pct {np.percentile(d_m, 99):.0f} m, "
      f"max {d_m.max():.0f} m")

# %% [markdown]
# ## 2.4 Which H3 resolution — and how wide a ring?
#
# Here the resolution stops being a guess. We have ground truth: cars we *know* were charging, and
# the stations they were plugged into. So ask the data directly — **at which resolution, and within
# how many rings, does a charging car sit relative to its station?**
#
# Two things are being chosen at once:
#
# - **Resolution** sets the grain. Too coarse and one cell sweeps in half a city; too fine and the
#   car and the charger land in different cells and the pair is lost.
# - **k**, the ring radius, sets the reach — and, crucially, **how graded the answer is.** This
#   matters because H3 is going to *replace* the distance calculation entirely: instead of metres,
#   the model gets `grid_distance` — how many rings apart the car and the station are.
#
# ⚠️ **A coarse cell with `k=1` cannot express proximity.** Every candidate is either 0 or 1 rings
# away — a yes/no, which throws away nearly everything distance was telling us. To get a graded
# signal out of H3 you need **finer cells and a wider ring**.

# %%
res_rows = []
for res in range(8, 13):
    stop_cell = [h3.latlng_to_cell(a, b, res) for a, b in zip(chk.lat_stop, chk.lon_stop)]
    st_cell   = [h3.latlng_to_cell(a, b, res) for a, b in zip(chk.lat_st,   chk.lon_st)]
    gd = np.array([h3.grid_distance(u, v) for u, v in zip(stop_cell, st_cell)])
    edge = h3.average_hexagon_edge_length(res, unit='m')
    for k in (1, 2, 3, 4):
        res_rows.append(dict(
            resolution=res, edge_m=round(edge, 1), k=k,
            reach_m=round(edge * 1.732 * k),          # centre-to-centre step is edge * sqrt(3)
            recall=(gd <= k).mean(),
            ring_levels=len(set(gd[gd <= k].tolist())),   # how graded the feature can be
            cells_per_station=3 * k * k + 3 * k + 1,
        ))

grid = pd.DataFrame(res_rows)
grid.style.format({"edge_m": "{:,.1f}", "reach_m": "{:,.0f}", "recall": "{:.1%}"}) \
    .hide(axis="index")

# %% [markdown]
# **Read the `within_k1` column.** Down to res 9 it stays essentially perfect, then starts to
# fall away: by res 11 a car and its charger are in the same neighbourhood of cells only ~93% of the
# time, and by res 12 barely two thirds.
#
# **We take res 9 with `k = 1`** — cells ~200 m on a side, a catchment of one cell plus its six
# neighbours. Fine enough to be a tiny haystack, coarse enough to keep essentially every needle.
#
# That number was **measured, not assumed** — and the measurement is only possible because ABC's
# session records told us which pairs were real.

# %% [markdown]
# **Read `ring_levels` next to `recall`.**
#
# - **res 9, k=1** — 100% recall, but only **2** ring levels. That is the binary indicator again.
# - **res 11, k=6** — beautifully graded, but recall has fallen to ~99% and each station now
#   occupies 127 cells in the index.
# - **res 10, k=4** — **100% recall, 5 ring levels, ~526 m reach.** Graded, complete, and its reach
#   matches the 500 m radius the haversine approach will use, so the two are genuinely comparable.
#
# That last point matters for the experiment: if the H3 path also changed the search radius, any
# difference in model quality would be unattributable.

# %%
BLOCK_RES  = 10     # chosen from the grid above
BLOCK_K    = 4      # a cell plus 4 rings -> grid_distance in 0..4
MAX_DIST_M = 500    # the equivalent radius for the haversine approach

_edge = h3.average_hexagon_edge_length(BLOCK_RES, unit='m')
print(f"H3        : res {BLOCK_RES}, k={BLOCK_K}  "
      f"({_edge:.0f} m edge, reach ~{_edge * 1.732 * BLOCK_K:.0f} m, "
      f"{3*BLOCK_K**2 + 3*BLOCK_K + 1} cells per station)")
print(f"Haversine : everything within {MAX_DIST_M} m")
print()
print("Both reach about the same distance - so the only thing that differs between the two")
print("approaches is HOW the pairs are found, and what the model is told about proximity.")

# %% [markdown]
# ## 2.5 Approach A — the haversine cross join
#
# The obvious way. Pair **every stop with every ABC station**, compute the distance for all of them,
# and keep the close ones.
#
# Watch the pair count. This is 24,000 stops against 80 stations — a rounding error next to ABC's
# real fleet, and it already runs to millions of distance calculations, **99.9% of which exist only
# to be thrown away.**

# %%
t0 = time.perf_counter()

stop_lat = stops.lat.to_numpy(); stop_lon = stops.lon.to_numpy()
abc_lat  = abc.lat.to_numpy();   abc_lon  = abc.lon.to_numpy()

# The cross join, in full: every stop against every station.
D = haversine_km(stop_lat[:, None], stop_lon[:, None],
                 abc_lat[None, :],  abc_lon[None, :]) * 1000      # metres
pairs_haversine = D.size

si, sj = np.where(D <= MAX_DIST_M)
cand_hav = pd.DataFrame({"stop_id": stops.stop_id.to_numpy()[si],
                         "station_id": abc.station_id.to_numpy()[sj],
                         "dist_m": D[si, sj]})
time_hav = time.perf_counter() - t0

print(f"pairs evaluated : {pairs_haversine:,}")
print(f"candidates kept : {len(cand_hav):,}  ({len(cand_hav)/pairs_haversine:.4%} of the work)")
print(f"wall clock      : {time_hav:.3f} s")

# %% [markdown]
# ## 2.6 Approach B — H3 as the blocking key **and** the distance measure
#
# Two changes at once, and both are the point:
#
# 1. **Candidates come from a lookup, not a scan.** Index every station into its cell and the ring
#    around it. A stop's candidates are whatever sits in *its* cell — a dictionary hit.
# 2. **`grid_distance` replaces the metre distance entirely.** How many rings apart the car and the
#    station are *is* the proximity feature.
#
# So in this path **no haversine is ever computed** — not for filtering, not for the model. There is
# no `sin`, no `cos`, no `sqrt` anywhere. Just integer cell arithmetic.
#
# The pairs that never share a neighbourhood are not computed and discarded. **They are never
# created.**

# %%
t0 = time.perf_counter()

# Index the stations once: each claims its cell and every cell within BLOCK_K rings.
station_index = {}
for sid, la, lo in zip(abc.station_id, abc.lat, abc.lon):
    home = h3.latlng_to_cell(la, lo, BLOCK_RES)
    for cell in h3.grid_disk(home, BLOCK_K):
        station_index.setdefault(cell, []).append((sid, home))

# Look each stop up. No distances, no scanning - a dict hit, then integer ring arithmetic.
rows, lookups = [], 0
for sid_stop, la, lo in zip(stops.stop_id, stop_lat, stop_lon):
    cell = h3.latlng_to_cell(la, lo, BLOCK_RES)
    hits = station_index.get(cell)
    if hits:
        lookups += len(hits)
        for sid_st, home in hits:
            rows.append((sid_stop, sid_st, h3.grid_distance(cell, home)))

cand_h3 = pd.DataFrame(rows, columns=["stop_id", "station_id", "ring_dist"])
time_h3 = time.perf_counter() - t0

print(f"pairs materialised : {lookups:,}   (haversine evaluated {pairs_haversine:,})")
print(f"candidates kept    : {len(cand_h3):,}")
print(f"wall clock         : {time_h3:.3f} s")
print()
print(f"pairs avoided      : {pairs_haversine - lookups:,} "
      f"({1 - lookups/pairs_haversine:.2%} of the cross join never happened)")
print(f"reduction          : {pairs_haversine / max(lookups, 1):,.0f}x fewer pairs")
print()
print("ring_dist distribution across candidates:")
print(cand_h3.ring_dist.value_counts().sort_index().to_string())

# %% [markdown]
# ### Did the shortcut cost us anything? — **blocking recall**
#
# Speed is worthless if the filter quietly drops real charging sessions. This is the number that
# must be checked, and it is the one people skip.
#
# Of the labelled charging pairs, how many survive each approach?

# %%
def recall_of(cand):
    got = set(zip(cand.stop_id, cand.station_id))
    return len(truth & got) / len(truth)

r_hav, r_h3 = recall_of(cand_hav), recall_of(cand_h3)
print(f"blocking recall  haversine : {r_hav:.2%}   ({len(cand_hav):,} candidates)")
print(f"blocking recall  H3        : {r_h3:.2%}   ({len(cand_h3):,} candidates)")
print()
if r_h3 >= r_hav - 0.005:
    print("H3 keeps the same needles, from a far smaller haystack.")
else:
    print(f"H3 drops {(r_hav - r_h3):.2%} of true pairs - the price of the speed-up. "
          "Coarsen the resolution or raise k to buy it back.")

# %% [markdown]
# ## 2.7 Features, and **two** models
#
# Everything except the spatial feature is shared, so the comparison isolates exactly one thing:
# *how the model is told where the car was.*
#
# | Shared feature | Why it might indicate charging |
# |---|---|
# | `dwell_min`, `n_pings` | charging takes time — a run of pings in one place |
# | `soc_delta` | the battery **gained charge** during the stop |
# | `align` | a plugged-in car sits square in its bay — `cos` of heading vs bay bearing |
# | `battery_kwh`, `max_charge_kw`, `is_bev` | how long *should* this car take to gain that much? |
# | `station_kw` | a 350 kW site fills a car faster than a 50 kW one |
# | `hour` | demand varies across the day |
#
# | Model | Spatial feature |
# |---|---|
# | **A — haversine** | `dist_m`, exact metres |
# | **B — H3** | `ring_dist`, integer rings apart |
#
# **Split by vehicle, never by row.** One car produces many stops; letting two of them straddle the
# train/test boundary leaks.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

SHARED = ["dwell_min", "n_pings", "soc_delta", "align",
          "battery_kwh", "max_charge_kw", "is_bev", "station_kw", "hour"]


def build_features(cand, spatial):
    """Attach stop, vehicle and station attributes plus the label to a candidate set."""
    f = (cand.merge(stops[["stop_id", "vehicle_id", "n_pings", "dwell_min",
                           "soc_delta", "heading_deg", "start"]], on="stop_id")
             .merge(vehicles[["vehicle_id", "engine_type", "battery_kwh", "max_charge_kw"]],
                    on="vehicle_id")
             .merge(stations[["station_id", "bay_bearing_deg", "max_power_kw"]],
                    on="station_id"))
    # Alignment: +1 = car facing along the bay, -1 = facing against it. NaN where the tracker
    # reported no heading, which the gradient booster handles natively.
    f["align"]      = np.cos(np.radians(f.heading_deg - f.bay_bearing_deg))
    f["is_bev"]     = (f.engine_type == "BEV").astype(int)
    f["station_kw"] = f.max_power_kw
    f["hour"]       = f.start.dt.hour
    f["y"] = [1 if k in truth else 0 for k in zip(f.stop_id, f.station_id)]
    return f, SHARED + [spatial]


def fit_and_score(cand, spatial, label):
    data, feats = build_features(cand, spatial)
    tr_ix, te_ix = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=7)
                        .split(data, groups=data.vehicle_id))
    tr, te = data.iloc[tr_ix], data.iloc[te_ix]
    mdl = HistGradientBoostingClassifier(max_iter=250, random_state=7).fit(tr[feats], tr.y)
    pr = mdl.predict_proba(te[feats])[:, 1]
    ap = average_precision_score(te.y, pr)
    print(f"{label:<26} train {len(tr):>6,} / test {len(te):>6,}   "
          f"PR-AUC {ap:.4f}   ROC-AUC {roc_auc_score(te.y, pr):.4f}")
    return dict(model=mdl, feats=feats, data=data, tr=tr, te=te, proba=pr, ap=ap)


A = fit_and_score(cand_hav, "dist_m",    "A  haversine metres")
Bm = fit_and_score(cand_h3,  "ring_dist", "B  H3 ring distance")

print()
print(f"Replacing metres with rings gives up {A['ap'] - Bm['ap']:.4f} PR-AUC "
      f"({(A['ap'] - Bm['ap']) / A['ap']:.1%} relative).")
print("That is the price of never computing a distance. Whether it is worth paying is a")
print("business question, not a technical one - and now it is a measured number.")
print()
print(classification_report(Bm["te"].y, Bm["proba"] >= 0.5,
                            target_names=["not charging", "charging"], digits=3))

# %% [markdown]
# ### "Why not just use a distance threshold?"
#
# The fairest challenge in the room, and worth answering with a number rather than an opinion.
# Here is what each feature achieves **on its own**:

# %%
tr, te, feats = Bm["tr"], Bm["te"], Bm["feats"]
solo = []
for f in feats:
    m1 = HistGradientBoostingClassifier(max_iter=120, random_state=7).fit(tr[[f]], tr.y)
    solo.append((f, average_precision_score(te.y, m1.predict_proba(te[[f]])[:, 1])))
solo.append(("ALL COMBINED", Bm["ap"]))

solo_df = pd.DataFrame(solo, columns=["feature", "PR_AUC"]).sort_values("PR_AUC")
for f, s in solo_df.itertuples(index=False):
    print(f"  {f:<14} {s:.3f}  {'#' * int(s * 50)}")

# %% [markdown]
# **Distance alone is the best single feature and it is not close to sufficient.** The lift comes
# from combining *where* the car was with *how long it stayed*, *whether the battery filled* and
# *which way it faced* — and no single threshold expresses that.
#
# Note also what the confusers did: a car charging at a **competitor station 200 m away** has a
# rising SoC, a long dwell and a bay-aligned heading. Only the geometry separates it. And a car on a
# home charger has the SoC rise with no station at all. **Neither signal alone is safe.**

# %% [markdown]
# ## 2.8 Inference — now score the competitors
#
# The model has only ever seen ABC's own stations. Point it at **all 330** and the 250 blind spots
# light up: every stop is paired with every station via the same H3 index, scored, and the
# high-confidence hits counted per station.
#
# This is the number ABC could never buy: **how busy is my competitor's site?**

# %%
t0 = time.perf_counter()

# Same H3 index, now over every station rather than only ABC's.
all_index = {}
for sid, la, lo in zip(stations.station_id, stations.lat, stations.lon):
    home = h3.latlng_to_cell(la, lo, BLOCK_RES)
    for cell in h3.grid_disk(home, BLOCK_K):
        all_index.setdefault(cell, []).append((sid, home))

rows = []
for sid_stop, la, lo in zip(stops.stop_id, stop_lat, stop_lon):
    cell = h3.latlng_to_cell(la, lo, BLOCK_RES)
    hits = all_index.get(cell)
    if hits:
        for sid_st, home in hits:
            rows.append((sid_stop, sid_st, h3.grid_distance(cell, home)))

inf = pd.DataFrame(rows, columns=["stop_id", "station_id", "ring_dist"])
X, _ = build_features(inf, "ring_dist")
X["p_charging"] = Bm["model"].predict_proba(X[Bm["feats"]])[:, 1]
time_infer = time.perf_counter() - t0

full_cross_join = len(stops) * len(stations)
print(f"full cross join would have been : {full_cross_join:,} pairs")
print(f"H3 actually scored              : {len(X):,} pairs   in {time_infer:.1f} s")
print(f"                                  ({full_cross_join / len(X):,.0f}x smaller)")

# %%
THRESHOLD = 0.5
hits = X[X.p_charging >= THRESHOLD]
demand = (hits.groupby("station_id").size().rename("predicted_sessions").reset_index()
          .merge(stations, on="station_id"))

# --- credibility check, done honestly -------------------------------------------------
# Compare predicted against actual session counts at ABC's own stations, but ONLY for the
# vehicles held out of training. Counting every stop would score the model on data it was
# fitted to and flatter it badly.
test_vehicles = set(Bm["te"].vehicle_id)
X_held = X[X.vehicle_id.isin(test_vehicles)]
pred_held = (X_held[X_held.p_charging >= THRESHOLD]
             .groupby("station_id").size().rename("predicted"))
act_held = (best[best.vehicle_id.isin(test_vehicles)]
            .groupby("station_id").size().rename("actual"))
chk2 = (pd.concat([pred_held, act_held], axis=1).fillna(0)
        .loc[lambda d: d.index.isin(set(abc.station_id))])
r = np.corrcoef(chk2.predicted, chk2.actual)[0, 1]

print(f"Held-out vehicles only, ABC's own {len(chk2)} stations:")
print(f"  predicted vs actual session counts correlate {r:.3f}")
print(f"  total predicted {int(chk2.predicted.sum())} vs actual {int(chk2.actual.sum())}")
print("The same method, measured on vehicles the model never saw, at stations where the")
print("true answer is known. ABC reports ACTUALS for its own sites and PREDICTIONS for")
print("everyone else's - this is what earns the right to do that.\n")

top = (demand[~demand.is_abc]
       .nlargest(10, "predicted_sessions")
       [["station_id", "operator", "city", "country", "n_chargers",
         "max_power_kw", "predicted_sessions"]])
print("Busiest COMPETITOR stations - demand ABC has never been able to see:")
print(top.reset_index(drop=True).to_string(index=False))

# %% [markdown]
# ## 2.9 What this costs at ABC's real scale
#
# The demo is 3,000 vehicles and 330 stations. ABC operates across a continent. Leadership does not
# ask "how many seconds" — it asks **"can we afford to run this every day?"**
#
# Scaling rule, and why the gap *widens* rather than holding:
#
# - the **cross join** grows with `stops × stations` — every new station multiplies every stop;
# - the **H3 path** grows with `stops × stations within reach`, and that second term is capped by
#   **local density**, not by the size of the network.
#
# Add stations in new cities and the cross join grows; the H3 candidate count barely moves.
#
# > ⚠️ **The honest caveat, and it cuts against H3 here.** In this notebook the haversine path is
# > *vectorised numpy* and the H3 path is a *Python loop*, so H3 is far slower **per pair** even
# > while touching 383× fewer of them. In production both are the same primitive — a hash join
# > versus a nested-loop join in Spark, BigQuery or Snowflake — so the cost model below assumes a
# > **common per-pair rate**. The pair counts and the data volumes are arithmetic and do not depend
# > on that assumption.

# %%
# --- assumptions, all adjustable -----------------------------------------------------
PROD_VEHICLES   = 2_000_000     # ABC's addressable fleet
PROD_STATIONS   = 50_000        # public charging sites across Europe
NEARBY_STATIONS = 12            # stations within reach of a typical urban stop
BYTES_PER_PAIR  = 12            # two ids + one float, before any features are attached
PAIRS_PER_SEC   = 20_000_000    # one core in a columnar engine; same rate for both paths
CORE_RATE_EUR   = 0.05          # EUR per core-hour

stops_per_vehicle = len(stops) / pings.vehicle_id.nunique()
prod_stops = PROD_VEHICLES * stops_per_vehicle

pairs_hav = prod_stops * PROD_STATIONS
pairs_h3  = prod_stops * NEARBY_STATIONS


def human_bytes(n):
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if n >= size:
            return f"{n/size:,.1f} {unit}"
    return f"{n:,.0f} B"


def row(label, pairs):
    hrs = pairs / PAIRS_PER_SEC / 3600
    print(f"{label:20}{pairs:>19,.0f}{human_bytes(pairs*BYTES_PER_PAIR):>16}"
          f"{hrs:>13,.2f}{hrs*CORE_RATE_EUR:>13,.3f}")


print(f"Assumed production scale: {PROD_VEHICLES:,} vehicles -> {prod_stops:,.0f} stops/week "
      f"against {PROD_STATIONS:,} stations\n")
print(f"{'':20}{'pairs':>19}{'intermediate':>16}{'core-hours':>13}{'EUR/run':>13}")
print("-" * 81)
row("A  haversine", pairs_hav)
row("B  H3", pairs_h3)
print("-" * 81)
print(f"{'reduction':20}{pairs_hav/pairs_h3:>18,.0f}x{'':>16}"
      f"{pairs_hav/pairs_h3:>12,.0f}x{pairs_hav/pairs_h3:>12,.0f}x")
print()
print(f"Run daily: EUR {pairs_hav/PAIRS_PER_SEC/3600*CORE_RATE_EUR*365:,.0f} a year against "
      f"EUR {pairs_h3/PAIRS_PER_SEC/3600*CORE_RATE_EUR*365:,.2f}.")
print()
print("But the euros are not the reason this fails. Look at the intermediate column: the cross")
print("join materialises several TERABYTES of pairs that exist only to be filtered away, and it")
print("does it on every run. That is a cluster, a shuffle, and a nightly batch window. The H3")
print("side is a few gigabytes - a job that fits on one machine.")
print()
print("The difference is not 'slow versus fast'. It is 'infrastructure project' versus 'script'.")

# %% [markdown]
# ## 2.10 Scoreboard

# %%
print("=" * 78)
print("SCOREBOARD".center(78))
print("=" * 78)
print(f"{'':34}{'A haversine':>21}{'B  H3':>21}")
print("-" * 78)
print(f"{'pairs materialised':34}{pairs_haversine:>21,}{lookups:>21,}")
print(f"{'candidates kept':34}{len(cand_hav):>21,}{len(cand_h3):>21,}")
print(f"{'blocking recall':34}{r_hav:>20.1%}{r_h3:>21.1%}")
print(f"{'trigonometry computed':34}{'1.9M haversines':>21}{'none':>21}")
print(f"{'spatial feature':34}{'metres':>21}{'rings (0-' + str(BLOCK_K) + ')':>21}")
print(f"{'model PR-AUC':34}{A['ap']:>21.4f}{Bm['ap']:>21.4f}")
print("-" * 78)
print(f"{'pairs avoided by H3':34}{'':>21}{pairs_haversine - lookups:>21,}")
print(f"{'reduction':34}{'':>21}{f'{pairs_haversine / max(lookups,1):,.0f}x':>21}")
print(f"{'accuracy given up':34}{'':>21}{f"{A['ap'] - Bm['ap']:+.4f}":>21}")
print("=" * 78)
print(f"Best single feature ({solo_df.iloc[-2].feature}) reaches only "
      f"{solo_df.iloc[-2].PR_AUC:.3f} - the combination is the model.")
print()
print("The pair counts are arithmetic: identical on every machine, forever.")
print("Wall-clock is not - it depends on the laptop, so it is not the number to quote.")
print()
print("H3 did not make the distance calculation faster.")
print("It removed the distance calculation.")

# %% [markdown]
# ## 2.11 The answer on a map
#
# ABC's own stations in blue; the competitor sites we now estimate to be busiest in red, sized by
# predicted sessions. Every red circle is demand that was invisible an hour ago.

# %%
m = folium.Map(location=[stations.lat.mean(), stations.lon.mean()],
               zoom_start=4, tiles='cartodbpositron')

for s in abc.itertuples(index=False):
    folium.CircleMarker([s.lat, s.lon], radius=3, color='#1f78b4', fill=True,
                        fill_opacity=0.6,
                        tooltip=f'ABC {s.station_id} | {s.city}').add_to(m)

for s in top.itertuples(index=False):
    st = stations[stations.station_id == s.station_id].iloc[0]
    folium.CircleMarker([st.lat, st.lon],
                        radius=6 + s.predicted_sessions ** 0.5, color='#e31a1c', fill=True,
                        fill_opacity=0.85,
                        tooltip=(f'{s.operator} {s.station_id} | {s.city} | '
                                 f'~{s.predicted_sessions} sessions/week')).add_to(m)
print(m)
