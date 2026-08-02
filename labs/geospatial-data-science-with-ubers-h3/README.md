# Geospatial Data Science with Uber's H3

Two Colab notebooks. The lab answers one question twice — **what is a block of houses worth,
given what's around it?** — first the traditional way, then with H3. The visual guide is the part
the lab skips: what the sixteen resolutions actually look like, and how to choose one.

Both run top to bottom in Google Colab. Nothing to install beyond one line, no API key, no login.
Every cell opens with a comment block explaining what it does and why, so the notebooks stand on
their own after the session.

| File | What it is |
|---|---|
| `H3_Hands_On_Lab.ipynb` | **The lab** (~20 min). Clean, no stored outputs — run it live. |
| `H3_Hands_On_Lab_prerun.ipynb` | Same notebook with every output saved. Reference, and a fallback if the network dies. |
| `H3_Resolutions_Visual_Guide.ipynb` | **The companion** — all 16 resolutions, with maps. |
| `H3_Resolutions_Visual_Guide_prerun.ipynb` | Same, with outputs saved. |
| `data/housing.csv` | Mirror of the dataset, so a dead upstream URL can't take either notebook down. |

## The data

**California housing** — [`ageron/handson-ml2`](https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv),
which is Aurélien Géron's copy of the **StatLib California housing dataset** from Pace & Barry,
*"Sparse Spatial Autoregressions"* (Statistics & Probability Letters, 1997), built from the
**1990 US Census**.

**20,640 rows**, one per **census block group** — the smallest area the US Census publishes sample
data for, typically 600–3,000 people. Columns: `longitude`, `latitude`, `housing_median_age`,
`total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` (in units of
$10,000), `median_house_value` (the target), `ocean_proximity`.

Three quirks the lab checks on screen rather than assuming: `median_house_value` is **capped at
$500,001** (965 rows, 4.7%), `housing_median_age` is capped at 52 (1,273 rows), and
`total_bedrooms` is missing for 207 rows. The cap is why the RMSE floor is higher than it looks
like it should be.

## What the lab shows

- **Round 1 — traditional.** Hand-written haversine. 426 million pairs; a plain Python loop
  extrapolates to ~5 minutes, vectorised numpy gets it to ~7 s. Then the classic no-tools
  aggregation: round the coordinates to 0.01° and group.
- **Round 2 — H3.** One line indexes all 20,640 rows in 0.015 s. Density becomes a `groupby`;
  proximity becomes `grid_disk`. The **hybrid** — H3 filters the candidates, haversine measures
  the survivors — returns *bit-identical* answers about 12× faster, and the gap widens with n
  because brute force is O(n²) and this is roughly linear.
- **Round 3 — does it help?** Neighbourhood features cut model RMSE from **$65,916 to $48,399
  (−26.6%)**. A sweep across resolutions 4–10 produces a clean inverted U peaking at **res 7**,
  which earns the rule of thumb: *pick the finest resolution whose neighbourhood still holds
  enough members for a stable statistic — and watch coverage, not cell size.*

## What the visual guide shows

- **All 16 resolutions** with edge length, area, global cell count and a real-world size for each.
  Each step divides the area by about 7, so choosing a resolution is choosing an order of
  magnitude.
- **One place, five grids** — the same 10 km box around Perungudi, Chennai, at res 5 through 9,
  drawn at identical extent so the panels show granularity rather than zoom.
- **How resolutions nest, and where that breaks.** Seven hexagons cannot tile one larger hexagon,
  and you can see the children spilling over the parent's edge. Measured: **6.8% of a parent's
  area belongs to cells that are not its children** — which is why you must never roll counts up
  the hierarchy, and should re-aggregate from raw points instead.
- **A resolution diagnostic** to run on your own data — coordinates in, candidate resolution out,
  no model required.

## Two things these notebooks are careful about

- **Hexagons aren't sold on equal area.** Within one state, H3's area spread (1.115×) is close to
  a rounded lat/lon grid's (1.135×). The real argument is neighbours: H3 has six, all the same
  step away (spread **1.034×**), against a square grid's eight at four different distances
  (spread **1.596×**).
- **No leakage.** Neighbourhood statistics are built from training rows only, and every row is
  excluded from its own neighbourhood. Both are called out on screen.

Written for h3-py **4.5.0**. Most tutorials online are still v3 and will fail — Appendix B of the
lab has the rename table.
