# Geospatial Data Science with Uber's H3 — hands-on lab

A 20-minute lab that answers one question twice: **what is a block of houses worth, given what's
around it?** First the traditional way (haversine over every pair), then with H3.

## Run it

Open `H3_Hands_On_Lab.ipynb` in Google Colab and run the cells in order. Nothing to install
beyond one line, no API key, no login.

| File | What it is |
|---|---|
| `H3_Hands_On_Lab.ipynb` | The lab. Clean, no stored outputs — run it live. |
| `H3_Hands_On_Lab_prerun.ipynb` | Same notebook with every output saved. Reference, and a fallback if the network dies. |
| `data/housing.csv` | Mirror of the dataset, so a dead upstream URL can't take the lab down. |

## The data

California housing — 20,640 census block groups, each with a latitude, a longitude, six ordinary
numeric columns, and a target: `median_house_value`. Loaded from
[ageron/handson-ml2](https://github.com/ageron/handson-ml2), with the mirror above as a fallback.

## What it shows

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

## Two things the lab is careful about

- **Hexagons aren't sold on equal area.** Within one state, H3's area spread (1.115×) is close to
  a rounded lat/lon grid's (1.135×). The real argument is neighbours: H3 has six, all the same
  step away (spread **1.034×**), against a square grid's eight at four different distances
  (spread **1.596×**).
- **No leakage.** Neighbourhood statistics are built from training rows only, and every row is
  excluded from its own neighbourhood. Both are called out on screen.

Written for h3-py **4.5.0**. Most tutorials online are still v3 and will fail — Appendix B has the
rename table.
