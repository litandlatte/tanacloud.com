# H3 for Data Science — Detecting EV Charging to See a Competitor's Demand (ABC Corp)

**Format:** hands-on lab inside a 50-minute session (the notebook itself is the closing ~20 minutes)
**Audience:** Data Scientists, Data Engineers, Analytics leads
**Goal:** Show how **Uber H3** turns an intractable "compare every point to every point" spatial
join into a hash lookup — on a real, high-value decision: **where should ABC Corp build its next
EV charging stations across Europe?**

> Every number in this document was taken from an actual end-to-end run of the notebook, not from
> memory. The data is seeded and the models use a fixed `random_state`, so these are the figures
> you will see on screen.

---

## 1. The business problem

**ABC Corporation** operates **~20% of Europe's EV charging stations** and wants **50% by 2030**.
To invest well it must know where charging demand actually is — including **at its competitors'
sites**, which is where the market share it wants currently sits.

ABC knows exactly what happens at its **own 80 stations**: it has the billing records. It knows
**nothing** about the **250 competitor stations**. But it does see Europe-wide telematics — every
EV's position, speed, heading and state of charge.

> **The question the model answers:** a car stopped somewhere for 45 minutes. Was it **charging at
> that station**, or just **parked near it**?

Answer that reliably at ABC's own sites, where the truth is known, and you can replay it against
every competitor station on the continent.

### The data

| Dataset | File | Role |
|---|---|---|
| EV telematics | `data/ev_pings.csv` | **Raw signal** — 145,729 pings, 3,000 vehicles, one day |
| Fleet master | `data/ev_vehicles.csv` | 3,000 vehicles — battery size, charge rate, BEV vs PHEV |
| Charging stations | `data/ev_stations.csv` | 80 ABC + 250 competitor, 21 countries |
| ABC's charging sessions | `data/ev_charging_sessions.csv` | **The labels — ABC's stations only** |

The last file's limit *is* the business problem: **968 sessions** across ABC's 80 stations, and
**nothing, ever, for the other 250**. Full column definitions and the synthetic-data disclosure
are in `data/README.md`.

**The data covers one day** (Mon 8 June 2026), so every rate in the lab is a **daily** one.

---

## 2. Why this is a compute problem

To ask "did this stop happen at that station?" you must **pair every stop with every station** —
a cross join.

In the demo that is **24,089 stops × 80 ABC stations = 1,927,120 pairs**, of which **0.26%**
survive a 500 m filter. Every one of the rest is a distance computed purely in order to be thrown
away. At ABC's real scale — millions of vehicles, tens of thousands of candidate sites — the
cross join simply cannot be built.

*"The problem isn't haversine. It's calling it on pairs that are obviously irrelevant."*

---

## 3. How we model it

A **binary classification** at the grain of a **(stop, station) pair**:

- **Stop** — a run of consecutive parked pings in one place. Carries how long the car stayed, how
  much charge it gained, and which way it faced. 145,729 pings collapse to **24,089 stops**.
- **Label** — 1 if ABC's session records show that vehicle charging at that station over that
  window. Each session claims exactly **one** stop, the one it overlaps most.
- **Train** on ABC's 80 stations → **predict** across all 330 → **count** confident hits per
  competitor station.

### Features

Nine features are shared by both models; only the **spatial** one differs.

| Feature | Why it might indicate charging |
|---|---|
| `dwell_min`, `n_pings` | charging takes time — a run of pings in one place |
| `soc_delta` | the battery **gained charge** during the stop |
| `align` | a plugged-in car sits square in its bay — `cos` of heading vs bay bearing |
| `battery_kwh`, `max_charge_kw`, `is_bev` | how long *should* this car take to gain that much? |
| `station_kw` | a 350 kW site fills a car faster than a 50 kW one |
| `hour` | demand varies across the day |
| **spatial** | **model A: `dist_m` (metres) · model B: `ring_dist` (H3 rings)** |

**Split by vehicle, never by row.** One car produces many stops; letting two of them straddle the
train/test boundary leaks.

---

## 4. Choosing the resolution — measured, not guessed

We have ground truth: cars we *know* were charging, and the stations they were plugged into. So
ask the data at which resolution, and within how many rings, a charging car lands relative to its
station.

**Two things are chosen at once, and this is the part people get wrong:**

- **Resolution** sets the grain. Too coarse and one cell sweeps in half a city; too fine and the
  car and the charger land in different cells and the pair is lost.
- **k**, the ring radius, sets the reach — and **how graded the answer is.** This matters because
  H3 *replaces* the distance calculation: the model is given `grid_distance`, not metres.

⚠️ **A coarse cell with `k=1` cannot express proximity.** Every candidate is 0 or 1 rings away — a
yes/no, which throws away nearly everything distance was telling us.

| Choice | Recall | Ring levels | Reach | Verdict |
|---|---:|---:|---:|---|
| res 9, k=1 | 100% | **2** | ~350 m | Complete, but the feature is binary |
| res 11, k=6 | ~99% | 7 | ~300 m | Graded, but recall slips and 127 cells per station |
| **res 10, k=4** | **100%** | **5** | **~526 m** | **Chosen** — graded, complete, 61 cells per station |

**We take res 10 with k = 4.** Its ~526 m reach deliberately matches the 500 m radius the
haversine path uses, so the two approaches are genuinely comparable — if the H3 path also changed
the search radius, any difference in model quality afterwards would be unattributable.

**This is the honest way to pick a resolution**, and it is only possible because the session
records tell you which pairs were real.

---

## 5. The two implementations

**A — haversine cross join.** Materialise all 1,927,120 pairs, compute every distance, keep those
under 500 m.

**B — H3 as blocking key *and* distance measure.** Index each station into its cell and the
`grid_disk(k=4)` ring around it. A stop's candidates are whatever sits in its cell — a dictionary
lookup. **No haversine is computed anywhere in this path**: no `sin`, no `cos`, no `sqrt`. The
other pairs are not computed and discarded; **they are never created at all.**

### Results

| | Haversine | H3 |
|---|---:|---:|
| Pairs materialised | 1,927,120 | **5,027** |
| Candidates kept | 5,018 | 5,027 |
| **Blocking recall** | **100.0%** | **100.0%** |
| Trigonometry computed | 1.9M haversines | **none** |

**383× fewer pairs, with no true pairs lost at all.** At inference across all 330 stations the
full cross join would be **7,949,370 pairs**; H3 scores **18,718** — **425× smaller**.

> **Quote the pair count, not the clock.** 383× is arithmetic and identical on every machine.
> Wall-clock depends on the laptop, and a slow Colab will contradict any timing you put on a slide.

**Always report blocking recall.** A filter that is fast because it silently drops real matches is
not an optimisation, it is a bug. Speed without this number is meaningless.

---

## 6. Model results

Held out **by vehicle**, the same classifier on the same features, differing only in how it is
told where the car was:

| Model | Spatial feature | PR-AUC | ROC-AUC |
|---|---|---:|---:|
| **A** | `dist_m`, exact metres | **0.978** | 0.989 |
| **B** | `ring_dist`, integer rings | **0.960** | 0.985 |

At a 0.5 threshold model B scores precision **0.906** / recall **0.851** on the charging class.

### Is that gap real? — checked across 10 splits

A single split puts only ~300 charging stops in the test set, so one 0.018 difference proves
little. Refitting both models on **10 different vehicle splits**:

| | Mean PR-AUC |
|---|---:|
| A — metres | 0.983 |
| B — rings | 0.956 |
| **Gap** | **+0.0264 ± 0.0120** (1 sd) |

95% interval **[+0.003, +0.050]**, and **A beat B on 10 splits out of 10.** The interval excludes
zero, so the effect is real.

> ⚠️ **Quote 0.026, not 0.018.** The headline split (`random_state=7`) happens to land at the
> optimistic end of the range. Dropping the distance calculation costs about **0.026 PR-AUC** —
> roughly 2.7% relative. Earlier versions of this document quoted 0.018, which understated it.

Whether that is worth paying is a business question, not a technical one. The point is that it is
now a *measured* number, with an error bar, rather than an assumption.

### "Why not just use a distance threshold?"

The fairest challenge in the room. Answer it with the number — PR-AUC from each feature **alone**:

| Feature alone | PR-AUC |
|---|---:|
| `soc_delta` | 0.470 |
| `ring_dist` | 0.443 |
| `dwell_min` / `n_pings` | 0.375 |
| `align` | 0.281 |
| **all combined** | **0.960** |

**The best single feature is not the spatial one** — *the battery filled up* narrowly beats *the
car was near a charger*, and neither comes close on its own. The lift comes from combining *where*
the car was with *how long it stayed*, *whether the battery filled* and *which way it faced*.

The dataset is built to make that true. **40% of competitor stations sit within 100–400 m of an
ABC station**, so a car charging next door shows a rising SoC, a long dwell and a bay-aligned
heading — only geometry separates it. And cars on home chargers show the SoC rise with no station
at all. Neither signal alone is safe.

### The deliverable

Predicted **daily** sessions at every competitor station — demand ABC could never buy.

**Credibility check:** run the same method on **held-out vehicles** at **ABC's own** stations,
where the answer is known. Predicted vs actual session counts correlate **0.904** (278 predicted
against 296 actual, across 73 stations). Do this check on held-out vehicles only; scoring every
stop includes the ones the model trained on and inflates the correlation to ~0.998, which proves
nothing.

---

## 7. Run sheet — the notebook's ~20 minutes

The full session is 50 minutes; the slides carry the first four sections and the notebook is the
last one. Within that slot:

| Time | Segment | Notebook |
|-----:|---------|----------|
| 0–4 | H3 fundamentals — hexagons, resolutions, `grid_disk` | §1 (skim; the Chennai map does the work) |
| 4–7 | Pings → stops → labels, and the label sanity check | §2.1–2.3 |
| 7–10 | Choosing resolution and ring width from ground truth | §2.4 |
| 10–14 | Both approaches; pair counts and blocking recall | §2.5–2.6 |
| 14–17 | The two models, and why not a threshold | §2.7 |
| 17–20 | Score the competitors, cost at scale, the map | §2.8–2.11 |

**Runtime is not the constraint** — the notebook computes in well under a minute end to end. The
two things that actually take time are the `pip install` at the top and the ~9.9 MB data download
in §2.1. Run the install cell before the audience is watching.

---

## 8. Where else this applies

The pattern — **"pair a huge set of points against another set, but only where they are near
each other"** — is everywhere:

- **Entity resolution across space** — matching deliveries to addresses, claims to properties.
- **Ride-hailing** — riders to drivers (H3's origin at Uber).
- **Retail** — footfall attribution to stores.
- **Telecom** — devices to cell sites.
- **Fraud** — co-location of cards and terminals.

**Transferable lesson:** *don't compute exact distance for every pair. Bin space with H3, use the
cell as a join key, and compute exact distance only for the pairs that survive — then measure what
the filter cost you.*

---

## 9. Running it

In the session, use Colab — one click, nothing to install:

```
https://colab.research.google.com/github/litandlatte/tanacloud.com/blob/H3/labs/h3/H3_EV_Demand_Demo.ipynb
```

Locally (macOS / Linux):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

On Windows the activation line is `.\.venv\Scripts\Activate.ps1`; everything else is the same.

**Files**

- `H3_EV_Demand_Demo.ipynb` — **the lab.** Section 1 is H3 fundamentals with interactive maps;
  Section 2 is the business demo above. **Ships with all outputs saved**, so it can be presented
  from even if the network fails.
- `demo.py` — Section 2 only, headless, no folium.
- `H3_EV_Demand_Demo.py` — the whole notebook as a percent-format script.
- `data/` — four CSVs, 200-row samples, data dictionary, `datapackage.json`, seeded generator.
- `requirements.txt` · `README.md` (this guide).

**🔑 Both `.py` files are generated from the notebook — edit the notebook, never them.** They are
regenerated with `export_py.py`; hand-editing them is what let a stale paragraph survive in three
files at once.

**Every code cell opens with a `WHAT / WHY / OUT` header** describing what it does, why it is
there, and what should appear on screen. If a cell's output disagrees with its `OUT` line,
something upstream has changed.

**On the data:** entirely synthetic. Only the city coordinates are real, so the maps are
recognisable; ABC Corp, the vehicles, the sessions and the competitor brands are invented. Full
disclosure in `data/README.md`.
