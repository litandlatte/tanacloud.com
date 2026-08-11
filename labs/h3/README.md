# H3 for Data Science — Detecting EV Charging to See a Competitor's Demand (ABC Corp)

**Format:** 1-hour demo / knowledge-sharing session
**Audience:** Data Scientists, Data Engineers, Analytics leads
**Goal:** Show how **Uber H3** turns an intractable "compare every point to every point" spatial
join into a hash lookup — on a real, high-value decision: **where should ABC Corp build its next
EV charging stations across Europe?**

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
| EV telematics | `data/ev_pings.csv` | **Raw signal** — 161k pings, 3,000 vehicles, one week |
| Charging stations | `data/ev_stations.csv` | 80 ABC + 250 competitor |
| ABC's charging sessions | `data/ev_charging_sessions.csv` | **The labels — ABC's stations only** |

The third file's limit *is* the business problem: 991 sessions across ABC's 80 stations, and
**nothing, ever, for the other 250**. Full column definitions and the synthetic-data disclosure
are in `data/README.md`.

---

## 2. Why this is a compute problem

To ask "did this stop happen at that station?" you must **pair every stop with every station** —
a cross join.

In the demo that is 24,059 stops × 80 ABC stations = **1,924,720 pairs**, of which **0.26%**
survive a 500 m filter. Every one of the rest is a distance computed purely in order to be thrown
away. At ABC's real scale — millions of vehicles, hundreds of thousands of candidate sites — the
cross join simply cannot be built.

*"The problem isn't haversine. It's calling it on pairs that are obviously irrelevant."*

---

## 3. How we model it

A **binary classification** at the grain of a **(stop, station) pair**:

- **Stop** — a run of consecutive parked pings in one place. Carries how long the car stayed, how
  much charge it gained, and which way it faced.
- **Label** — 1 if ABC's session records show that vehicle charging at that station over that
  window. Each session claims exactly **one** stop, the one it overlaps most.
- **Train** on ABC's 80 stations → **predict** across all 330 → **count** confident hits per
  competitor station.

### Features

| Feature | Why it might indicate charging |
|---|---|
| `dist_m` | you have to be at the charger to use it |
| `dwell_min`, `n_pings` | charging takes time — a run of pings in one place |
| `soc_delta` | the battery **gained charge** during the stop |
| `align` | a plugged-in car sits square in its bay — `cos` of heading vs bay bearing |
| `max_power_kw`, `hour` | a 350 kW site fills a car faster; demand varies by time of day |

**Split by vehicle, never by row.** One car produces many stops; letting two of them straddle the
train/test boundary leaks.

---

## 4. Choosing the resolution — measured, not guessed

We have ground truth: cars we *know* were charging, and the stations they were plugged into. So
ask the data at which resolution a charging car lands in its station's hexagon.

| Res | Edge | Same cell | Within k=1 |
|----:|-----:|----------:|-----------:|
| 7 | 1,406 m | 98.8% | 100.0% |
| 8 | 531 m | 97.9% | 100.0% |
| **9** | **201 m** | **90.0%** | **99.8%** |
| 10 | 76 m | 79.6% | 97.5% |
| 11 | 29 m | 53.0% | 92.9% |
| 12 | 11 m | 21.6% | 65.9% |

Too coarse and each cell sweeps in half a city of irrelevant stops. Too fine and the car and the
charger fall into different cells and you lose the pairs you were looking for.

**We take res 9 with k = 1** — the finest resolution that still keeps recall essentially perfect.
Smallest possible haystack without dropping needles. **This is the honest way to pick a
resolution**, and it is only possible because the session records tell you which pairs were real.

---

## 5. The two implementations

**A — haversine cross join.** Materialise all 1,924,720 pairs, compute every distance, keep those
under 500 m.

**B — H3 as a blocking key.** Index each station into its cell and the `grid_disk(k=1)` ring
around it. Index each stop. A stop's candidates are whatever sits in its cell — a dictionary
lookup. The other pairs are not computed and discarded; **they are never created at all.**

Both produce the same candidate set and feed the same model. Only the cost of getting there
differs.

### Results

| | Haversine | H3 |
|---|---:|---:|
| Pairs materialised | 1,924,720 | **4,779** |
| Candidates kept | 4,959 | 4,759 |
| **Blocking recall** | 100.0% | **99.8%** |

**403× fewer pairs**, for 0.2% of true pairs lost. At inference across all 330 stations the full
cross join would be **7,939,470 pairs**; H3 scores **17,559** — **452× smaller**.

> **Quote the pair count, not the clock.** 403× is arithmetic and identical on every machine.
> Wall-clock depends on the laptop, and a slow Colab will contradict any timing you put on a slide.

**Always report blocking recall.** A filter that is fast because it silently drops real matches is
not an optimisation, it is a bug. Speed without this number is meaningless.

---

## 6. Model results

Held-out **by vehicle**: **PR-AUC 0.991**, ROC-AUC 0.996, precision 0.986 / recall 0.956 at a 0.5
threshold.

### "Why not just use a distance threshold?"

The fairest challenge in the room. Answer it with the number — PR-AUC from each feature **alone**:

| Feature alone | PR-AUC |
|---|---:|
| `dist_m` | 0.618 |
| `soc_delta` | 0.457 |
| `align` | 0.344 |
| `n_pings` / `dwell_min` | 0.274 |
| **all combined** | **0.991** |

Distance is the best single feature and nowhere near sufficient. The lift comes from combining
*where* the car was with *how long it stayed*, *whether the battery filled* and *which way it
faced*.

The dataset is built to make that true. **40% of competitor stations sit within 100–400 m of an
ABC station**, so a car charging next door shows a rising SoC, a long dwell and a bay-aligned
heading — only geometry separates it. And cars on home chargers show the SoC rise with no station
at all. Neither signal alone is safe.

### The deliverable

Predicted weekly sessions at every competitor station — demand ABC could never buy.

**Credibility check:** run the same method on **held-out vehicles** at **ABC's own** stations,
where the answer is known. Predicted vs actual session counts correlate **0.989** (286 predicted
against 296 actual). Do this check on held-out vehicles only; scoring every stop includes the ones
the model trained on and flatters it badly.

---

## 7. Suggested 60-minute run sheet

| Time | Segment | Section |
|-----:|---------|---------|
| 0–6 | Business problem: the 250 blind spots | 1 |
| 6–12 | Why a cross join breaks | 2 |
| 12–20 | H3 fundamentals — hexagons, resolutions, `grid_disk` | notebook §1 |
| 20–28 | Pings → stops → labels | notebook §2.2–2.3 |
| 28–36 | Choosing the resolution from ground truth | notebook §2.4 |
| 36–46 | Run both approaches; pair counts and blocking recall | notebook §2.5–2.6 |
| 46–54 | The model, and why not a threshold | notebook §2.7 |
| 54–60 | Score the competitors; the map | notebook §2.8–2.10 |

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

Locally:

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo.py
```

**Files**

- `H3_EV_Demand_Demo.ipynb` — **the lab.** Section 1 is H3 fundamentals with interactive maps;
  Section 2 is the business demo above.
- `demo.py` — Section 2 only, headless.
- `H3_EV_Demand_Demo.py` — the whole notebook as a percent-format script.
- `data/` — three CSVs, 200-row samples, data dictionary, `datapackage.json`, seeded generator.
- `requirements.txt` · `README.md` (this guide).

**🔑 Both `.py` files are generated from the notebook — edit the notebook, never them.**

**On the data:** entirely synthetic. Only the city coordinates are real, so the maps are
recognisable; ABC Corp, the vehicles, the sessions and the competitor brands are invented. Full
disclosure in `data/README.md`.
