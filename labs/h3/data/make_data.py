"""
Generate the synthetic dataset for the H3 charging-detection demo.

Three CSVs plus 200-row samples, all reproducible from SEED:

    ev_pings.csv             per-vehicle telematics trajectories over one day
    ev_stations.csv          charging stations, ABC Corp's and competitors'
    ev_charging_sessions.csv ABC's INTERNAL charging records - the labels

The business question: given a vehicle that stopped somewhere, was it charging at a
particular station? ABC can answer that for its own 80 stations because it has the
session records. It cannot answer it for the 250 competitor stations - which is
exactly what the model is for.

Nothing here is real. Demand is seeded around real European city coordinates so the
maps look like Europe, but every vehicle, station, operator, session and heading is
invented. See README.md for the full disclosure.

Run:  python make_data.py [outdir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
RNG = np.random.default_rng(SEED)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

N_VEHICLES = 3_000
N_ABC = 80
N_COMPETITOR = 250

LOW_SPEED_KMH = 5.0                                  # <= this counts as parked
WEEK_START = np.datetime64("2026-06-08T00:00:00")    # a Monday; kept for seed stability
# NOTE: despite the name, the walk is 6-10 stops per vehicle, so the data spans ONE DAY
# (2026-06-08 00:00 to 19:03), not a week. Renaming it would not change the output.
PING_MINUTES = 12                                    # nominal reporting interval

# --- Real European cities: name, country, lat, lon, metro population (millions) ----------
CITIES = [
    ("Paris",       "FR", 48.8566,  2.3522, 11.1), ("Berlin",      "DE", 52.5200, 13.4050,  6.1),
    ("Madrid",      "ES", 40.4168, -3.7038,  6.7), ("Rome",        "IT", 41.9028, 12.4964,  4.3),
    ("Amsterdam",   "NL", 52.3676,  4.9041,  2.5), ("Vienna",      "AT", 48.2082, 16.3738,  2.9),
    ("Hamburg",     "DE", 53.5511,  9.9937,  5.1), ("Munich",      "DE", 48.1351, 11.5820,  2.9),
    ("Milan",       "IT", 45.4642,  9.1900,  4.3), ("Barcelona",   "ES", 41.3874,  2.1686,  5.6),
    ("Brussels",    "BE", 50.8503,  4.3517,  2.1), ("Copenhagen",  "DK", 55.6761, 12.5683,  2.0),
    ("Stockholm",   "SE", 59.3293, 18.0686,  2.4), ("Oslo",        "NO", 59.9139, 10.7522,  1.5),
    ("Warsaw",      "PL", 52.2297, 21.0122,  3.1), ("Prague",      "CZ", 50.0755, 14.4378,  2.7),
    ("Budapest",    "HU", 47.4979, 19.0402,  3.0), ("Lisbon",      "PT", 38.7223, -9.1393,  2.9),
    ("Porto",       "PT", 41.1579, -8.6291,  1.7), ("Dublin",      "IE", 53.3498, -6.2603,  2.1),
    ("Lyon",        "FR", 45.7640,  4.8357,  2.3), ("Marseille",   "FR", 43.2965,  5.3698,  1.9),
    ("Toulouse",    "FR", 43.6047,  1.4442,  1.4), ("Nice",        "FR", 43.7102,  7.2620,  1.0),
    ("Frankfurt",   "DE", 50.1109,  8.6821,  2.7), ("Cologne",     "DE", 50.9375,  6.9603,  3.6),
    ("Stuttgart",   "DE", 48.7758,  9.1829,  2.7), ("Dusseldorf",  "DE", 51.2277,  6.7735,  1.5),
    ("Rotterdam",   "NL", 51.9244,  4.4777,  1.4), ("Utrecht",     "NL", 52.0907,  5.1214,  1.0),
    ("Antwerp",     "BE", 51.2194,  4.4025,  1.2), ("Zurich",      "CH", 47.3769,  8.5417,  1.4),
    ("Geneva",      "CH", 46.2044,  6.1432,  1.0), ("Turin",       "IT", 45.0703,  7.6869,  1.8),
    ("Naples",      "IT", 40.8518, 14.2681,  3.1), ("Bologna",     "IT", 44.4949, 11.3426,  1.0),
    ("Valencia",    "ES", 39.4699, -0.3763,  1.6), ("Seville",     "ES", 37.3891, -5.9845,  1.5),
    ("Bilbao",      "ES", 43.2630, -2.9350,  1.0), ("Gothenburg",  "SE", 57.7089, 11.9746,  1.1),
    ("Malmo",       "SE", 55.6050, 13.0038,  0.7), ("Helsinki",    "FI", 60.1699, 24.9384,  1.5),
    ("Krakow",      "PL", 50.0647, 19.9450,  1.0), ("Bucharest",   "RO", 44.4268, 26.1025,  2.2),
    ("Athens",      "GR", 37.9838, 23.7275,  3.2), ("Sofia",       "BG", 42.6977, 23.3219,  1.3),
    ("Zagreb",      "HR", 45.8150, 15.9819,  1.1), ("Ljubljana",   "SI", 46.0569, 14.5058,  0.5),
    ("Bratislava",  "SK", 48.1486, 17.1077,  0.7), ("Lille",       "FR", 50.6292,  3.0573,  1.2),
]
EV_ADOPTION = {
    "NO": 1.90, "SE": 1.45, "NL": 1.40, "DK": 1.30, "FI": 1.20, "CH": 1.15,
    "DE": 1.05, "BE": 1.00, "AT": 1.00, "FR": 0.95, "IE": 0.95, "PT": 0.85,
    "ES": 0.75, "IT": 0.70, "SI": 0.70, "CZ": 0.65, "HR": 0.60, "SK": 0.60,
    "PL": 0.55, "HU": 0.55, "GR": 0.50, "RO": 0.45, "BG": 0.45,
}
COMPETITORS = ["Helio Charge", "Kestrel Power", "Northwind EV", "Corvus Grid", "Solace Charge"]

# Fictional carmakers and models. Invented for this demo - see README.md.
MODELS = [
    # make,     model,       engine, battery kWh, max charge kW
    ("Aurex",   "Lumen",     "BEV",   58,  120), ("Aurex",   "Lumen LR", "BEV",  77,  150),
    ("Vantor",  "Arc",       "BEV",   64,  135), ("Vantor",  "Arc GT",   "BEV",  91,  220),
    ("Selva",   "Corta",     "BEV",   42,   85), ("Selva",   "Corta+",   "BEV",  52,  100),
    ("Marden",  "Ridge",     "BEV",   82,  175), ("Marden",  "Ridge XL", "BEV", 105,  250),
    ("Aurex",   "Lumen PHV", "PHEV",  14,    7), ("Vantor",  "Arc PHV",  "PHEV", 11,    7),
    ("Selva",   "Corta PHV", "PHEV",   9,    4), ("Marden",  "Ridge PHV","PHEV", 18,    7),
]


def make_vehicles():
    """Fleet master data: what each vehicle actually is."""
    # A real fleet skews to full battery-electric, with a PHEV tail.
    w = np.array([1.0]*8 + [0.30]*4); w = w/w.sum()
    pick = RNG.choice(len(MODELS), size=N_VEHICLES, p=w)
    rows = [MODELS[i] for i in pick]
    return pd.DataFrame({
        "vehicle_id":   [f"EV{v:06d}" for v in range(N_VEHICLES)],
        "make":         [r[0] for r in rows],
        "model":        [r[1] for r in rows],
        "engine_type":  [r[2] for r in rows],
        "battery_kwh":  [r[3] for r in rows],
        "max_charge_kw":[r[4] for r in rows],
        "model_year":   RNG.integers(2019, 2027, N_VEHICLES),
    })

city_df = pd.DataFrame(CITIES, columns=["city", "country", "lat", "lon", "pop_m"])
city_df["adoption"] = city_df.country.map(EV_ADOPTION)
city_df["weight"] = city_df.pop_m * city_df.adoption
CITY_P = (city_df.weight / city_df.weight.sum()).to_numpy()

KM_PER_DEG = 111.32


def offset(lat, lon, north_km, east_km):
    """Shift a point by a given number of km north and east."""
    return lat + north_km / KM_PER_DEG, lon + east_km / (KM_PER_DEG * np.cos(np.radians(lat)))


def jitter(lat, lon, metres):
    """Random offset of roughly `metres` in a random direction (GPS error)."""
    th = RNG.uniform(0, 2 * np.pi)
    r = abs(RNG.normal(0, metres)) / 1000.0
    return offset(lat, lon, r * np.cos(th), r * np.sin(th))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ stations
def make_stations():
    n = N_ABC + N_COMPETITOR
    idx = RNG.choice(len(city_df), size=n, p=CITY_P)
    lat0 = city_df.lat.to_numpy()[idx]
    lon0 = city_df.lon.to_numpy()[idx]
    lat = lat0 + RNG.normal(0, 6.0 / KM_PER_DEG, n)
    lon = lon0 + RNG.normal(0, 6.0 / KM_PER_DEG, n) / np.cos(np.radians(lat0))

    is_abc = np.zeros(n, bool)
    is_abc[RNG.choice(n, size=N_ABC, replace=False)] = True

    abc_seq = cmp_seq = 0
    sid, operator = [], []
    for a in is_abc:
        if a:
            sid.append(f"ABC{abc_seq:04d}"); operator.append("ABC Corp"); abc_seq += 1
        else:
            sid.append(f"CMP{cmp_seq:04d}")
            operator.append(COMPETITORS[cmp_seq % len(COMPETITORS)]); cmp_seq += 1

    # Chargers cluster: retail parks and motorway services host several operators side by
    # side. Co-locating 40% of competitor sites next to an ABC one is realistic AND is what
    # makes the problem hard - a car charging at the competitor next door has a rising SoC,
    # a long dwell and a bay-aligned heading, metres from an ABC charger it never touched.
    abc_ix = np.where(is_abc)[0]
    for j in np.where(~is_abc)[0]:
        if RNG.random() < 0.40:
            k = int(RNG.choice(abc_ix))
            th = RNG.uniform(0, 2 * np.pi); r = RNG.uniform(0.10, 0.40)
            lat[j], lon[j] = offset(lat[k], lon[k], r * np.cos(th), r * np.sin(th))

    return pd.DataFrame({
        "station_id": sid,
        "operator": operator,
        "is_abc": is_abc,
        "lat": np.round(lat, 5),
        "lon": np.round(lon, 5),
        "country": city_df.country.to_numpy()[idx],
        "city": city_df.city.to_numpy()[idx],
        "n_chargers": RNG.choice([2, 4, 6, 8, 12], n, p=[.30, .30, .20, .13, .07]),
        "max_power_kw": RNG.choice([50, 150, 350], n, p=[.45, .40, .15]),
        # Which way a car faces when plugged in. Real station records rarely carry this;
        # it is here because bay orientation is the "alignment" signal the demo tests.
        "bay_bearing_deg": RNG.integers(0, 360, n),
    })


# ------------------------------------------------------------------ trajectories
def make_pings(stations, vehicles):
    """Walk each vehicle through a day of stops and drives, emitting pings."""
    st_lat = stations.lat.to_numpy(); st_lon = stations.lon.to_numpy()
    st_bear = stations.bay_bearing_deg.to_numpy(); st_kw = stations.max_power_kw.to_numpy()
    st_id = stations.station_id.to_numpy(); st_is_abc = stations.is_abc.to_numpy()

    # Group stations by city so a vehicle charges near where it lives.
    home_city = RNG.choice(len(city_df), size=N_VEHICLES, p=CITY_P)
    city_stations = {c: np.where(
        haversine_km(city_df.lat[c], city_df.lon[c], st_lat, st_lon) < 25.0)[0]
        for c in range(len(city_df))}

    veh_batt = vehicles.battery_kwh.to_numpy()
    veh_rate = vehicles.max_charge_kw.to_numpy()
    veh_type = vehicles.engine_type.to_numpy()

    rows, sessions = [], []
    ping_id = 0

    for v in range(N_VEHICLES):
        vid = f"EV{v:06d}"
        c = home_city[v]
        home_lat, home_lon = offset(city_df.lat[c], city_df.lon[c],
                                    RNG.normal(0, 5.0), RNG.normal(0, 5.0))
        near = city_stations[c]
        has_home_charger = RNG.random() < 0.45
        soc = RNG.uniform(35, 90)
        t = WEEK_START + np.timedelta64(int(RNG.integers(0, 240)), "m")

        n_stops = RNG.integers(6, 11)
        for _ in range(n_stops):
            # ---- decide what kind of stop this is --------------------------------
            roll = RNG.random()
            charging_here = None
            near_bay = None
            if roll < 0.19 and len(near):                      # public charging stop
                s = int(RNG.choice(near))
                lat, lon = jitter(st_lat[s], st_lon[s],
                                  150 if RNG.random() < 0.10 else 22)   # 10% poor GPS
                charging_here = s
            elif roll < 0.45 and len(near):
                # Parked NEAR a station but not charging - shopping, offices, street
                # parking. These are the negatives that make the problem interesting:
                # proximity alone will not separate them from real sessions.
                s = int(RNG.choice(near))
                lat, lon = jitter(st_lat[s], st_lon[s], RNG.uniform(45, 130))
                # Same car park, same bay orientation - so heading alignment is a hint,
                # not a giveaway. 60% of these look exactly like a plugged-in car.
                if RNG.random() < 0.60:
                    near_bay = (st_bear[s] + RNG.normal(0, 40)) % 360
            elif roll < 0.70:
                lat, lon = offset(home_lat, home_lon, RNG.normal(0, 0.4), RNG.normal(0, 0.4))
            else:
                lat, lon = offset(city_df.lat[c], city_df.lon[c],
                                  RNG.normal(0, 7.0), RNG.normal(0, 7.0))

            if charging_here is not None:
                # How long a charge takes is physics, not a random draw: the energy needed
                # divided by the slower of the car's rate and the station's. A 105 kWh car
                # gaining 40% sits far longer than a 9 kWh PHEV doing the same.
                want = RNG.uniform(18, 55) if veh_type[v] == "BEV" else RNG.uniform(12, 40)
                want = min(100 - soc, want)
                kwh = want / 100 * veh_batt[v]
                rate = min(veh_rate[v], st_kw[charging_here])
                n_ping = int(np.clip(round(kwh / rate * 60 / PING_MINUTES), 2, 12))
            else:
                n_ping = int(RNG.integers(2, 8))
            start = t

            if charging_here is not None:
                # Plugged in: the car sits square in the bay, and the battery fills.
                bearing = ((st_bear[charging_here] + RNG.normal(0, 35)) % 360
                           if RNG.random() > 0.15 else RNG.uniform(0, 360))
                gain = want
                soc_step = gain / n_ping
            else:
                bearing = near_bay if near_bay is not None else RNG.uniform(0, 360)
                # A few home stops trickle-charge: SoC rises with no station involved.
                # This is the confuser that stops SoC alone from being the whole answer.
                gain = (RNG.uniform(15, 40)
                        if (has_home_charger and 0.45 <= roll < 0.70
                            and RNG.random() < 0.55) else 0.0)
                gain = min(100 - soc, gain)
                soc_step = gain / n_ping

            for k in range(n_ping):
                plat, plon = jitter(lat, lon, 8)
                # Heading is unreliable when stationary - real units drop it or hold a
                # stale value. 20% missing keeps the feature honest.
                head = np.nan if RNG.random() < 0.20 else (bearing + RNG.normal(0, 6)) % 360
                soc = min(100.0, soc + soc_step)
                rows.append((ping_id, vid, t, plat, plon,
                             round(float(RNG.uniform(0, 3)), 1), head, soc))
                ping_id += 1
                t = t + np.timedelta64(PING_MINUTES, "m")

            if charging_here is not None and st_is_abc[charging_here]:
                # ABC only records sessions at ABC's own chargers. Competitor charging
                # happens in the ping data but leaves no trace in ABC's systems - which
                # is the entire reason this model has to exist.
                mins = n_ping * PING_MINUTES
                sessions.append((st_id[charging_here], vid, start, t,
                                 mins, round(gain / 100 * 60 * min(1.0, mins / 60), 2),
                                 int(st_kw[charging_here])))

            # ---- drive to the next place -----------------------------------------
            for _ in range(int(RNG.integers(1, 4))):
                dlat, dlon = offset(lat, lon, RNG.normal(0, 3.0), RNG.normal(0, 3.0))
                soc = max(3.0, soc - RNG.uniform(0.4, 2.2))
                rows.append((ping_id, vid, t, dlat, dlon,
                             round(float(RNG.uniform(25, 115)), 1),
                             round(float(RNG.uniform(0, 360)), 1), soc))
                ping_id += 1
                t = t + np.timedelta64(PING_MINUTES, "m")

    pings = pd.DataFrame(rows, columns=[
        "ping_id", "vehicle_id", "ts_utc", "lat", "lon", "speed_kmh", "heading_deg", "soc_pct"])
    pings["lat"] = pings.lat.round(5)
    pings["lon"] = pings.lon.round(5)
    pings["heading_deg"] = pings.heading_deg.round(1)
    pings["soc_pct"] = pings.soc_pct.round(1)
    pings["ts_utc"] = pd.to_datetime(pings.ts_utc).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    ses = pd.DataFrame(sessions, columns=[
        "station_id", "vehicle_id", "start_utc", "end_utc",
        "duration_min", "energy_kwh", "power_kw"])
    ses.insert(0, "session_id", [f"S{i:06d}" for i in range(len(ses))])
    for c in ("start_utc", "end_utc"):
        ses[c] = pd.to_datetime(ses[c]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return pings, ses


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stations = make_stations()
    vehicles = make_vehicles()
    pings, sessions = make_pings(stations, vehicles)

    vehicles.to_csv(OUT / "ev_vehicles.csv", index=False)
    stations.to_csv(OUT / "ev_stations.csv", index=False)
    pings.to_csv(OUT / "ev_pings.csv", index=False)
    sessions.to_csv(OUT / "ev_charging_sessions.csv", index=False)
    for f, df in [("ev_pings", pings), ("ev_stations", stations),
                  ("ev_charging_sessions", sessions), ("ev_vehicles", vehicles)]:
        df.head(200).to_csv(OUT / f"{f}_sample.csv", index=False)

    parked = pings[pings.speed_kmh <= LOW_SPEED_KMH]
    print(f"vehicles          : {pings.vehicle_id.nunique():,}")
    print(f"pings             : {len(pings):,}  (parked {len(parked):,})")
    print(f"stations          : {len(stations)}  (ABC {int(stations.is_abc.sum())}, "
          f"competitor {int((~stations.is_abc).sum())})")
    print(f"ABC sessions      : {len(sessions):,}  "
          f"({sessions.vehicle_id.nunique():,} distinct vehicles)")
    print(f"heading missing   : {pings.heading_deg.isna().mean():.1%} of pings")
    print(f"fleet             : {(vehicles.engine_type=='BEV').sum():,} BEV, "
          f"{(vehicles.engine_type=='PHEV').sum():,} PHEV")
    for f in ["ev_pings.csv", "ev_stations.csv", "ev_charging_sessions.csv",
              "ev_vehicles.csv"]:
        print(f"  {f:28s} {(OUT / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
