"""Generate drop-camera-style training points for the Tongatapu exercise.

These are NOT real drop-camera observations. They are derived from the expert-drawn
training polygons (tongatapu_training.geojson) to resemble what a drop-camera survey
would deliver: one record per drop, a single majority class, a boat GPS position with
error, and drops grouped into survey days.

How the points are made:
1. A number of drops per class (more for common/interesting habitats), shared among that
   class's polygons in proportion to the square root of polygon area (every polygon gets at
   least one drop, as a field team would visit every habitat patch it had marked).
2. Drop locations are random inside the polygons, at least MIN_SPACING_M apart (a boat
   rarely drops twice within a few tens of metres).
3. Drops are grouped into survey areas by location and ordered along a route; each route is
   split into days of at most MAX_PER_DAY drops.
4. The class is taken at the true drop location. The recorded position adds GPS error only
   (no boat drift: the sites are mostly < 5 m deep with weak currents), so a few recorded
   positions fall outside the polygon the label came from, as in real surveys.

Only polygons inside the notebook AOI are used.

Two files are written from the same drops:
  tongatapu_drops.geojson        recorded (boat GPS) positions, with error
  tongatapu_drops_exact.geojson  true positions, all inside their polygons (for comparison)

Run from the notebook folder:  python prep/make_drop_points.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from sklearn.cluster import KMeans

SEED = 2024
CRS = "EPSG:32701"
DROPS_PER_CLASS = {"Seagrass": 240, "Coral": 200, "Sand": 170, "Rubble": 140,
                   "Algal mat/sparse seagrass": 45, "Deep Water": 55}   # uneven, as in real surveys
MIN_SPACING_M = 40
N_AREAS = 16          # survey areas (KMeans on location); an area with many drops takes several days
MAX_PER_DAY = 60
GPS_SD_M = 5          # per axis (radial: median ~6 m, 95% ~12 m)
AOI = (-175.36, -21.20, -174.99, -20.99)   # as notebook 1
FIRST_SURVEY_DAY = "2023-07-03"

here = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)
polys = gpd.read_file(here / "tongatapu_training.geojson")
polys = polys[polys.within(box(*AOI))].to_crs(CRS)

# 1. drops per polygon
rows = []
for cls, n_cls in DROPS_PER_CLASS.items():
    p = polys[polys["Class"] == cls]
    w = np.sqrt(p.area.values)
    n = np.maximum(1, np.round(n_cls * w / w.sum())).astype(int)
    rows += [(cls, geom, k) for geom, k in zip(p.geometry, n)]

# 2. random locations, minimum spacing
placed = []
for cls, geom, k in rows:
    x0, y0, x1, y1 = geom.bounds
    got, tries = 0, 0
    while got < k and tries < 5000:
        tries += 1
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if not geom.contains(Point(x, y)):
            continue
        if placed and min(np.hypot(x - px, y - py) for _, px, py in placed) < MIN_SPACING_M:
            continue
        placed.append((cls, x, y))
        got += 1
drops = pd.DataFrame(placed, columns=["class", "x_true", "y_true"])

# 3. survey days and routes
xy = drops[["x_true", "y_true"]].to_numpy()
drops["day"] = KMeans(N_AREAS, n_init=10, random_state=SEED).fit_predict(xy)
drops = drops.sort_values("x_true").reset_index(drop=True)
day_order = drops.groupby("day")["x_true"].mean().sort_values().index
day_rank = {d: i for i, d in enumerate(day_order)}
routes = {}
for d, g in drops.groupby("day"):
    todo, route = list(g.index), [g["x_true"].idxmin()]
    todo.remove(route[0])
    while todo:                                   # nearest-neighbour route
        last = drops.loc[route[-1], ["x_true", "y_true"]].to_numpy(float)
        nxt = min(todo, key=lambda j: np.hypot(*(drops.loc[j, ["x_true", "y_true"]].to_numpy(float) - last)))
        route.append(nxt); todo.remove(nxt)
    routes[day_rank[d]] = route
days = [chunk for a in sorted(routes) for chunk in np.array_split(routes[a], -(-len(routes[a]) // MAX_PER_DAY))]
survey_dates = pd.bdate_range(FIRST_SURVEY_DAY, periods=len(days) * 2)[::2]   # every second working day
records = []
for i, route in enumerate(days):
    t = pd.Timestamp(survey_dates[i]) + pd.Timedelta(hours=8)
    for k, j in enumerate(route):
        t += pd.Timedelta(minutes=float(rng.uniform(5, 12)))
        ex, ey = rng.normal(0, GPS_SD_M, 2)
        records.append({"drop_id": f"TBU-{i + 1:02d}-{k + 1:03d}", "survey_day": i + 1,
                        "datetime": t.strftime("%Y-%m-%dT%H:%M"), "class": drops.loc[j, "class"],
                        "x": drops.loc[j, "x_true"] + ex, "y": drops.loc[j, "y_true"] + ey,
                        "x_true": drops.loc[j, "x_true"], "y_true": drops.loc[j, "y_true"]})

out = pd.DataFrame(records).sort_values("drop_id")
for name, x, y in [("tongatapu_drops", "x", "y"), ("tongatapu_drops_exact", "x_true", "y_true")]:
    gdf = gpd.GeoDataFrame(out[["drop_id", "survey_day", "datetime", "class"]],
                           geometry=gpd.points_from_xy(out[x], out[y]), crs=CRS).to_crs(4326)
    gdf.to_file(here.parent / f"{name}.geojson", driver="GeoJSON")
print(gdf["class"].value_counts(), f"\n{len(gdf)} drops over {len(days)} days")
