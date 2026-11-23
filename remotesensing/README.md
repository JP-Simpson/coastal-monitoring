# Benthic habitat mapping in Tonga: Python counterpart of the GEE scripts

| Notebook | GEE equivalent | Content |
|---|---|---|
| `01_composite.ipynb` | script 1 | STAC search (≤ 30% cloud, May – Oct 2023, dry season), SCL cloud/shadow mask + 50 m buffer, solar-day mosaics, 25th percentile composite, edge-based Otsu land mask (green vs min(B11, B8A)), optically deep water mask from the hue/saturation deep water index of B1–B3 (as GEE script 1; Lee et al. 2022) with an adjustable threshold `DEEP_THRESHOLD`, checked against the drops in notebook 2 |
| `02_classification.ipynb` | script 2 | Drop-camera points, deep water mask check, spectra by class, GPS position error (3 × 3 window sampling, edge drops left out of training), label correlogram, spatial block test set and folds, Lyzenga DIIs from the sand profile. Then **three Random Forests on the same drops**: pixel features (B1–B5, DIIs, NDVI / red-edge NDVI / red–green); the same features averaged over SLIC superpixels (two-stage, object-based); and AlphaEarth Foundations embeddings. Random vs spatial CV, held-out test blocks, F1 by class, area of applicability for each, maps, agreement |
| `03_transfer_vavau.ipynb` | — | Same composite steps for Vava'u; the three Tongatapu models applied (Tongatapu DII calibration, Vava'u superpixels, Vava'u embeddings from UTM zones 1S and 2S); area of applicability for each against the Tongatapu training drops; class shares inside/outside each, and agreement between methods |

**Training data.** `tongatapu_drops.geojson` holds 852 drop-camera-style points (fields `drop_id`, `survey_day`, `datetime`, `class`). They are **synthetic**: generated from the expert habitat polygons by `prep/make_drop_points.py`, with ~40 m minimum spacing, 23 survey days (at most 60 drops a day), deliberately uneven class counts (algal mat 45, deep water 55, seagrass 242), and boat-GPS error only (5 m per axis: median offset ~6 m, 95% within ~12 m; no drift, as the sites are shallow and sheltered). Only polygons inside the notebook AOI are used. About 5% of recorded positions fall outside a polygon of their own class. The polygons are kept in `prep/tongatapu_training.geojson`. Replace the drops file with real drop-camera data (same fields) when available.

`tongatapu_drops_exact.geojson` holds the same drops at their true positions (all inside their polygons), for testing how much position error costs: point notebook 2 at it to compare.

## Setup
```bash
conda env create -f environment.yml && conda activate tonga-benthic && jupyter lab
```

## Notes
* **Embeddings.** Notebooks 2 and 3 read the 2023 AlphaEarth Foundations Satellite Embedding tiles over the internet from Source Cooperative (`https://data.source.coop/tge-labs/aef/v1/annual`, free, no account): the tile index once (cached as `data/aef_index.parquet`), then about 0.5 GB per AOI. *The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind* (CC-BY 4.0). Tiles are resampled onto the composite grid using their own georeferencing, so their storage orientation and UTM zone don't matter.
* Notebook 1 reads six months of Sentinel-2 imagery for a ~38 × 23 km area (a few GB), so the composite takes a while on first run. Notebooks 2 and 3 read the saved composite.
* Cloud masking uses Sen2Cor's scene classification (`SCL`) rather than s2cloudless, which is not published in the STAC catalogue used here.
* The notebooks were tested end to end only with synthetic scenes at 30 m, and with synthetic embedding tiles (including one stored bottom-up and one in zone 2S); the real embedding source could not be reached from the build environment. On the first real run, check the land and deep water histograms in notebook 1, and the share of drops kept per class in notebook 2. `BLOCK_M` in notebook 2 should be set from the correlogram.
