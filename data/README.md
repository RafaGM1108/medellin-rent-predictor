# Data layers

Data moves through numbered layers. Each layer is produced only from the layers
before it, by code in `src/`, so any layer can be rebuilt from `01_raw`.

| Layer | Contents | Written by |
|-------|----------|------------|
| `01_raw` | Data exactly as received or scraped, in `listings/` (rental listings) and `geo/` (comunas/barrios boundaries). **Immutable: never edit, overwrite or clean in place.** | `medellin_rent.data`, `medellin_rent.geo` |
| `02_intermediate` | Raw data parsed into a typed, consistent format (e.g. Parquet), with no business logic. | `medellin_rent.data` |
| `03_primary` | Cleaned, validated, domain-level tables (one row = one entity). | `medellin_rent.data` |
| `04_feature` | Engineered features, keyed by entity. | `medellin_rent.features` |
| `05_model_input` | Features joined and split into train/validation/test sets. | Feature pipeline |
| `06_models` | Trained, serialised models. | Training pipeline |
| `07_model_output` | Predictions and scores from the models. | Inference pipeline |
| `08_reporting` | Metrics (JSON/CSV) and figures (PNG) used in the README and reports. | Training / reporting code |

## Rules

- `01_raw` is read-only. If raw data is wrong, fix it in `02_intermediate`.
- Paths are set in `conf/base.yaml`. Code never hard-codes them.
- Only small, redistributable samples (`*_sample.*`) and `08_reporting/` are committed.
  Everything else is ignored by git. No current source allows a sample, so none is
  committed; tests use a synthetic fixture instead.
- Each source's origin, license and download date is recorded below.

## Sources

| Source | Used for | License | Retrieved | Notes |
|--------|----------|---------|-----------|-------|
| [Properati Colombia](https://www.kaggle.com/datasets/lauramartinezortiz/colombian-properties) (Kaggle mirror of Properati Data) | Listings (`operation = Alquiler`, Medellín) | Unknown on Kaggle; original license being verified | Manual download; checksum and date in `listings.source.json` (`make listings`) | `co_properties.csv`, not committed |
| [Límite Catastral de Comunas y Corregimientos](https://www.medellin.gov.co/geomedellin/datosAbiertos/1043) | Comuna boundaries (21 polygons) | CC BY-SA 4.0 + "no puede ser comercializada o transferida" | `make geo` (date in `comunas.source.json`) | EPSG:9377, reprojected to EPSG:4326 on load; not committed |
| [Límite Catastral de Barrios y Veredas](https://www.medellin.gov.co/geomedellin/datosAbiertos/1044) | Barrio boundaries + barrio → comuna (349 polygons) | CC BY-SA 4.0 + "no puede ser comercializada o transferida" | `make geo` (date in `barrios.source.json`) | EPSG:9377, reprojected to EPSG:4326 on load; not committed |
| [Estrato Socioeconómico](https://www.medellin.gov.co/geomedellin/datosAbiertos/396) | Estrato by location | CC BY-SA 4.0 + "no puede ser comercializada o transferida" | `make geo` (date in `estrato.source.json`) | 32,384 polygons, EPSG:9377; not committed |

## Source survey (issue #7, checked 2026-10-05)

### Real-estate portals

`robots.txt` was checked for a generic user agent (`*`); the terms of use were read in full.

| Portal | robots.txt (`*`) | Terms of use | Scraping allowed? |
|--------|------------------|--------------|-------------------|
| [Fincaraíz](https://www.fincaraiz.com.co/informacion#terminos-y-condiciones) | Listings allowed; ~1,770 named bots fully disallowed | Users must refrain from "scraping, bots, crawlers, extracción masiva de datos"; databases may not be reproduced without written authorization | **No** |
| [Metrocuadrado](https://www.metrocuadrado.com/metrocuadrado-header/terminoscondiciones) | `/detail/` and `/metrocuadrado-results/` disallowed | Content may only be viewed/shared, not exploited; no automated operations | **No** |
| [Ciencuadras](https://www.ciencuadras.com/terminos-y-condiciones) | Search pages disallowed | §7.10-7.12 forbid reproducing content and "web scraping" / robots | **No** |
| MercadoLibre Inmuebles | Listings allowed, filter URLs disallowed; AI crawlers fully disallowed | Terms forbid "robots, harvesters, spiders, scraping" | **No** |
| [Espacio Urbano](https://espaciourbano.com/Terminos.htm) | No `*` rule (only named bots) | No explicit scraping clause, but "Todos los Derechos Reservados" | **Not without written permission** |

**Decision: no scraping.** Every major portal forbids it in its terms of use; Espacio Urbano
does not forbid it explicitly but reserves all rights, so it is only an option with written
permission. Phase 1 therefore builds a **loader for a public dataset** (issue #8).

### Public listing datasets

| Dataset | Operation | Fields relevant to the spec | License | Usable? |
|---------|-----------|-----------------------------|---------|---------|
| Properati Colombia (Properati Data; mirrored on Kaggle as [Colombian Properties](https://www.kaggle.com/datasets/lauramartinezortiz/colombian-properties)) | Sale **and rent** (`operation = Alquiler`, `price_period = Mensual`) | lat/lon, city, neighbourhood, rooms, bedrooms, bathrooms, surface, price, currency, title/description | Kaggle mirror: "Unknown"; Properati's open data was reportedly published under CC BY — **original license not yet verified** | Candidate, pending license check |
| [Precios inmuebles Medellín](https://www.kaggle.com/datasets/valentinafeve/precios-inmuebles-medelln) | Sale only | estrato, area, rooms, parking, age, floor | Unknown | No (sale prices, unknown license) |
| [Medellín Properties (2023)](https://www.kaggle.com/datasets/cesaregr/medelln-properties) | Sale only | area, rooms, age (binned) | Apache 2.0 | No (sale prices) |

### Geographic layers (Medellín open data, GeoMedellín / MEData)

- **Comunas y corregimientos** (21 polygons, EPSG:6257, updated daily) for mapping listings
  to comunas and for maps.
- **Estrato socioeconómico** (mode of estrato by lot/block), so estrato can be assigned to
  any listing with coordinates even if the listing does not state it.
- Both are CC BY-SA 4.0 with the extra condition that the data "no puede ser comercializada o
  transferida", so they are downloaded by code and **never committed**; outputs cite the
  Alcaldía de Medellín.

### Decisions (2026-10-05)

- **Listings:** Properati Colombia, downloaded manually from Kaggle into
  `data/01_raw/listings/`. Until its license is confirmed, the raw data and anything derived
  row by row from it are **not committed**; only code and aggregated outputs in
  `08_reporting` are.
- **Geo layers:** used for this public, non-commercial portfolio project. The files are
  never redistributed: they are downloaded by code and git-ignored; outputs credit the
  Alcaldía de Medellín.

### Properati file profile (measured 2026-10-08 with `make listings`)

- `co_properties.csv`: 617,694,986 bytes, exactly 1,000,000 rows, 25 columns; listings
  published between 2020-07-26 and 2021-08-19.
- Filter `l3 = Medellín`, `operation_type = Arriendo`, `property_type = Apartamento`,
  `currency = COP`: **102,169 listings**.
- About 74% of those repeat the same title, description and price (re-published ads), so
  roughly 26,000 are distinct.
- Structured fields are sparse: coordinates 29% present, `l4` (barrio) 30%, bedrooms 16%,
  surface 1-2%. The listing text mentions rooms in 68%, bathrooms in 62%, parking in 56%,
  area (m²) in 23%, estrato in 2.5%.
- Mobile numbers and emails appear in about 1% of texts and are redacted on load. The text
  also contains agent names, so `title` and `description` stay in git-ignored layers and are
  dropped once features are extracted (#12).

### Intermediate table (`02_intermediate/listings.parquet`, `make feature`, 2026-10-08)

Structured value first, then the listing text. Share of the 102,169 rows with a value:
area 23.5%, bedrooms 80.4%, bathrooms 98.7%, parking mentioned 55.8% (number of spots
4.5%), floor 2.6%, estrato 2.5%, building age ~0%, coordinates 28.2%, barrio 28.6%.
`title` and `description` are not in this table.

### Location mapping (`medellin_rent.geo.mapping`, 2026-10-08)

- Properati's `l4` (`barrio_raw`) is the **comuna**, not the barrio: its 21 values are exactly
  Medellín's 16 comunas and 5 corregimientos. All 21 match the official names after
  normalization (accents, case, leading article: "Candelaria" = "LA CANDELARIA").
- The barrio comes only from the coordinates (point in polygon on the barrios layer). If the
  coordinates fall in a different comuna than the declared one, the declared comuna is kept
  and the barrio is left empty.
- On the 102,169 intermediate rows: comuna from the name 29,262, from coordinates only 11,
  none 72,896; barrio 28,349; name/coordinate conflicts 220.
- Most rows without a location are re-published duplicates: among the 26,116 distinct
  listings (same text and rent), 81% have a location.

### Cleaning rules (`medellin_rent.data.clean`, `03_primary/listings.parquet`, 2026-10-08)

Thresholds were chosen from the real distribution of the intermediate table (rent 1st-99th
percentile 620,000-9,700,000 COP; median area 70 m²; median rent per m² 21,400 COP).

| # | Rule | Effect on the real run |
|---|------|------------------------|
| 1 | Same text (hash) and rent = re-published ad: keep the most recent | dropped 76,053 of 102,169 |
| 2 | Rent outside 300,000-30,000,000 COP: drop (rent is the target) | dropped 21 of 26,116 |
| 3 | Area outside 20-1,000 m² or rent per m² outside 5,000-150,000 COP: area set to NA | 52 values cleared |
| 4 | Bedrooms, bathrooms or parking above 10, floor above 50: set to NA | 12 / 1 / 0 / 10 values cleared |

Result: **26,095 listings**, 21,112 with a comuna. The table passes `PrimarySchema`.

### Estrato (`medellin_rent.geo.mapping.assign_estrato`, 2026-10-08)

The estrato stated in the listing is kept (2,546 of 102,169 rows); otherwise it comes from
the official estrato polygon that contains the coordinates (26,410 rows). Where both exist
(1,201 primary listings), the layer matches the declared estrato exactly in 68.3% of cases
and within one level in 95.8%. `estrato_source` records `listing` or `layer`.
In `03_primary`, estrato coverage goes from 5.5% to **79.2%**; 20,460 listings have both a
comuna and an estrato, 7,151 of them also an area.

### Open points

- Verify the license of the original Properati Colombia data before downloading it.
- Coverage gaps versus the spec: Properati has no explicit estrato, parking, floor or
  building age. Estrato comes from the geographic layer; parking, floor, age and amenities
  can only be parsed from the listing text where present.
