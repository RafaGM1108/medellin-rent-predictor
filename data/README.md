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
  Everything else is ignored by git.
- Each source's origin, license and download date is recorded below.

## Sources

| Source | Used for | License | Retrieved | Notes |
|--------|----------|---------|-----------|-------|
| TODO: listings source (pending decision, see survey below) | Listings | TODO | TODO | TODO |
| [Límite Catastral de Comunas y Corregimientos](https://www.medellin.gov.co/geomedellin/datosAbiertos/1043) | Comuna boundaries | CC BY-SA 4.0 + "no puede ser comercializada o transferida" | TODO | Not committed; downloaded by code |
| [Estrato Socioeconómico](https://www.medellin.gov.co/geomedellin/datosAbiertos/396) | Estrato by location | CC BY-SA 4.0 + "no puede ser comercializada o transferida" | TODO | Not committed; downloaded by code |

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

### Open points

- Verify the license of the original Properati Colombia data before downloading it.
- Coverage gaps versus the spec: Properati has no explicit estrato, parking, floor or
  building age. Estrato comes from the geographic layer; parking, floor, age and amenities
  can only be parsed from the listing text where present.
