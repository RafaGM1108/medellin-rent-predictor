# Feature dictionary

Features built by `medellin_rent.features.build` (`04_feature/features.parquet`) and used in
`05_model_input/{train,test}.parquet`. Every feature is computed from the listing's own
columns; nothing is derived from other listings, so the table cannot leak the target.
Missing-value handling that depends on the data (area imputation, barrio target encoding)
happens inside the training folds (Phase 5). Coverage of each source field is in
`data/08_reporting/eda_missing_values.csv`.

## Features

| Feature | Type | Definition | Source column (`03_primary`) |
|---------|------|------------|------------------------------|
| `area_m2` | float, NaN allowed | Area in m², after cleaning rule 3 | `area_m2` (structured, else text) |
| `area_missing` | bool | `True` when `area_m2` is unknown | `area_m2` |
| `bedrooms` | float, NaN allowed | Number of bedrooms | `bedrooms` (structured `bedrooms`/`rooms`, else text) |
| `bathrooms` | float, NaN allowed | Number of bathrooms | `bathrooms` (structured, else text) |
| `lat`, `lon` | float, NaN allowed | Coordinates in WGS84 (EPSG:4326) | `lat`, `lon` |
| `has_elevator` | bool | Text mentions an elevator ("ascensor") | text, via `has_elevator` |
| `has_pool` | bool | Text mentions a pool ("piscina") | text, via `has_pool` |
| `has_gym` | bool | Text mentions a gym ("gimnasio") | text, via `has_gym` |
| `has_balcony` | bool | Text mentions a balcony ("balcón") | text, via `has_balcony` |
| `has_doorman` | bool | Text mentions a doorman or security ("portería", "vigilancia") | text, via `has_doorman` |
| `is_furnished` | bool | Text says furnished ("amoblado") | text, via `is_furnished` |
| `comuna_code` | category (21 official codes) | Comuna or corregimiento | `comuna_code` (declared name, else coordinates) |
| `estrato` | category (`1`-`6`, `unknown`) | Socioeconomic stratum | `estrato` (declared, else estrato layer) |
| `parking` | category (`yes`, `no`, `unknown`) | Parking stated, stated absent, or not mentioned | `has_parking` |
| `listing_quarter` | category (`2020Q3`-`2021Q3`, `unknown`) | Quarter the listing was published | `start_date` |
| `barrio_name` | string, NaN allowed | Official barrio from the coordinates (experiment only, decision 5) | `barrio_name` |

Category levels are fixed in code (`COMUNAS`, `ESTRATOS`, `PARKING`, `QUARTERS`), so a single
listing at prediction time is encoded exactly like the training data.

## Targets (model input only)

| Column | Definition |
|--------|------------|
| `rent_cop` | Monthly rent in COP (cleaning rule 2: 300,000-30,000,000) |
| `log_rent` | Natural log of `rent_cop`, the training target (decision 1) |

## Not used

Floor, number of parking spots and building age (too sparse, decision 7), the listing text
(may contain personal data) and `text_hash` (only for duplicate detection).
