# Data

Full source survey, licenses and file profile: [`data/README.md`](https://github.com/RafaGM1108/medellin-rent-predictor/blob/main/data/README.md).

## Sources

| Source | What it gives | License | How it is collected |
|--------|---------------|---------|---------------------|
| Properati Colombia (`co_properties.csv`, Kaggle mirror of Properati Data) | Rental listings: price, text, rooms, bathrooms, area, coordinates, barrio | Unknown on Kaggle; original not verified | Manual download, then `make listings` |
| GeoMedellín OD1043, Límite Catastral de Comunas y Corregimientos | Comuna boundaries | CC BY-SA 4.0, may not be "comercializada o transferida" | `make geo` |
| GeoMedellín OD1044, Límite Catastral de Barrios y Veredas | Barrio boundaries and barrio → comuna | CC BY-SA 4.0, may not be "comercializada o transferida" | `make geo` |
| DANE IPC, subclass 04130100 Arriendo efectivo | Price adjustment of predictions to current prices | Use with citation, non-commercial | `make prices` |
| GeoMedellín OD396, Estrato Socioeconómico | Estrato by location | CC BY-SA 4.0, may not be "comercializada o transferida" | `make geo` |

## Collection process

1. **Listings.** Sign in to Kaggle, download
   [Colombian Properties](https://www.kaggle.com/datasets/lauramartinezortiz/colombian-properties)
   and unzip `co_properties.csv` into `data/01_raw/listings/`. Then run `make listings`: it
   checks the file, writes `listings.source.json` (URL, license note, SHA-256, size, date) and
   loads the Medellín apartment rentals.
2. **Boundaries.** `make geo` downloads both GeoJSON layers from GeoMedellín with an
   identified user agent and writes `<layer>.source.json` (URL, license, date) next to each.
3. Raw files are never edited. Cleaning happens in `02_intermediate` and later layers.

## What is committed

Nothing from these sources. The Properati license is unknown and the geographic layers may
not be transferred, so no sample is committed either. The tests use a **synthetic** fixture
(`tests/data/fixtures/co_properties.csv`) in the same format.

## Ethics

- **No scraping.** Fincaraíz, Metrocuadrado, Ciencuadras and MercadoLibre forbid it in their
  terms of use. Espacio Urbano reserves all rights.
- **Personal data.** Emails and mobile numbers are redacted when listings are loaded. Agent
  names may remain in the text, so `title` and `description` only live in git-ignored layers
  and are dropped once features are extracted.
- **Attribution.** Outputs built on the geographic layers credit the Alcaldía de Medellín
  (GeoMedellín).
