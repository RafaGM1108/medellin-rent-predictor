# Data

## Listings

Scraping Colombian real-estate portals is **not** used: Fincaraíz, Metrocuadrado,
Ciencuadras and MercadoLibre forbid it in their terms of use, and Espacio Urbano reserves all
rights. Listings come from a public dataset loaded by `medellin_rent.data` (see the source
survey in `data/README.md`).

## Geography

Comunas/corregimientos boundaries and the estrato layer come from Medellín's open data portal
(GeoMedellín / MEData), CC BY-SA 4.0. They cannot be redistributed ("no puede ser
comercializada o transferida"), so they are downloaded by `medellin_rent.geo` at run time and
never committed.

## Ethics

- No scraping where robots.txt or the terms of use forbid it.
- No personal data (names, phone numbers, emails) is kept, even if a dataset contains it.
- Data that cannot be redistributed is not committed; only code and, where the license
  allows, small samples are.
