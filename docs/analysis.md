# Analysis

All numbers and figures on this page are produced by `make analysis` from
`data/03_primary/listings.parquet` and saved in
[`data/08_reporting/`](https://github.com/RafaGM1108/medellin-rent-predictor/tree/main/data/08_reporting).
Re-running the command regenerates them.

## Exploratory analysis (`medellin_rent.analysis.eda`)

Source files: `eda_summary.csv`, `eda_missing_values.csv`, `eda_rent.png`, `eda_area.png`,
`eda_bedrooms.png`, `eda_missing_values.png`.

### Distributions

| Field | Listings | Median | 25%-75% | 1%-99% |
|-------|---------:|-------:|--------:|-------:|
| Monthly rent (COP) | 26,095 | 1,600,000 | 1,100,000-2,400,000 | 550,000-8,500,000 |
| Area (m²) | 9,004 | 71 | 55-97 | 25-277 |
| Rent per m² (COP) | 9,004 | 21,642 | 17,742-26,667 | 9,413-60,000 |
| Bedrooms | 22,972 | 3 | 2-3 | 1-4 |
| Bathrooms | 25,006 | 2 | 2-2 | 1-5 |

- **Rent is right-skewed.** The mean (2.03M COP) is well above the median (1.60M) and the
  99th percentile is 8.5M. On a log scale the distribution is close to symmetric.
- **Area is concentrated between 55 and 97 m²**, with a long tail of large apartments.
- **Rent per m² is more stable than rent**: its interquartile range spans -18%/+23% around
  the median, against -31%/+50% for rent.
- **Three bedrooms is the most common layout**, followed by two.

### Data quality

Share of the 26,095 primary listings with a value:

| Field | Share | Field | Share |
|-------|------:|-------|------:|
| rent | 100% | barrio | 77.5% |
| bathrooms | 95.8% | parking mentioned | 57.9% |
| bedrooms | 88.0% | area | 34.5% |
| comuna | 80.9% | floor | 5.7% |
| estrato | 79.2% | parking spots | 4.7% |
| coordinates | 79.0% | building age | ~0% |

- **Area is the main gap.** Only about a third of listings state it, so any analysis of rent
  per m² uses 9,004 listings.
- **Floor, number of parking spots and building age are too sparse to use** as features.
- Location (comuna, barrio, estrato) is available for about four in five listings.

## Rent per m² by comuna and estrato (`medellin_rent.analysis.price_m2`)

Source files: `price_m2_by_comuna.csv`, `price_m2_by_estrato.csv`,
`price_m2_comuna_estrato.csv` and the matching PNGs. Only the 9,004 listings that state their
area have a rent per m². Groups with fewer than 30 such listings stay in the CSVs with their
`n` but are left out of the figures and of the conclusions below.

### By comuna

- **El Poblado is the most expensive comuna**: median 26,992 COP/m² (n = 2,332), 25% above
  Laureles (21,622, n = 1,478), the second comuna by number of listings.
- **Castilla is the cheapest** of the comunas with enough data: 11,943 COP/m² (n = 44),
  less than half of El Poblado.
- Most comunas cluster between 17,000 and 22,000 COP/m²: Belén, La Candelaria, Buenos Aires,
  La América, San Javier and Robledo.

### By estrato

| Estrato | Listings | Median COP/m² |
|--------:|---------:|--------------:|
| 2 | 157 | 17,742 |
| 3 | 1,259 | 16,923 |
| 4 | 1,889 | 20,909 |
| 5 | 2,217 | 22,000 |
| 6 | 1,799 | 27,174 |

- **Rent per m² rises with estrato from 3 upwards**: estrato 6 is 61% above estrato 3.
- Estratos 2 and 3 have about the same median; estrato 1 has only 19 listings.

### Comuna and estrato together

- **The comuna matters beyond the estrato.** Within estrato 3 the median ranges from
  11,667 COP/m² in Castilla (n = 38) to 23,913 in Guayabal (n = 45); within estrato 5 from
  18,391 in La América (n = 385) to 26,667 in El Poblado (n = 356).
- **In El Poblado the estrato matters little**: 24,833 (estrato 4), 26,667 (5) and
  27,424 (6) COP/m².
- So a model should use both location and estrato, not one as a proxy for the other.
