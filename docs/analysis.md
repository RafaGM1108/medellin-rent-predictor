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

## Hypothesis tests (`medellin_rent.analysis.hypothesis`)

Source file: `hypothesis_tests.json`. Rents are right-skewed, so all tests are rank-based,
and every p-value comes with an effect size. Bootstrap intervals use the project seed (42),
so re-running gives the same numbers.

**Assumptions.** Listings are treated as independent (duplicates were removed in cleaning,
but one agency may still publish similar units). Mann-Whitney U tests whether one group
tends to have higher values; it reads as a shift in medians only when the two distributions
have similar shapes. Where several comparisons are made, p-values are Holm-adjusted.

### Parking

H0: monthly rent has the same distribution for listings that state they have parking and
listings that state they have none. The 10,991 listings that do not mention parking are
excluded: silence is not "no parking".

| Group | With parking | Without | Median ratio (95% CI) | Rank-biserial r | p (Holm) |
|-------|-------------:|--------:|----------------------:|----------------:|---------:|
| All listings | 14,911 | 193 | 1.65 (1.58-1.77) | 0.68 | < 0.001 |
| Estrato 3 | 1,356 | 49 | 1.10 (1.00-1.15) | 0.14 | 0.099 |
| Estrato 4 | 2,622 | 62 | 1.32 (1.27-1.45) | 0.57 | < 0.001 |
| Estrato 5 | 3,129 | 51 | 1.39 (1.29-1.50) | 0.63 | < 0.001 |

- **Overall, listings with parking ask 65% more** (median 1.90M vs 1.15M COP).
- **Part of that is the estrato**: parking is more common in higher estratos. Within estrato
  3 the difference shrinks to 10% and is not significant; within estratos 4 and 5 it stays
  large (32% and 39%).
- **Limits.** The "without parking" group is small (49-62 per estrato) and comes from phrases
  such as "sin parqueadero". The test uses total rent, not rent per m² (too few listings
  without parking state their area), so parking also partly stands for larger apartments.
  Estratos 1, 2 and 6 have fewer than 20 listings without parking and are not tested.

### Estrato

H0: rent per m² has the same distribution in every estrato (2-6; estrato 1 has fewer than
30 listings with area). n = 7,321 listings with both area and estrato.

- **Kruskal-Wallis rejects H0** (H = 1,640, p < 0.001) with a **large effect**:
  epsilon² = 0.22.
- **The trend is monotonic overall**: Spearman's rho = 0.46 (p < 0.001).
- Adjacent estratos (ratio of median rent per m², higher / lower):

| Step | Median ratio (95% CI) | Rank-biserial r | p (Holm) |
|------|----------------------:|----------------:|---------:|
| 2 → 3 | 0.95 (0.92-1.06) | -0.01 | 0.88 |
| 3 → 4 | 1.24 (1.21-1.26) | 0.40 | < 0.001 |
| 4 → 5 | 1.05 (1.03-1.07) | 0.12 | < 0.001 |
| 5 → 6 | 1.24 (1.21-1.27) | 0.39 | < 0.001 |

- **The price per m² does not climb evenly**: there are two big jumps (3 → 4 and 5 → 6,
  about +24% each), a small one (4 → 5, +5%) and none between 2 and 3. Estrato is better
  treated as a category than as a linear number in the model.

## Map of rent per m² by comuna (`medellin_rent.analysis.rent_map`)

Source files: `rent_map_comunas.png` (committed) and `rent_map_comunas.html` (interactive,
generated by `make analysis` but **not committed**: it embeds the GeoMedellín polygons,
which may not be redistributed). Colors use the median rent per m² from
`price_m2_by_comuna.csv`; comunas with fewer than 30 listings with area are grey.

- **The most expensive comuna is El Poblado**, in the south-east (26,992 COP/m²), followed by
  Altavista (24,000, n = 81), Guayabal (22,000) and Laureles (21,622).
- **The cheapest comunas with enough data are in the north and centre-east**: Castilla
  (11,943, north-west), Aranjuez (15,294, north-east) and Villa Hermosa (15,000,
  centre-east), together with the corregimiento San Antonio de Prado (15,000, south-west).
- **Five areas are grey** (fewer than 30 listings with area): the corregimientos Palmitas
  (none) and Santa Elena (18), and the north-eastern comunas Popular (12), Santa Cruz (14)
  and Doce de Octubre (10).

Boundaries: Alcaldía de Medellín (GeoMedellín), CC BY-SA 4.0.
