# Findings and modelling decisions

Summary of Phases 1-3 and the decisions they lead to for the feature and training pipelines
(Phases 4-5). Numbers come from `data/08_reporting/` (see [Analysis](analysis.md)) and the
data profile in [`data/README.md`](https://github.com/RafaGM1108/medellin-rent-predictor/blob/main/data/README.md).

## Key findings

1. **The data.** 26,095 distinct apartment rentals in Medellín (Properati, published
   2020-07-26 to 2021-08-19), after removing 76,053 re-published duplicates.
2. **Location is good, size is not.** 80.9% of listings have a comuna, 79.2% an estrato and
   77.5% a barrio, but only 34.5% state their area. Listings with and without area have the
   same median rent (1.6M COP) and a similar estrato mix, so the area subset is not
   obviously biased.
3. **Rent is right-skewed** (median 1.60M, mean 2.03M, 99th percentile 8.5M COP) and
   roughly symmetric on a log scale.
4. **Location drives price.** Median rent per m² goes from 11,943 COP (Castilla) to 26,992
   (El Poblado). The comuna matters beyond the estrato: within estrato 3 it ranges from
   11,667 to 23,913.
5. **Estrato matters, unevenly.** Large overall effect (epsilon² = 0.22), with jumps at
   3 → 4 and 5 → 6 (about +24% each), a small step at 4 → 5 (+5%) and none at 2 → 3.
6. **Parking is partly a proxy.** ×1.65 in median rent overall, but ×1.10 and not
   significant within estrato 3; ×1.32-1.39 within estratos 4-5.
7. **Some fields are too sparse to use**: floor (5.7%), number of parking spots (4.7%),
   building age (~0%).

## Decisions for Phases 4-5

| # | Decision | Why |
|---|----------|-----|
| 1 | **Target: monthly rent in COP, modelled as `log(rent)`**; metrics (MAE, RMSE, MAPE) reported back in COP | Right-skewed rent (finding 3); errors in COP are what a renter understands |
| 2 | **Train on listings with a comuna** (21,112); drop the rest | The app always asks for a location; without it a listing teaches little (finding 4) |
| 3 | **Keep listings without area.** `area_m2` stays a feature with missing values; LightGBM handles them natively, Ridge and Random Forest get a median imputation by bedrooms plus an `area_missing` flag | Dropping them would lose 65% of the data (finding 2) |
| 4 | **Estrato and comuna as categories**, with an `unknown` estrato level; not as numbers | Uneven estrato steps (finding 5) |
| 5 | **Location features**: comuna (category), latitude/longitude where present. Barrio (260 values, 113 with at least 30 listings) only as an experiment with target encoding inside the CV folds | Avoids leakage and very sparse categories |
| 6 | **Other features**: bedrooms, bathrooms, `has_parking` as three levels (yes / no / not mentioned), the six amenity flags, and the listing quarter | Quarter captures the 2020-2021 price trend; "not mentioned" is not "no" |
| 7 | **Not used**: floor, number of parking spots, building age, listing text | Too sparse (finding 7); text can contain personal data |
| 8 | **Validation: 5-fold CV**, shuffled with the project seed, plus a held-out test set (20%) | As in the roadmap; near-duplicates from one agency may make CV slightly optimistic, noted in the results |
| 9 | **Baseline**: median rent per m² of the comuna × area when the area is known, otherwise the median rent of the comuna | A per-m² baseline alone would fail for 65% of listings (decision 3) |
| 10 | **Prices are 2020-2021**: the app and API must bring predictions to current prices with an official index before comparing with a rent listed today | Tracked in #67 |

## Follow-up issues

- #67: adjust 2020-2021 rents to current prices for the app.
- #26 updated to the baseline in decision 9.
