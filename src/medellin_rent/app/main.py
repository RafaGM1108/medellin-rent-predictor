"""Streamlit app "Is this rent fair?". Run with ``make app``.

Prices an apartment with the saved model (at current prices when ``conf/rent_index.json``
exists), compares the listed rent with the 80% range and explains the prediction.
"""

from typing import Any

import pandas as pd
import streamlit as st

from medellin_rent.app.logic import explain, verdict
from medellin_rent.data.prices import load_rent_index
from medellin_rent.features.build import COMUNA_NAMES
from medellin_rent.inference.predict import download_model, load_model, predict
from medellin_rent.utils.config import get_config

VERDICT_TEXT = {
    "good deal": "Below the usual range for an apartment like this: a good deal.",
    "fair": "Within the usual range for an apartment like this: a fair rent.",
    "expensive": "Above the usual range for an apartment like this: expensive.",
}


@st.cache_resource
def _load() -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    config = get_config()
    paths = config.paths
    if not (paths.models / "model.joblib").is_file() and "model_release_url" in config.params:
        download_model(paths.models, config.params["model_release_url"])
    model, metadata = load_model(paths.models)
    return model, metadata, load_rent_index(paths.rent_index)


def _millions(value: float) -> str:
    return f"{value / 1e6:.2f}M"


st.set_page_config(page_title="Is this rent fair?")
st.title("Is this rent fair?")
st.write(
    "Enter what you know about an apartment in Medellín and the rent you were offered. "
    "The model estimates the usual rent for an apartment like it."
)

try:
    model, metadata, rent_index = _load()
except OSError as error:  # missing files or a failed download
    st.error(
        "No trained model available. Locally, run `make feature` and `make train`; "
        f"online, the model is downloaded from the GitHub release ({error})."
    )
    st.stop()

with st.form("listing"):
    comunas = sorted(COMUNA_NAMES.values())
    comuna = st.selectbox("Comuna", comunas, index=comunas.index("LAURELES"), format_func=str.title)
    rent = st.number_input(
        "Listed monthly rent (COP)", min_value=100_000, value=2_000_000, step=50_000
    )
    estrato = st.selectbox("Estrato", ["I don't know", 1, 2, 3, 4, 5, 6], index=4)
    left, right = st.columns(2)
    area = left.number_input(
        "Area (m²), 0 if you don't know", min_value=0, max_value=1_000, value=0
    )
    parking = right.radio("Parking", ["yes", "no", "unknown"], index=2, horizontal=True)
    bedrooms = left.number_input("Bedrooms", min_value=0, max_value=10, value=2)
    bathrooms = right.number_input("Bathrooms", min_value=0, max_value=10, value=2)
    st.write("The building or apartment has:")
    cols = st.columns(3)
    amenities = {
        "has_elevator": cols[0].checkbox("Elevator"),
        "has_pool": cols[1].checkbox("Pool"),
        "has_gym": cols[2].checkbox("Gym"),
        "has_balcony": cols[0].checkbox("Balcony"),
        "has_doorman": cols[1].checkbox("Doorman / security"),
        "is_furnished": cols[2].checkbox("Furnished"),
    }
    submitted = st.form_submit_button("Check the rent")

if submitted:
    listing = pd.DataFrame(
        [
            {
                "comuna": comuna,
                "estrato": None if estrato == "I don't know" else estrato,
                "area_m2": float(area) if area >= 10 else None,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "parking": parking,
                **amenities,
            }
        ]
    )
    row = predict(model, metadata, listing, rent_index).iloc[0]
    suffix = "_current" if rent_index else ""
    predicted = row["predicted_rent_cop" + suffix]
    low, high = row["interval_low_cop" + suffix], row["interval_high_cop" + suffix]
    result = verdict(float(rent), predicted, low, high)

    st.subheader(VERDICT_TEXT[result.label])
    a, b, c = st.columns(3)
    a.metric("Estimated rent (COP)", _millions(predicted), help=f"{predicted:,.0f} COP per month")
    b.metric("Usual range (80%)", f"{_millions(low)} - {_millions(high)}")
    c.metric("Listed rent vs estimate", f"{result.difference_pct:+.0f}%")

    st.subheader("What drives this estimate")
    st.caption(
        "Effect of each feature on the estimate compared with an average listing in the data."
    )
    drivers = explain(model, listing)
    drivers["effect"] = drivers["effect_pct"].map(lambda v: f"{v:+.0f}%")
    st.table(drivers[["feature", "value", "effect"]].set_index("feature"))

    prices = (
        f"Prices as of {rent_index['current_month']}: model trained on 2020-2021 listings and "
        f"multiplied by {rent_index['factor']:.3f} with DANE's CPI for rent "
        f"({rent_index['citation']})."
        if rent_index
        else "Prices of the training data (2020-2021 listings), not adjusted to current prices."
    )
    test = metadata.get("test", {})
    st.caption(
        f"{prices} On held-out listings the model is off by {test.get('mape', float('nan')):.1f}% "
        "on average. The usual range covers 80% of similar listings. Area and amenities "
        "change the estimate a lot: fill in what you know."
    )
