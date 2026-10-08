"""SHAP interpretation of the selected LightGBM model on the test set.

SHAP values come from LightGBM's own TreeSHAP (``predict(pred_contrib=True)``), which handles
the model's native categories and missing values exactly; they add up to the model's output
for every listing. The model predicts ``log(rent)``, so a SHAP value of ``s`` multiplies the
predicted rent by ``exp(s)``: values are reported as a percentage effect,
``100 * (exp(s) - 1)``.

Writes to ``data/08_reporting``: ``shap_importance.csv``/``.png``, ``shap_by_estrato.csv``,
``shap_by_comuna.csv``, ``shap_dependence_area.png`` and ``shap_by_estrato.png``.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from medellin_rent.analysis.plotting import (
    GRID,
    SERIES,
    TEXT_SECONDARY,
    label_axes,
    new_figure,
    save,
)
from medellin_rent.features.build import FEATURES


def shap_values(model: Any, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values (log-rent units) and the model inputs they explain.

    Args:
        model: Fitted pipeline from :func:`~medellin_rent.model.gradient_boosting.make_lightgbm`.
        data: Listings with ``FEATURES``.

    Returns:
        ``(values, inputs)``: one SHAP column per model input (plus ``base_value``), and the
        transformed inputs, both indexed like ``data``.

    Raises:
        TypeError: If the model is not a LightGBM pipeline.
    """
    booster = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    if booster is None or not hasattr(booster, "booster_"):
        raise TypeError("SHAP interpretation needs the LightGBM pipeline (pred_contrib)")
    inputs = model.named_steps["features"].transform(data[FEATURES])
    contrib = booster.predict(inputs, pred_contrib=True)
    columns = [*inputs.columns, "base_value"]
    return pd.DataFrame(contrib, columns=columns, index=data.index), inputs


def as_percent(values: pd.Series | pd.DataFrame | float) -> Any:
    """Turn log-rent effects into percentage changes of the rent."""
    return 100 * (np.exp(values) - 1)


def importance(values: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, most important first."""
    mean_abs = values.drop(columns="base_value").abs().mean()
    table = pd.DataFrame(
        {
            "feature": mean_abs.index,
            "mean_abs_shap": mean_abs.to_numpy(),
            "mean_abs_effect_pct": as_percent(mean_abs).to_numpy(),
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def by_level(values: pd.DataFrame, inputs: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Mean SHAP effect (%) of a categorical feature for each of its levels."""
    df = pd.DataFrame({"level": inputs[feature].astype("str"), "shap": values[feature]})
    grouped = df.groupby("level")["shap"]
    table = pd.DataFrame({"n": grouped.size(), "mean_effect_pct": as_percent(grouped.mean())})
    return table.reset_index().sort_values("mean_effect_pct", ascending=False, ignore_index=True)


def _percent(value: float, _: object = None) -> str:
    return f"{value:+.0f}%"


def plot_importance(table: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars of the mean absolute effect of each feature."""
    shown = table.iloc[::-1]
    fig, ax = new_figure("What drives the predicted rent (mean |SHAP|, test set)", height=5.0)
    ax.barh(shown["feature"], shown["mean_abs_effect_pct"], color=SERIES, height=0.65)
    for y, value in enumerate(shown["mean_abs_effect_pct"]):
        ax.text(value, y, f"  {value:.1f}%", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, shown["mean_abs_effect_pct"].max() * 1.15)
    label_axes(ax, "Average change in predicted rent caused by the feature", "")
    return save(fig, path)


def plot_area_dependence(values: pd.DataFrame, inputs: pd.DataFrame, path: Path) -> Path:
    """Scatter of the area's effect against the area (listings that state it)."""
    known = inputs["area_m2"].notna()
    area, effect = inputs.loc[known, "area_m2"], as_percent(values.loc[known, "area_m2"])
    fig, ax = new_figure(f"Effect of the area on the predicted rent (n = {known.sum():,})")
    ax.scatter(area.clip(upper=400), effect, s=8, color=SERIES, alpha=0.35, linewidths=0)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    label_axes(ax, "m² (values above 400 shown at 400)", "Effect on predicted rent")
    return save(fig, path)


def plot_estrato(table: pd.DataFrame, path: Path) -> Path:
    """Bars of the mean effect of each estrato level."""
    order = [*[str(e) for e in range(1, 7)], "unknown"]
    shown = table.set_index("level").reindex([o for o in order if o in set(table["level"])])
    fig, ax = new_figure("Effect of the estrato on the predicted rent (mean SHAP, test set)")
    ax.bar(shown.index, shown["mean_effect_pct"], color=SERIES, width=0.6)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    for x, (value, n) in enumerate(zip(shown["mean_effect_pct"], shown["n"], strict=True)):
        offset = 0.3 if value >= 0 else -0.3
        ax.text(
            x,
            value + offset,
            f"{value:+.0f}%\nn={n:,}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.margins(y=0.25)
    label_axes(ax, "Estrato", "Effect on predicted rent")
    return save(fig, path)


def run(model: Any, data: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write SHAP tables and figures for ``data`` (usually the test set) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    values, inputs = shap_values(model, data)
    table = importance(values)
    estrato = by_level(values, inputs, "estrato")
    comuna = by_level(values, inputs, "comuna_code")
    table.to_csv(out_dir / "shap_importance.csv", index=False, float_format="%.4f")
    estrato.to_csv(out_dir / "shap_by_estrato.csv", index=False, float_format="%.2f")
    comuna.to_csv(out_dir / "shap_by_comuna.csv", index=False, float_format="%.2f")
    return [
        out_dir / "shap_importance.csv",
        out_dir / "shap_by_estrato.csv",
        out_dir / "shap_by_comuna.csv",
        plot_importance(table, out_dir / "shap_importance.png"),
        plot_area_dependence(values, inputs, out_dir / "shap_dependence_area.png"),
        plot_estrato(estrato, out_dir / "shap_by_estrato.png"),
    ]


def main() -> None:
    """Explain the saved model on the test set (``make interpret``)."""
    from medellin_rent.pipelines.feature_pipeline.pipeline import TEST_FILE
    from medellin_rent.pipelines.training_pipeline.pipeline import MODEL_FILE
    from medellin_rent.utils.config import get_config
    from medellin_rent.utils.log import get_logger

    paths = get_config().paths
    model = joblib.load(paths.models / MODEL_FILE)
    test = pd.read_parquet(paths.model_input / TEST_FILE)
    for path in run(model, test, paths.reporting):
        get_logger(__name__).info("Wrote %s", path)


if __name__ == "__main__":
    main()
