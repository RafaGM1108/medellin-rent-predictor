"""Bring 2020-2021 rents to current prices with DANE's CPI for effective rent.

Index: DANE *IPC*, subclass 04130100 "Arriendo efectivo" (national; base December
2018 = 100). DANE publishes the pure rent subclass only nationally; for Medellín only the
whole housing division (rent plus utilities) exists. Terms of use: use and transformation
allowed with the citation in ``CITATION``; no commercial exploitation.

``make prices`` reads the index for the base month (last month of the training data) and
the current month set in ``conf/base.yaml`` from DANE's monthly annexes and writes
``conf/rent_index.json``. The model's predictions (August 2021 prices) are multiplied by
``current_index / base_index``.
"""

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from medellin_rent.geo.boundaries import USER_AGENT

SUBCLASS = "04130100"  # Arriendo efectivo
CITATION = "Fuente: Departamento Administrativo Nacional de Estadística: www.dane.gov.co"
_MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
_MONTH_NAMES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]


def annex_url(month: str) -> str:
    """URL of DANE's monthly CPI annex for ``month`` (``YYYY-MM``).

    Annexes up to 2023 live in ``investigaciones/boletines/ipc`` with one sheet per
    division; later ones in ``operaciones/IPC`` with one sheet of subclass indices.
    """
    year, number = int(month[:4]), int(month[5:7])
    tag = _MONTHS[number - 1]
    # ponytail: old pattern checked for 2021 and new one for 2024-2026; 2022-2023 unverified.
    if year <= 2023:
        return (
            "https://www.dane.gov.co/files/investigaciones/boletines/ipc/"
            f"anexo_ipc_{tag}{year % 100:02d}.xlsx"
        )
    return (
        f"https://www.dane.gov.co/files/operaciones/IPC/{tag}{year}/"
        f"anex-IPC-subclases-ingresos-total-{tag}{year}.xlsx"
    )


def read_index(workbook: bytes, month: str) -> float:
    """Find the effective-rent index for ``month`` in a DANE annex.

    Handles both layouts: old annexes (one sheet per division, the month's index in the
    column after the subclass name) and new ones (one sheet, one column per month).

    Raises:
        ValueError: If the subclass or the month is not in the workbook.
    """
    month_name = _MONTH_NAMES[int(month[5:7]) - 1]
    sheets = pd.read_excel(io.BytesIO(workbook), sheet_name=None, header=None, dtype=str)
    for sheet in sheets.values():
        rows = sheet.index[sheet[0].astype(str).str.strip() == SUBCLASS]
        if rows.empty:
            continue
        row = sheet.loc[rows[0]]
        header = sheet.apply(lambda col: col.astype(str).str.strip()).eq(month_name)
        month_columns = header.any()
        if month_columns.any():  # new layout: a column per month
            column = month_columns[month_columns].index[0]
            value = row[column]
        else:  # old layout: the annex is for this month only; index after the name
            value = row[2]
        if pd.notna(value) and str(value).strip() not in ("", "nan"):
            return float(value)
    raise ValueError(f"Subclass {SUBCLASS} for {month} not found in the annex")


def _download(url: str, timeout: float = 120) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    # The URL is built from fixed DANE https patterns, never user input.
    with urlopen(request, timeout=timeout) as response:  # nosec B310
        return bytes(response.read())


def build_rent_index(base_month: str, current_month: str) -> dict[str, Any]:
    """Download both annexes and return the adjustment record."""
    base = read_index(_download(annex_url(base_month)), base_month)
    current = read_index(_download(annex_url(current_month)), current_month)
    return {
        "index": "IPC, subclase 04130100 Arriendo efectivo (nacional, base dic 2018 = 100)",
        "base_month": base_month,
        "base_index": base,
        "current_month": current_month,
        "current_index": current,
        "factor": round(current / base, 4),
        "sources": [annex_url(base_month), annex_url(current_month)],
        "citation": CITATION,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def load_rent_index(path: Path) -> dict[str, Any] | None:
    """The saved adjustment record, or ``None`` if ``make prices`` has not been run."""
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text()))


def main() -> None:
    """``make prices``: write ``conf/rent_index.json`` for the months in the config."""
    from medellin_rent.utils.config import get_config

    config = get_config()
    months = config.params["rent_index"]
    record = build_rent_index(months["base_month"], months["current_month"])
    config.paths.rent_index.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
