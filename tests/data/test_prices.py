import io
import json
from pathlib import Path

import pandas as pd
import pytest

from medellin_rent.data import prices


def _new_layout(values: dict[str, float]) -> bytes:
    """One sheet with a column per month, like the 2024+ subclass annexes."""
    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto"]
    rows = [
        ["Código", "Subclase", "Ponderación", "Índice", *[None] * 7],
        [None, None, None, *months],
        ["01110100", "Arroz", 0.93, *[150.0] * 8],
        [prices.SUBCLASS, "Arriendo efectivo", 10.6, *[values.get(m) for m in months]],
    ]
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, header=False, index=False)
    return buffer.getvalue()


def _old_layout(value: float) -> bytes:
    """Several sheets, index right after the subclass name, like the 2021 annexes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame([["Índice general"]]).to_excel(writer, sheet_name="1", header=False)
        rows = [["Código", "Subclase", "Índice"], [prices.SUBCLASS, "Arriendo efectivo", value]]
        pd.DataFrame(rows).to_excel(writer, sheet_name="8", header=False, index=False)
    return buffer.getvalue()


def test_annex_url_patterns() -> None:
    assert prices.annex_url("2021-08").endswith("/boletines/ipc/anexo_ipc_ago21.xlsx")
    assert prices.annex_url("2026-09").endswith(
        "/IPC/sep2026/anex-IPC-subclases-ingresos-total-sep2026.xlsx"
    )


def test_read_index_both_layouts() -> None:
    assert prices.read_index(_new_layout({"Julio": 138.4, "Agosto": 139.0}), "2026-08") == 139.0
    assert prices.read_index(_old_layout(105.37), "2021-08") == 105.37


def test_read_index_errors() -> None:
    with pytest.raises(ValueError, match=prices.SUBCLASS):
        prices.read_index(_new_layout({"Enero": 1.0}), "2026-08")  # month not published
    workbook = io.BytesIO()
    pd.DataFrame([["x"]]).to_excel(workbook, header=False, index=False)
    with pytest.raises(ValueError, match=prices.SUBCLASS):
        prices.read_index(workbook.getvalue(), "2026-01")


def test_build_rent_index(monkeypatch: pytest.MonkeyPatch) -> None:
    books = {
        prices.annex_url("2021-08"): _old_layout(100.0),
        prices.annex_url("2026-08"): _new_layout({"Julio": 128.0, "Agosto": 130.0}),
    }
    monkeypatch.setattr(prices, "_download", lambda url: books[url])

    record = prices.build_rent_index("2021-08", "2026-08")

    assert (record["base_index"], record["current_index"], record["factor"]) == (100.0, 130.0, 1.3)
    assert record["citation"] == prices.CITATION
    assert record["sources"] == list(books)


def test_load_rent_index(tmp_path: Path) -> None:
    assert prices.load_rent_index(tmp_path / "missing.json") is None
    path = tmp_path / "rent_index.json"
    path.write_text(json.dumps({"factor": 1.2}))
    assert prices.load_rent_index(path) == {"factor": 1.2}
