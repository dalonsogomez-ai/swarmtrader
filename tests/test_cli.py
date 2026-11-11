"""Regresiones de la interfaz publica con datos locales, sin broker ni red."""
from pathlib import Path

from swarmtrader.cli import main


def test_rank_offline_csv_without_optional_congress_data(capsys):
    history = Path(__file__).resolve().parents[1] / "examples" / "history_long_example.csv"

    assert main(["rank", str(history), "--top", "3"]) == 0

    report = capsys.readouterr().out
    assert "TOP 3" in report
    assert "EVIDENCIA" in report
