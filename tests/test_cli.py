"""Regresiones de la interfaz publica con datos locales, sin broker ni red."""
from pathlib import Path

from swarmtrader.cli import main
import swarmtrader.cli as cli


def test_rank_offline_csv_without_optional_congress_data(capsys):
    history = Path(__file__).resolve().parents[1] / "examples" / "history_long_example.csv"

    assert main(["rank", str(history), "--top", "3"]) == 0

    report = capsys.readouterr().out
    assert "TOP 3" in report
    assert "EVIDENCIA" in report


def test_validate_rejects_noise_and_detects_observable_momentum(tmp_path, capsys):
    assert main([
        "validate", "--n-symbols", "40", "--n-days", "900",
        "--population", "10", "--generations", "3",
        "--db", str(tmp_path / "validation.db"),
    ]) == 0

    report = capsys.readouterr().out
    assert "certificado=False" in report
    assert "VALIDACION OK" in report


def test_validate_fails_when_positive_control_has_no_signal(tmp_path, capsys, monkeypatch):
    original_loader = cli.load_synthetic

    def without_signal(**kwargs):
        kwargs["momentum_strength"] = 0.0
        return original_loader(**kwargs)

    monkeypatch.setattr(cli, "load_synthetic", without_signal)
    assert main([
        "validate", "--n-symbols", "40", "--n-days", "900",
        "--population", "10", "--generations", "3",
        "--db", str(tmp_path / "validation.db"),
    ]) == 1

    report = capsys.readouterr().out
    assert "no detecta la senal observable inyectada" in report
    assert "VALIDACION FALLIDA" in report
