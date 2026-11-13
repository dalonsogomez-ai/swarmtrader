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


def test_state_paths_are_per_account():
    """Cada cuenta lleva su mandato y su libro; las rutas explicitas mandan."""
    from argparse import Namespace

    def ns(**kw):
        return Namespace(**{"state_dir": None, "mandate_file": None, **kw})

    assert cli._state_paths(ns(broker="paper")) == (
        Path("runs/paper-local"), Path("runs/paper-local/mandate.json"))
    assert cli._state_paths(ns(broker="alpaca"))[0] == Path("runs/alpaca-paper")
    assert cli._state_paths(ns(broker="alpaca"), live=True)[0] == Path("runs/alpaca-live")
    assert cli._state_paths(ns(broker="manual", manual_venue="lightyear"))[0] == Path(
        "runs/manual-lightyear")
    assert cli._state_paths(ns(broker="alpaca", state_dir="x")) == (
        Path("x"), Path("x/mandate.json"))
    assert cli._state_paths(ns(broker="alpaca", mandate_file="m.json")) == (
        Path("runs/alpaca-paper"), Path("m.json"))


def test_cycle_simulation_does_not_start_the_mandate(monkeypatch, capsys):
    """Un `cycle` sin --execute no arranca el reloj del mandato; con --execute si,
    y en el directorio de su cuenta."""
    from types import SimpleNamespace

    import swarmtrader.cycle as cycle_mod
    import swarmtrader.data.universe as universe_mod
    from swarmtrader.data.loaders import load_synthetic

    seen = {}

    class StubCycle:
        def __init__(self, md, broker, mandate, scfg, ccfg, listings=None):
            seen["mandate"], seen["cfg"] = mandate, ccfg

        def run(self):
            return SimpleNamespace(status="completed")

        @staticmethod
        def format_report(rep):
            return "ciclo simulado"

    monkeypatch.setattr(cycle_mod, "DailyCycle", StubCycle)
    monkeypatch.setattr(universe_mod, "load_universe",
                        lambda *a, **k: load_synthetic(n_symbols=5, n_days=60, seed=1))

    assert main(["cycle", "universo.csv"]) == 0
    assert seen["mandate"].state.started_utc is None
    assert seen["cfg"].state_dir == str(Path("runs") / "paper-local")
    assert not Path("runs").exists()

    assert main(["cycle", "universo.csv", "--execute"]) == 0
    assert seen["mandate"].state.started_utc is not None
    assert [p.name for p in Path("runs").iterdir()] == ["paper-local"]
    assert (Path("runs") / "paper-local" / "mandate.json").exists()
