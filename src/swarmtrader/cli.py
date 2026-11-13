"""CLI.

  tournament   torneo evolutivo walk-forward + certificacion + informe
  validate     tests de control obligatorios (mercado nulo, baselines)
  signals      que compraria el campeon hoy (0..top_k nombres)
  paper        un ciclo de rebalanceo en broker simulado o Alpaca paper
  live-check   pre-vuelo antes de dinero real
  certify      estadisticas de multiple testing de un run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .agents.genome import Genome
from .backtest import FeaturePanel
from .config import load_config
from .data.features import compute_features, compute_regime, dollar_volume
from .data.loaders import load_parquet, load_synthetic, load_yfinance
from .evolution.engine import SwarmConfig, SwarmEngine
from .evolution.meta import MetaLabeler
from .evolution.selection import DeathConfig
from .stats.drift import DriftConfig
from .stats.registry import TrialRegistry


# ------------------------------------------------------------------ helpers
def _signals(args, md):
    """Construye extras y grupos desde los flags --news/--analyst/... si existen."""
    from .data.signals import build_extra, load_groups
    extra = build_extra(
        md.prices.index, list(md.prices.columns), md.prices,
        news=getattr(args, "news", None), analyst=getattr(args, "analyst", None),
        earnings=getattr(args, "earnings", None), insider=getattr(args, "insider", None),
        short=getattr(args, "short", None), fundamentals=getattr(args, "fundamentals", None),
        available_lag=getattr(args, "available_lag", 1), news_scorer=getattr(args, "news_scorer", "auto"),
        nominal=getattr(md, "nominal", None))
    groups = load_groups(args.groups) if getattr(args, "groups", None) else None
    return extra, groups


def _add_drift_flags(sp):
    sp.add_argument("--drift-alpha", type=float, default=0.01,
                    help="G0: tasa nominal de falsa alarma del monitor de deriva; 0 lo desactiva")
    sp.add_argument("--drift-mode", choices=["gate", "warn"], default="gate",
                    help="gate = un estado de mercado fuera de soporte veta el veredicto; "
                         "warn = solo se informa")


def _nonneg_float(text: str) -> float:
    v = float(text)
    if not v >= 0.0:
        raise argparse.ArgumentTypeError("debe ser un numero >= 0")
    return v


def _add_conviction_flags(sp):
    sp.add_argument("--decision", choices=["gated", "conviction"], default="gated",
                    help="gated = sin veredicto accionable no se asigna capital; conviction = "
                         "se asigna la cartera de crecimiento optimo AUNQUE la ventaja no este "
                         "demostrada, sin topes por nombre (puede ser una sola accion)")
    sp.add_argument("--risk-aversion", type=_nonneg_float, default=1.0,
                    help="modo conviction: 1 = Kelly (defecto); <1 concentra mas; 0 = todo a la "
                         "de mayor retorno esperado sin mirar el riesgo; >1 reparte mas")
    sp.add_argument("--allow-cash", action="store_true",
                    help="modo conviction: permitir dejar parte en efectivo (defecto: todo invertido)")
    sp.add_argument("--scenarios", type=int, default=3000,
                    help="modo conviction: escenarios de la distribucion predictiva")


def _conviction_config(args):
    from .portfolio.conviction import ConvictionConfig
    return ConvictionConfig(risk_aversion=args.risk_aversion, allow_cash=args.allow_cash,
                            n_scenarios=args.scenarios)


def _add_signal_flags(sp):
    sp.add_argument("--news", default=None, help="CSV date,symbol,headline|score")
    sp.add_argument("--analyst", default=None, help="CSV date,symbol,consensus[,target]")
    sp.add_argument("--earnings", default=None, help="CSV date,symbol,eps_actual,eps_estimate|surprise_pct")
    sp.add_argument("--insider", default=None, help="CSV date,symbol,value (USD netos)")
    sp.add_argument("--short", default=None, help="CSV date,symbol,short_interest|days_to_cover")
    sp.add_argument("--fundamentals", default=None, help="CSV date(filing),symbol,gross_profit,total_assets,...")
    sp.add_argument("--groups", default=None, help="CSV symbol,sector (industry momentum + neutralizacion)")
    sp.add_argument("--neutralize", action="store_true", help="demean sectorial por fecha (requiere --groups)")
    sp.add_argument("--available-lag", type=int, default=1, help="sesiones de retraso PIT para senales externas")
    sp.add_argument("--news-scorer", choices=["auto", "lexicon", "finbert"], default="auto")


def _build_panel(args) -> FeaturePanel:
    if args.data == "synthetic":
        md = load_synthetic(n_symbols=args.n_symbols, n_days=args.n_days,
                            signal_strength=args.signal, seed=args.seed)
    elif args.data == "parquet":
        md = load_parquet(args.path)
    else:
        symbols = [s.strip().upper() for s in Path(args.symbols).read_text().split() if s.strip()]
        md = load_yfinance(symbols, start=args.start, end=args.end)
    extra, groups = _signals(args, md)
    feats = compute_features(md.prices, md.volumes, extra=extra, groups=groups,
                             neutralize=getattr(args, "neutralize", False))
    return FeaturePanel(feats, md.prices, dollar_volume(md.prices, md.volumes), compute_regime(md.prices))


def _swarm_config(args) -> SwarmConfig:
    cfg, _ = load_config(args.config)
    for k in ("population", "generations", "capital", "cost_bps", "impact", "n_jobs", "seed"):
        v = getattr(args, k, None)
        if v is not None:
            key = {"population": "population_size", "capital": "initial_capital",
                   "impact": "impact_coef"}.get(k, k)
            setattr(cfg, key, v)
    if getattr(args, "death_mode", None):
        cfg.death = DeathConfig(mode=args.death_mode)
    if getattr(args, "no_meta", False):
        cfg.use_meta = False
    return cfg


def _load_champion(path: str) -> tuple[Genome, MetaLabeler | None]:
    spec = json.loads(Path(path).read_text())
    g = Genome.from_dict(spec["genome"])
    meta = None
    md = (spec.get("verdict") or {}).get("meta_model")
    if md and md.get("fitted"):
        from .evolution.meta import META_FEATURE_NAMES
        meta = MetaLabeler()
        meta.fitted, meta.n_train, meta.base_rate, meta.b = True, md["n_train"], md["base_rate"], md["bias"]
        meta.w = np.array([md["weights"][k] for k in META_FEATURE_NAMES])
        meta.mu, meta.sd = np.array(md.get("mu", [0.0] * len(META_FEATURE_NAMES))), np.array(md.get("sd", [1.0] * len(META_FEATURE_NAMES)))
    return g, meta


# ----------------------------------------------------------------- comandos
def cmd_rank(args) -> int:
    """Modo manual: CSV/API de entrada -> top-k con probabilidades calibradas."""
    from .data.universe import etoro_watchlist_symbols, load_symbols_yfinance, load_universe
    from .rank import RankConfig, format_report, rank_universe

    try:
        if args.etoro:
            syms = etoro_watchlist_symbols()
            print(f"eToro watchlist: {len(syms)} simbolos")
            md = load_symbols_yfinance(syms, years=args.years)
        elif args.symbols_list:
            syms = [t.strip().upper() for t in args.symbols_list.split(",") if t.strip()]
            md = load_symbols_yfinance(syms, years=args.years)
        else:
            md = load_universe(args.input, volumes_path=args.volumes, years=args.years)
    except ImportError as exc:
        print(f"falta una dependencia: {exc}\n  -> pip install -e '.[data]'  (o: make install)")
        return 2
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"no se pudo cargar el universo: {exc}")
        return 2

    cfg = RankConfig(horizon=args.horizon, top_k=args.top, success_threshold=args.threshold,
                     model=args.model, oos_months=args.oos_months,
                     min_dollar_volume=args.min_dollar_volume, min_price=args.min_price,
                     neutralize=args.neutralize, available_lag=args.available_lag,
                     news_scorer=args.news_scorer)
    try:
        extra, groups = _signals(args, md)
    except (ValueError, FileNotFoundError) as exc:
        print(f"senal externa invalida: {exc}")
        return 2
    if args.congress:
        from .data.congress import CongressConfig, congress_features, load_disclosures
        try:
            disc = load_disclosures(args.congress)
            pol = congress_features(disc, md.prices, CongressConfig(window=args.congress_window))
            extra = pol if extra is None else extra.join(pol, how="outer")
            print(f"declaraciones STOCK Act: {len(disc)} anclas en fecha de DIVULGACION "
                  f"(nunca de operacion)")
        except (ValueError, FileNotFoundError) as exc:
            print(f"declaraciones invalidas: {exc}")
            return 2
    if extra is not None:
        print(f"senales externas: {', '.join(extra.columns)}")
    res = rank_universe(md, cfg, extra=extra, groups=groups)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(res, cfg))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False, default=str))
        print(f"\nguardado en {args.out}")
    return 0


def cmd_swarm(args) -> int:
    """Enjambre de especialistas + triple filtro estadistico -> cartera o 'no'."""
    listings = None
    from .data.universe import etoro_watchlist_symbols, load_symbols_yfinance, load_universe
    from .portfolio.sizing import SizingConfig
    from .rank import RankConfig
    from .swarm_rank import SwarmRankConfig, format_swarm_report, swarm_rank

    try:
        if args.alpaca:
            from .data.brokers import AlpacaConfig, alpaca_market_data, alpaca_universe
            acfg = AlpacaConfig(years=args.years, paper=not args.live_feed,
                                feed=args.alpaca_feed, max_symbols=args.max_symbols)
            syms = alpaca_universe(acfg)
            print(f"Alpaca: {len(syms)} activos negociables -> descargando barras diarias")
            md = alpaca_market_data(syms, acfg)
            print(f"        {md.prices.shape[1]} con historico utilizable")
        elif args.sharadar:
            from .data.brokers import sharadar_market_data
            md, listings = sharadar_market_data(args.sharadar)
            print(f"Sharadar: {md.prices.shape[1]} simbolos, "
                  f"{int((listings['reason'] != 'active').sum())} deslistados "
                  f"(panel POINT-IN-TIME)")
        elif args.store:
            from .data.store import PanelStore
            st = PanelStore(args.store)
            md, listings = st.to_market_data(), st.listings()
            print(f"almacen local: {md.prices.shape[1]} simbolos x {md.prices.shape[0]} sesiones")
        elif args.etoro:
            syms = etoro_watchlist_symbols()
            print(f"eToro watchlist: {len(syms)} simbolos")
            md = load_symbols_yfinance(syms, years=args.years)
        elif args.symbols_list:
            syms = [t.strip().upper() for t in args.symbols_list.split(",") if t.strip()]
            md = load_symbols_yfinance(syms, years=args.years)
        else:
            md = load_universe(args.input, volumes_path=args.volumes, years=args.years)
    except ImportError as exc:
        print(f"falta una dependencia: {exc}\n  -> pip install -e '.[data,broker,deep,stats]'")
        return 2
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"no se pudo cargar el universo: {exc}")
        return 2

    rc = RankConfig(horizon=args.horizon, top_k=args.top, success_threshold=args.threshold,
                    model=args.model, oos_months=args.oos_months,
                    min_dollar_volume=args.min_dollar_volume, min_price=args.min_price,
                    neutralize=args.neutralize, available_lag=args.available_lag,
                    news_scorer=args.news_scorer)
    from .models.deep import TORCH_AVAILABLE, DeepConfig, best_device
    deep_cfg = DeepConfig(seeds=args.deep_seeds, epochs=args.deep_epochs,
                          seq_len=args.deep_seq, loss=args.deep_loss,
                          device=args.device)
    use_deep = not args.no_deep
    if use_deep and not TORCH_AVAILABLE:
        print("aviso: PyTorch no instalado -> el agente profundo queda inactivo "
              "(pip install -e '.[deep]')")
    elif use_deep:
        print(f"agente profundo: activo  (device={best_device(args.device)}, "
              f"{args.deep_seeds} semillas, perdida={args.deep_loss})")
    cfg = SwarmRankConfig(
        rank=rc, spa_boot=args.spa_boot, spa_alpha=args.alpha, pbo_max=args.pbo_max,
        drift=DriftConfig(alpha=args.drift_alpha, mode=args.drift_mode),
        deep=deep_cfg, allow_deep=use_deep,
        sizing=SizingConfig(kelly_fraction=args.kelly, max_weight=args.max_weight,
                            target_vol=args.target_vol),
        decision=args.decision, conviction=_conviction_config(args),
    )
    try:
        extra, groups = _signals(args, md)
    except (ValueError, FileNotFoundError) as exc:
        print(f"senal externa invalida: {exc}")
        return 2
    if args.congress:
        from .data.congress import CongressConfig, congress_features, load_disclosures
        try:
            disc = load_disclosures(args.congress)
            pol = congress_features(disc, md.prices, CongressConfig(window=args.congress_window))
            extra = pol if extra is None else extra.join(pol, how="outer")
            print(f"declaraciones STOCK Act: {len(disc)} anclas en fecha de DIVULGACION "
                  f"(nunca de operacion)")
        except (ValueError, FileNotFoundError) as exc:
            print(f"declaraciones invalidas: {exc}")
            return 2
    if extra is not None:
        print(f"senales externas: {', '.join(extra.columns)}")

    if listings is not None:
        print("etiqueta point-in-time ACTIVA: las acciones que mueren dentro del "
              "horizonte cuentan con su retorno terminal")
    previous = None
    if args.previous and Path(args.previous).exists():
        prev_raw = json.loads(Path(args.previous).read_text())
        rows = prev_raw.get("portfolio", prev_raw) if isinstance(prev_raw, dict) else prev_raw
        if rows:
            import pandas as _pd
            previous = _pd.Series({r["symbol"]: float(r["weight"]) for r in rows})

    res = swarm_rank(md, cfg, extra=extra, groups=groups, previous=previous,
                     verbose=not args.json, listings=listings)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(format_swarm_report(res, cfg))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False,
                                             default=str))
        print(f"\nguardado en {args.out}")
    # codigo de salida util para automatizar: 0 accionable, 3 no accionable
    return 0 if res.verdict.actionable else 3


def cmd_cycle(args) -> int:
    """Una ejecucion al dia, tras el cierre de EE. UU. Programar con cron/launchd."""
    from .cycle import CycleConfig, DailyCycle
    from .data.labels import TripleBarrierConfig
    from .mandate import Mandate, MandateConfig
    from .swarm_rank import SwarmRankConfig

    listings = None
    try:
        if args.sharadar:
            from .data.brokers import sharadar_market_data
            md, listings = sharadar_market_data(args.sharadar)
        elif args.store:
            from .data.store import PanelStore
            st = PanelStore(args.store)
            md, listings = st.to_market_data(), st.listings()
        elif args.alpaca:
            from .data.brokers import AlpacaConfig, alpaca_market_data, alpaca_universe
            acfg = AlpacaConfig(years=args.years)
            md = alpaca_market_data(alpaca_universe(acfg), acfg)
        else:
            from .data.universe import load_universe
            md = load_universe(args.input, years=args.years)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"no se pudo cargar el universo: {exc}")
        return 2

    # Union de universos: amplia lo que se ANALIZA, no donde se ejecuta.
    if args.venues:
        from .data.venues import KNOWN_VENUES, build_union, min_viable_notional
        names = [v.strip() for v in args.venues.split(",") if v.strip()]
        per_venue = {"alpaca": list(md.prices.columns)}
        for flag, key in ((args.etoro_universe, "etoro_eu"), (args.ibkr_universe, "ibkr")):
            if flag:
                import csv
                with open(flag) as fh:
                    per_venue[key] = [r[0].strip().upper() for r in csv.reader(fh)
                                      if r and r[0].strip()]
        per_venue = {k: v for k, v in per_venue.items()
                     if k in names or k.split("_")[0] in names}
        if len(per_venue) > 1:
            uni = build_union(per_venue)
            s = uni.summary()
            print(f"union de plazas: {s['total_union']} simbolos "
                  f"({s['en_las_tres']} en todas, exclusivos {s['exclusivos_de_cada_plaza']})")
            usable = [x for x in uni.symbols if x in md.prices.columns]
            faltan = s["total_union"] - len(usable)
            if faltan:
                print(f"  aviso: {faltan} simbolos de la union SIN historico cargado; "
                      "se excluyen del analisis en vez de analizarse con datos vacios")
            for name in names:
                spec = KNOWN_VENUES.get(name if name in KNOWN_VENUES else f"{name}_eu")
                if spec is None:
                    continue
                m = min_viable_notional(spec, args.max_cost_pct,
                                        fx=spec.fx_conversion_pct > 0)
                if m > 0:
                    print(f"  {spec.name}: posicion minima viable {m:,.0f} $ "
                          f"para que el coste no supere el {args.max_cost_pct:.0%}")

    if args.broker == "alpaca":
        from .execution.alpaca import AlpacaBroker
        broker = AlpacaBroker(paper=True)
    else:
        from .execution.paper import PaperBroker
        broker = PaperBroker(initial_cash=args.capital)
        broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})

    mandate = Mandate(args.mandate_file, MandateConfig())
    if mandate.state.started_utc is None:
        mandate.start(nav=float(broker.equity), cash=float(broker.cash))
        print("mandato iniciado (timestamp UTC inmutable)")

    scfg = SwarmRankConfig(allow_deep=not args.no_deep, spa_boot=args.spa_boot,
                           drift=DriftConfig(alpha=args.drift_alpha, mode=args.drift_mode),
                           decision=args.decision, conviction=_conviction_config(args))
    ccfg = CycleConfig(
        decision=args.decision,
        dry_run=not args.execute, max_positions=args.max_positions,
        max_position_pct=args.max_position_pct, force=args.force, as_of=args.as_of,
        state_dir=args.state_dir,
        barriers=TripleBarrierConfig(horizon=args.horizon, profit_target=args.profit_target,
                                     stop_loss=args.stop_loss, trail_stop=args.trail_stop),
    )
    rep = DailyCycle(md, broker, mandate, scfg, ccfg, listings=listings).run()
    print(rep.to_json() if args.json else DailyCycle.format_report(rep))
    return 0 if rep.status in ("completed", "skipped") else 3


def _load_for_decision(args):
    """Carga comun de `decide`: panel largo, Sharadar, almacen local, Alpaca o CSV."""
    listings = None
    if args.panel:
        from .data.loaders import load_long_panel
        md = load_long_panel(args.panel, args.prefilter_dollar_volume, args.prefilter_price,
                             start=args.start)
    elif args.sharadar:
        from .data.brokers import sharadar_market_data
        md, listings = sharadar_market_data(args.sharadar)
    elif args.store:
        from .data.store import PanelStore
        st = PanelStore(args.store)
        md, listings = st.to_market_data(), st.listings()
    elif args.alpaca:
        from .data.brokers import AlpacaConfig, alpaca_market_data, alpaca_universe
        acfg = AlpacaConfig(years=args.years, max_symbols=args.max_symbols)
        md = alpaca_market_data(alpaca_universe(acfg), acfg)
    else:
        from .data.universe import load_universe
        md = load_universe(args.input, years=args.years)
    return md, listings


def universe_exclusions(md, lim, max_spread: float = 0.10, window: int = 21):
    """Acciones que no deben recibir capital en la ultima sesion, con el motivo.

    precio    cierre NOMINAL por debajo de `lim.min_price`
    liquidez  mediana de `window` sesiones del volumen en dolares (nominal) por
              debajo de `lim.min_dollar_volume`. La mediana y no el ultimo dia:
              un solo dia raro no debe meter ni sacar una accion.
    spread    spread estimado (Abdi-Ranaldo, maximo/minimo nominales) por encima
              de `max_spread`: una sola ida y vuelta se come esa fraccion.
    attrs: dollar_volume (Series) y price (Series) de la ultima sesion.
    """
    import pandas as pd
    ref = md.nominal if md.nominal is not None else md.prices
    px = ref.iloc[-1].reindex(md.prices.columns).fillna(md.prices.iloc[-1])
    dvm = (ref * md.volumes).tail(window).median().reindex(md.prices.columns)
    rows = [(s, "precio") for s in px.index[px < lim.min_price]]
    rows += [(s, "liquidez") for s in px.index[~(dvm >= lim.min_dollar_volume)]
             if not px.get(s, 0) < lim.min_price]
    seen = {s for s, _ in rows}
    if md.high is not None and md.low is not None and md.nominal is not None and max_spread > 0:
        from .data.labels import abdi_ranaldo_spread
        sp = abdi_ranaldo_spread(md.nominal.tail(64), md.high.tail(64), md.low.tail(64))
        sp = sp.iloc[:-1].ffill().iloc[-1] if len(sp) > 1 else sp.iloc[-1]
        rows += [(s, "spread") for s in sp.index[sp > max_spread] if s not in seen]
    out = pd.DataFrame(rows, columns=["symbol", "reason"])
    out.attrs["dollar_volume"] = dvm
    out.attrs["price"] = px
    return out


def cmd_decide(args) -> int:
    """Analisis -> cartera por conviccion, plazo, entrada, salidas y plan de ordenes."""
    from dataclasses import replace

    from .decision import DecisionConfig, decide, format_decision_report
    from .execution.plan import build_order_plan, plan_risk_limits
    from .rank import RankConfig
    from .swarm_rank import SwarmRankConfig

    try:
        md, listings = _load_for_decision(args)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"no se pudo cargar el universo: {exc}")
        return 2
    print(f"universo: {md.prices.shape[1]} acciones x {md.prices.shape[0]} sesiones "
          f"(ultima {md.prices.index.max().date()})", file=sys.stderr)

    broker, positions = None, {}
    if args.broker != "none":
        try:
            broker = _open_broker(args, live=False)
            positions = broker.positions()
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            print(f"no se pudo leer la cuenta de {args.broker}: {exc}")
            return 2

    # Lo que las guardas de la ejecucion o el broker rechazarian no debe recibir
    # capital: una orden rechazada deja ese dinero sin invertir. Las guardas son
    # las MISMAS que viajan en el plan (antes el analisis usaba las de por defecto
    # aunque se cambiaran --min-price/--min-dollar-volume).
    lim = plan_risk_limits(args.min_price, args.min_dollar_volume)
    excl = universe_exclusions(md, lim, max_spread=args.max_spread)
    exclude = set(excl["symbol"])
    dv = excl.attrs["dollar_volume"]
    if broker is not None:
        exclude |= set(broker.untradable(list(md.prices.columns)))
        # no fraccionables: solo acciones enteras; una accion por encima del
        # `--max-lot-pct` del patrimonio no se puede dimensionar con sentido
        nf = getattr(broker, "non_fractionable", lambda s: set())(list(md.prices.columns))
        last_nom = excl.attrs["price"]
        cap = float(broker.equity) * args.max_lot_pct
        exclude |= {s for s in nf if not (last_nom.get(s, np.inf) <= cap)}
    n_cost = int((excl["reason"] == "spread").sum())
    print(f"excluidas: {len(exclude)} (precio/liquidez {len(excl) - n_cost}, spread {n_cost}, "
          f"broker {len(exclude) - len(excl)})", file=sys.stderr)

    rc = RankConfig(horizon=args.horizon, top_k=args.top, oos_months=args.oos_months,
                    min_dollar_volume=args.min_dollar_volume, min_price=args.min_price)
    scfg = SwarmRankConfig(rank=rc, allow_deep=not args.no_deep, spa_boot=args.spa_boot,
                           drift=DriftConfig(alpha=args.drift_alpha, mode=args.drift_mode))
    dcfg = DecisionConfig(report_top=args.report_top)
    conv = replace(dcfg.conviction, risk_aversion=args.risk_aversion,
                   allow_cash=args.allow_cash, n_scenarios=args.scenarios,
                   lookback=args.lookback)
    if broker is not None:
        eq = float(broker.equity)
        # minimo de orden de Alpaca: 1 USD (con 100 USD, el 1 % del capital)
        conv = replace(conv, min_weight=max(conv.min_weight, 1.05 / eq if eq > 0 else 1.0))
    dcfg = replace(dcfg, conviction=conv)
    extra, groups = _signals(args, md)
    d = decide(md, scfg, dcfg, listings=listings, extra=extra, groups=groups,
               exclude=exclude, verbose=args.verbose)
    print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False, default=str) if args.json
          else format_decision_report(d))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d.to_dict(), indent=2, ensure_ascii=False,
                                             default=str))
    # Todo efectivo (con --allow-cash) es una decision valida: su plan vende lo
    # que haya en cartera y no compra nada.
    decided = not d.weights.empty or d.cash > 1e-9
    if args.plan and decided:
        plan = build_order_plan(d, positions, dollar_volumes=dv.fillna(0.0).to_dict(),
                                min_price=lim.min_price, min_dollar_volume=lim.min_dollar_volume)
        plan.save(args.plan)
        print(f"\nplan de ordenes {plan.plan_id} -> {args.plan}")
    return 0 if decided else 3


def _open_broker(args, live: bool = False):
    """Broker de `decide`/`execute` por nombre. Paper/demo salvo `live`."""
    name = args.broker
    if name == "alpaca":
        from .execution.alpaca import AlpacaBroker
        return AlpacaBroker(paper=not live)
    if name == "etoro":
        from .execution.etoro import EToroBroker
        return EToroBroker(demo=not live)
    if name == "ibkr":
        from .execution.ibkr import IBKRBroker
        port = getattr(args, "ibkr_port", None) or (7496 if live else 7497)
        return IBKRBroker(port=port)
    if name == "manual":
        from .execution.manual import ManualBroker
        prices = json.loads(Path(args.prices).read_text()) if getattr(args, "prices", None) else None
        return ManualBroker(getattr(args, "manual_venue", "lightyear"),
                            cash=getattr(args, "cash", None) or getattr(args, "capital", 0.0),
                            positions_csv=getattr(args, "positions_csv", None), prices=prices)
    return None


def _add_broker_flags(sp):
    sp.add_argument("--ibkr-port", type=int, default=None,
                    help="IBKR: 7497 TWS paper (defecto), 4002 Gateway paper; reales 7496/4001")
    sp.add_argument("--manual-venue", default="lightyear",
                    help="broker=manual: plataforma sin API (las ordenes se escriben en CSV)")
    sp.add_argument("--positions-csv", default=None,
                    help="broker=manual: cartera exportada (symbol,qty,avg_price[,market_value])")
    sp.add_argument("--cash", type=float, default=None, help="broker=manual: efectivo disponible")


def cmd_fundamentals(args) -> int:
    """SEC companyfacts -> tablas point-in-time para --fundamentals y --earnings."""
    from .data import fundamentals as F
    tick = F.load_ticker_map(args.tickers)
    ciks = set(tick["cik"])
    src = Path(args.companyfacts)
    if src.suffix == ".zip":
        facts = F.read_companyfacts_zip(src, ciks=ciks)
    else:
        facts = F.read_companyfacts_parquet(src, args.dei_shares, ciks=ciks)
    P = F.first_reported(facts)
    fund, earn = F.signal_tables(F.pit_snapshots(facts, P), F.quarterly_sue(P), tick)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fund.to_csv(out / "fundamentals_sec.csv", index=False)
    earn.to_csv(out / "earnings_sec.csv", index=False)
    print(f"fundamentales: {len(fund)} filas, {fund['symbol'].nunique()} simbolos -> "
          f"{out / 'fundamentals_sec.csv'}\nresultados (SUE): {len(earn)} filas, "
          f"{earn['symbol'].nunique()} simbolos -> {out / 'earnings_sec.csv'}\n"
          f"uso: decide ... --fundamentals {out / 'fundamentals_sec.csv'} "
          f"--earnings {out / 'earnings_sec.csv'}")
    return 0


def cmd_universe(args) -> int:
    """Union de los universos de varias plataformas, con etiqueta por plaza."""
    from .data import venues as V
    parts = {}
    if args.listed:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from fetch_panel import listed_symbols
        parts["alpaca"] = listed_symbols(args.listed)
    if args.lightyear:
        ly = V.lightyear_universe(None if args.lightyear == "web" else args.lightyear)
        parts["lightyear"] = ly["canonical"].dropna().tolist()
    if args.sec_tickers:
        otc = V.sec_otc_universe(args.sec_tickers)["canonical"].tolist()
        parts["ibkr_otc"] = otc
    if args.etoro:
        parts["etoro_eu"] = V.etoro_universe()["canonical"].dropna().tolist()
    if not parts:
        print("nada que unir: pasa --listed, --lightyear, --sec-tickers y/o --etoro")
        return 2
    u = V.build_union(parts)
    m = u.membership.copy()
    m["us"] = ~m.index.str.contains(r"\.")
    m.to_csv(args.out)
    s = u.summary()
    print(json.dumps({"union": s["total_union"], "por_plaza": s["por_plaza"],
                      "exclusivos": s["exclusivos_de_cada_plaza"],
                      "fuera_de_EEUU": int((~m["us"]).sum()), "csv": args.out},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_execute(args) -> int:
    """Ejecuta un plan de ordenes (tras la apertura). Simulacion salvo --execute."""
    from .execution.plan import LiveConfirmationRequired, execute_plan, load_plan
    from .mandate import Mandate, MandateConfig
    try:
        plan = load_plan(args.plan)
    except (FileNotFoundError, ValueError) as exc:
        print(f"plan invalido: {exc}")
        return 2
    if args.broker != "paper":
        try:
            broker = _open_broker(args, live=args.live)
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            print(f"no se pudo abrir la cuenta de {args.broker}: {exc}")
            return 2
    else:
        from .execution.paper import PaperBroker
        broker = PaperBroker(initial_cash=args.capital)
        if args.prices:
            broker.set_prices(json.loads(Path(args.prices).read_text()))
    mode = "REAL" if args.live else "paper"
    print(f"plan {plan.plan_id}  ({plan.as_of}, entrada {plan.entry_session})  cuenta {mode}"
          f"  {'ENVIO DE ORDENES' if args.execute else 'SIMULACION'}")
    try:
        rep = execute_plan(broker, plan, dry_run=not args.execute, confirm=args.confirm,
                           ledger_path=Path(args.state_dir) / "positions.json",
                           mandate=Mandate(args.mandate_file, MandateConfig()))
    except LiveConfirmationRequired as exc:
        print(str(exc))
        return 4
    print(rep.to_json())
    bad = [o for o in rep.orders if o["status"] == "rejected"]
    return 0 if not bad else 3


def cmd_analogy(args) -> int:
    from .analogy import AnalogyConfig, find_analogues
    from .data.universe import load_universe

    try:
        md = load_universe(args.input, years=args.years)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"no se pudo cargar el universo: {exc}")
        return 2
    if args.symbol not in md.prices.columns:
        print(f"{args.symbol} no esta en el universo cargado")
        return 2
    as_of = args.as_of or md.prices.index[-1]
    res = find_analogues(md.prices, md.volumes, args.symbol, as_of,
                         AnalogyConfig(k=args.k, min_analogues=args.min_analogues,
                                       min_run_up=args.min_run_up, metric=args.metric))
    if args.json:
        print(json.dumps({"symbol": args.symbol, "as_of": str(as_of),
                          "actionable": res.actionable, "reason": res.reason,
                          "n_analogues": res.n_analogues,
                          "p_continue": res.p_continue,
                          "median_max_gain": res.median_max_gain,
                          "median_max_dd": res.median_max_dd,
                          "best_horizon": res.best_horizon,
                          "forward": res.forward.to_dict(orient="index")},
                         indent=2, ensure_ascii=False, default=str))
    else:
        print(f"{args.symbol} en {as_of}\n")
        print(res.summary())
        print("\n  Distribucion empirica de casos pasados comparables, no una prediccion.")
    return 0 if res.actionable else 3


def cmd_mandate(args) -> int:
    from .mandate import Mandate, MandateConfig

    cfg = MandateConfig(target_return=args.target, deadline_days=args.days,
                        max_drawdown=args.max_drawdown, cash_yield_annual=args.cash_yield)
    m = Mandate(args.file, cfg)
    if args.action == "start":
        if args.nav is None:
            print("se requiere --nav para iniciar la evaluacion")
            return 2
        m.start(nav=args.nav, cash=args.cash)
        print("evaluacion iniciada (timestamp UTC inmutable)\n")
    elif args.action == "mark":
        if args.nav is None:
            print("se requiere --nav")
            return 2
        m.mark(nav=args.nav, cash=args.cash, external_flow=args.flow)
    print(m.report())
    return 0


def cmd_tournament(args) -> int:
    from .report import export_run
    panel = _build_panel(args)
    cfg = _swarm_config(args)
    engine = SwarmEngine(panel, cfg, TrialRegistry(args.db))
    print(f"run_id={engine.run_id} simbolos={len(panel.symbols)} sesiones={len(panel.dates)} "
          f"ventana={cfg.eval_window} embargo={cfg.embargo_days} muerte={cfg.death.mode} "
          f"meta={'on' if cfg.use_meta else 'off'} n_jobs={cfg.n_jobs}\n")
    engine.run(verbose=True)
    champ = engine.champion()
    if champ is None:
        print("\nExtincion total.")
        return 1
    verdict = engine.certify(champ)
    if champ.meta is not None and champ.meta.fitted:
        verdict["meta_model"]["mu"] = champ.meta.mu.tolist()
        verdict["meta_model"]["sd"] = champ.meta.sd.tolist()
    print("\n" + "=" * 66)
    print(json.dumps({k: v for k, v in verdict.items() if k != "meta_model"}, indent=2, ensure_ascii=False, default=str))
    print("=" * 66)
    files = export_run(engine, champ, verdict, args.out_dir)
    if not verdict["certified"]:
        print("\nNO despliegues este agente.")
    print("\nficheros:", json.dumps(files, indent=1))
    return 0


def cmd_validate(args) -> int:
    """Controles que deben pasar antes de creerse cualquier resultado."""
    from .data.baselines import run_baselines
    from .rank import RankConfig, rank_universe
    ok = True
    print("[1/2] mercado NULO: el gate no debe certificar nada")
    md = load_synthetic(n_symbols=args.n_symbols, n_days=args.n_days, signal_strength=0.0, seed=args.seed)
    panel = FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                         dollar_volume(md.prices, md.volumes), compute_regime(md.prices))
    cfg = _swarm_config(args)
    eng = SwarmEngine(panel, cfg, TrialRegistry(args.db))
    eng.run(verbose=False)
    v = eng.certify(eng.champion())
    print(f"      Sharpe campeon {v['sharpe_annualized']:+.2f}  P(edge)={v['edge_probability']:.3f}  "
          f"DSR={v['deflated_sharpe_ratio']:.4f}  PBO={v['pbo']['pbo'] if v['pbo'] else 'nan'}  "
          f"certificado={v['certified']}")
    if v["certified"]:
        print("      FALLO: certificado en mercado sin senal. Hay un bug o una fuga."); ok = False
    else:
        print("      OK")

    print("[2/2] control positivo: el ranking debe detectar momentum conocido fuera de muestra")
    # El baseline usa momentum observable; signal_strength solo inyecta un
    # factor latente y no garantiza una relacion con el momentum de precios.
    md = load_synthetic(n_symbols=args.n_symbols, n_days=args.n_days,
                        signal_strength=0.0, momentum_strength=0.10, seed=args.seed + 1)
    panel = FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                         dollar_volume(md.prices, md.volumes), compute_regime(md.prices))
    diag = rank_universe(md, RankConfig()).diagnostics
    print(f"      rank IC OOS={diag.mean_rank_ic:+.3f}  t={diag.ic_tstat:.2f}  "
          f"evidencia={diag.evidence}")
    if diag.mean_rank_ic <= 0 or diag.evidence != "FUERTE":
        print("      FALLO: no detecta la senal observable inyectada"); ok = False
    else:
        print("      OK")
    # Detectar una relacion predictiva no implica que una regla de rebalanceo
    # bata a buy-and-hold en cada trayectoria. Se informa de ambos baselines,
    # sin usar su orden como prueba de correccion del detector de senal.
    print("      baselines netos (diagnostico; no implican rentabilidad real):")
    b = run_baselines(panel, cfg.warmup_days, len(panel.dates), cfg.cost_bps)
    for k, r in b.items():
        print(f"      {k:<18} sharpe={r.sharpe:+.2f} ret={r.total_return:+.3f} dd={r.max_dd:.3f} coste={r.cost_drag:.4f}")
    print("\nVALIDACION", "OK" if ok else "FALLIDA")
    return 0 if ok else 1


def cmd_signals(args) -> int:
    from .live import compute_targets
    genome, meta = _load_champion(args.champion)
    panel = _build_panel(args)
    targets = compute_targets(panel, genome, meta)
    t = len(panel.dates) - 1
    out = {"date": str(panel.dates[t].date()), "targets": targets,
           "cash": round(1.0 - sum(targets.values()), 4), "abstained": not targets}
    if args.json:
        print(json.dumps(out, indent=2)); return 0
    print(f"fecha: {out['date']}")
    if not targets:
        print("ABSTENCION: ningun candidato supera conviccion/meta-modelo. Liquidez.")
        return 0
    for sym, w in sorted(targets.items(), key=lambda kv: -kv[1]):
        print(f"  {sym:<8} peso={w:.1%}")
    print(f"  {'CASH':<8} peso={out['cash']:.1%}")
    return 0


def cmd_paper(args) -> int:
    from .execution.paper import PaperBroker
    from .live import run_once
    _, limits = load_config(args.config)
    genome, meta = _load_champion(args.champion)
    panel = _build_panel(args)
    # Union de universos: amplia lo que se ANALIZA, no donde se ejecuta.
    # (v1.6) el subcomando paper no define --venues: el bloque solo corre si existe
    if getattr(args, "venues", None):
        from .data.venues import KNOWN_VENUES, build_union, min_viable_notional
        names = [v.strip() for v in args.venues.split(",") if v.strip()]
        per_venue = {"alpaca": list(panel.symbols)}
        for flag, key in ((args.etoro_universe, "etoro_eu"), (args.ibkr_universe, "ibkr")):
            if flag:
                import csv
                with open(flag) as fh:
                    per_venue[key] = [r[0].strip().upper() for r in csv.reader(fh)
                                      if r and r[0].strip()]
        per_venue = {k: v for k, v in per_venue.items()
                     if k in names or k.split("_")[0] in names}
        if len(per_venue) > 1:
            uni = build_union(per_venue)
            s = uni.summary()
            print(f"union de plazas: {s['total_union']} simbolos "
                  f"({s['en_las_tres']} en todas, exclusivos {s['exclusivos_de_cada_plaza']})")
            usable = [x for x in uni.symbols if x in panel.symbols]
            faltan = s["total_union"] - len(usable)
            if faltan:
                print(f"  aviso: {faltan} simbolos de la union SIN historico cargado; "
                      "se excluyen del analisis en vez de analizarse con datos vacios")
            for name in names:
                spec = KNOWN_VENUES.get(name if name in KNOWN_VENUES else f"{name}_eu")
                if spec is None:
                    continue
                m = min_viable_notional(spec, args.max_cost_pct,
                                        fx=spec.fx_conversion_pct > 0)
                if m > 0:
                    print(f"  {spec.name}: posicion minima viable {m:,.0f} $ "
                          f"para que el coste no supere el {args.max_cost_pct:.0%}")

    if args.broker == "alpaca":
        from .execution.alpaca import AlpacaBroker
        broker = AlpacaBroker(paper=True)
    else:
        broker = PaperBroker(initial_cash=args.capital)
        t = len(panel.dates) - 1
        broker.set_prices({panel.symbols[j]: float(panel.prices[t, j]) for j in range(len(panel.symbols))})
    rep = run_once(panel, genome, broker, limits, meta, log_path=args.log)
    print(json.dumps(rep.__dict__, indent=2, default=str))
    return 0


def cmd_live_check(args) -> int:
    import os
    checks = {
        "SWARMTRADER_ALLOW_LIVE=1": os.environ.get("SWARMTRADER_ALLOW_LIVE") == "1",
        "ALPACA_API_KEY presente": bool(os.environ.get("ALPACA_API_KEY")),
        "ALPACA_SECRET_KEY presente": bool(os.environ.get("ALPACA_SECRET_KEY")),
        "campeon certificado (DSR+PBO+baseline)": False,
        "registro paper >= 120 ciclos": False,
    }
    p = Path(args.champion)
    if p.exists():
        checks["campeon certificado (DSR+PBO+baseline)"] = bool(json.loads(p.read_text()).get("verdict", {}).get("certified"))
    lp = Path(args.log)
    if lp.exists():
        checks["registro paper >= 120 ciclos"] = sum(1 for _ in lp.open()) >= 120
    print("Pre-vuelo de ejecucion real:")
    for k, v in checks.items():
        print(f"  [{'x' if v else ' '}] {k}")
    ok = all(checks.values())
    print("\n" + ("LISTO para real." if ok else "NO listo. No operes con dinero real."))
    return 0 if ok else 1


def cmd_certify(args) -> int:
    from .stats.deflated_sharpe import effective_n_trials
    from .stats.overfitting import effective_n_trials_clustering, probability_of_backtest_overfitting
    reg = TrialRegistry(args.db)
    m = reg.returns_matrix(args.run_id)
    out = {"n_trials_raw": reg.n_trials(args.run_id), "sharpe_variance": round(reg.sharpe_variance(args.run_id), 6)}
    if m.size:
        out["n_eff_spectral"] = effective_n_trials(m)
        out["n_eff_clustering"] = effective_n_trials_clustering(m)
        out["pbo"] = probability_of_backtest_overfitting(m).summary()
    print(json.dumps(out, indent=2, default=str))
    return 0


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarmtrader")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data(sp):
        sp.add_argument("--data", choices=["synthetic", "parquet", "yfinance"], default="synthetic")
        sp.add_argument("--path", default="data/")
        sp.add_argument("--symbols", default="universe.txt")
        sp.add_argument("--start", default="2016-01-01")
        sp.add_argument("--end", default=None)
        sp.add_argument("--n-symbols", type=int, default=120)
        sp.add_argument("--n-days", type=int, default=1500)
        sp.add_argument("--signal", type=float, default=0.04)
        sp.add_argument("--seed", type=int, default=42)

    def add_swarm(sp):
        sp.add_argument("--config", default=None, help="TOML (ver config/default.toml)")
        sp.add_argument("--population", type=int, default=None)
        sp.add_argument("--generations", type=int, default=None)
        sp.add_argument("--capital", type=float, default=None)
        sp.add_argument("--cost-bps", type=float, default=None)
        sp.add_argument("--impact", type=float, default=None, help="coef. de impacto de mercado")
        sp.add_argument("--n-jobs", type=int, default=None)
        sp.add_argument("--death-mode", choices=["bayesian", "drawdown", "capital", "any"], default=None)
        sp.add_argument("--no-meta", action="store_true")
        sp.add_argument("--db", default="runs/trials.db")

    r = sub.add_parser("rank", help="CSV o API -> top-k a 1 mes con probabilidad calibrada")
    r.add_argument("input", nargs="?", default="universe.csv",
                   help="CSV de tickers, CSV long/wide de precios, .txt o .parquet")
    r.add_argument("--volumes", default=None, help="CSV de volumenes (formato wide)")
    r.add_argument("--symbols-list", default=None, help="tickers separados por comas (sin fichero)")
    r.add_argument("--etoro", action="store_true", help="leer la watchlist via eToro Public API")
    r.add_argument("--years", type=float, default=4.0, help="anos de historico a descargar")
    r.add_argument("--horizon", type=int, default=21)
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--threshold", type=float, default=0.05, help="exito = retorno > umbral")
    r.add_argument("--model", choices=["ridge", "gbdt", "ensemble"], default="ridge")
    r.add_argument("--oos-months", type=int, default=18)
    r.add_argument("--min-dollar-volume", type=float, default=0.0)
    r.add_argument("--min-price", type=float, default=1.0)
    r.add_argument("--congress", default=None,
                   help="CSV de declaraciones STOCK Act con columna disclosure_date")
    r.add_argument("--congress-window", type=int, default=63)
    r.add_argument("--json", action="store_true")
    r.add_argument("--out", default=None, help="guardar resultado JSON")
    _add_signal_flags(r)
    r.set_defaults(func=cmd_rank)

    sw = sub.add_parser("swarm",
                        help="enjambre de especialistas + triple filtro -> cartera o 'no accionable'")
    sw.add_argument("input", nargs="?", default="universe.csv")
    sw.add_argument("--volumes", default=None)
    sw.add_argument("--symbols-list", default=None)
    sw.add_argument("--etoro", action="store_true")
    sw.add_argument("--years", type=float, default=4.0)
    sw.add_argument("--horizon", type=int, default=21)
    sw.add_argument("--top", type=int, default=5)
    sw.add_argument("--threshold", type=float, default=0.05)
    sw.add_argument("--model", choices=["ridge", "gbdt", "ensemble"], default="ridge")
    sw.add_argument("--oos-months", type=int, default=18)
    sw.add_argument("--min-dollar-volume", type=float, default=0.0)
    sw.add_argument("--min-price", type=float, default=1.0)
    sw.add_argument("--spa-boot", type=int, default=2000, help="replicas del bootstrap SPA")
    sw.add_argument("--alpha", type=float, default=0.05, help="nivel de los filtros SPA")
    sw.add_argument("--pbo-max", type=float, default=0.50, help="PBO maximo tolerado")
    _add_drift_flags(sw)
    _add_conviction_flags(sw)
    sw.add_argument("--kelly", type=float, default=0.25, help="fraccion de Kelly")
    sw.add_argument("--max-weight", type=float, default=0.25, help="tope por nombre")
    sw.add_argument("--target-vol", type=float, default=0.15, help="volatilidad anual objetivo")
    sw.add_argument("--congress", default=None,
                    help="CSV de declaraciones STOCK Act (REQUIERE columna disclosure_date; "
                         "usar transaction_date produce alfa falsa, ver data/congress.py)")
    sw.add_argument("--congress-window", type=int, default=63)
    sw.add_argument("--sharadar", default=None,
                    help="CSV de Sharadar SEP (o equivalente): panel POINT-IN-TIME con deslistados")
    sw.add_argument("--store", default=None, help="directorio de un PanelStore local")
    sw.add_argument("--alpaca", action="store_true",
                    help="universo e historico desde Alpaca (ALPACA_API_KEY_ID/SECRET_KEY)")
    sw.add_argument("--alpaca-feed", choices=["iex", "sip"], default="iex")
    sw.add_argument("--live-feed", action="store_true",
                    help="usar la cuenta real de Alpaca en vez de la paper (solo datos)")
    sw.add_argument("--max-symbols", type=int, default=None,
                    help="limitar el universo (util para una primera prueba)")
    sw.add_argument("--no-deep", action="store_true", help="enjambre solo clasico, sin red")
    sw.add_argument("--deep-seeds", type=int, default=3, help="bagging de semillas de la red")
    sw.add_argument("--deep-epochs", type=int, default=40)
    sw.add_argument("--deep-seq", type=int, default=60, help="sesiones de historico por accion")
    sw.add_argument("--deep-loss", choices=["listnet", "sharpe", "mixed"], default="listnet")
    sw.add_argument("--device", default=None, help="cpu | mps | cuda (por defecto: el mejor)")
    sw.add_argument("--previous", default=None, help="JSON previo, para la banda de no-negociacion")
    sw.add_argument("--json", action="store_true")
    sw.add_argument("--out", default=None)
    _add_signal_flags(sw)
    sw.set_defaults(func=cmd_swarm)

    cy = sub.add_parser("cycle",
                        help="ciclo diario completo: salidas, entradas, ordenes y mandato")
    cy.add_argument("input", nargs="?", default="universe.csv")
    cy.add_argument("--sharadar", default=None)
    cy.add_argument("--store", default=None)
    cy.add_argument("--alpaca", action="store_true", help="datos desde Alpaca")
    cy.add_argument("--years", type=float, default=4.0)
    cy.add_argument("--broker", choices=["paper", "alpaca"], default="paper")
    cy.add_argument("--venues", default=None,
                    help="union de universos de varias plazas para el ANALISIS, "
                         "p. ej. alpaca,etoro,ibkr. La ejecucion sigue yendo por "
                         "--broker: solo Alpaca tiene adaptador de ordenes probado")
    cy.add_argument("--etoro-universe", default=None, help="CSV de tickers de eToro")
    cy.add_argument("--ibkr-universe", default=None, help="CSV de tickers de IBKR")
    cy.add_argument("--max-cost-pct", type=float, default=0.02,
                    help="coste de ida y vuelta maximo tolerado por operacion")
    cy.add_argument("--capital", type=float, default=100.0, help="solo para broker=paper")
    cy.add_argument("--execute", action="store_true",
                    help="enviar ordenes de verdad (por defecto solo simula el plan)")
    cy.add_argument("--force", action="store_true")
    cy.add_argument("--as-of", default=None, help="repeticion historica: fecha simulada")
    cy.add_argument("--max-positions", type=int, default=5)
    cy.add_argument("--max-position-pct", type=float, default=0.25)
    cy.add_argument("--horizon", type=int, default=32)
    cy.add_argument("--profit-target", type=float, default=0.10)
    cy.add_argument("--stop-loss", type=float, default=-0.08)
    cy.add_argument("--trail-stop", type=float, default=-0.05)
    cy.add_argument("--mandate-file", default="runs/mandate.json")
    cy.add_argument("--state-dir", default="runs")
    cy.add_argument("--no-deep", action="store_true")
    cy.add_argument("--spa-boot", type=int, default=2000)
    _add_drift_flags(cy)
    _add_conviction_flags(cy)
    cy.add_argument("--json", action="store_true")
    cy.set_defaults(func=cmd_cycle)

    de = sub.add_parser("decide",
                        help="analisis -> cartera por conviccion (sin topes), plazo, entrada, "
                             "salidas y plan de ordenes")
    de.add_argument("input", nargs="?", default="universe.csv")
    de.add_argument("--panel", default=None,
                    help="Parquet largo: date, ticker, adjclose[, close_nominal], volume")
    de.add_argument("--prefilter-dollar-volume", type=float, default=0.0,
                    help="pre-filtro de memoria por liquidez reciente (sesgo de seleccion)")
    de.add_argument("--prefilter-price", type=float, default=0.0)
    de.add_argument("--start", default=None, help="descartar historia anterior (YYYY-MM-DD)")
    de.add_argument("--sharadar", default=None)
    de.add_argument("--store", default=None)
    de.add_argument("--alpaca", action="store_true", help="universo y datos desde Alpaca")
    de.add_argument("--max-symbols", type=int, default=None)
    de.add_argument("--years", type=float, default=6.0)
    de.add_argument("--horizon", type=int, default=21,
                    help="horizonte de ENTRENAMIENTO del enjambre")
    de.add_argument("--top", type=int, default=5)
    de.add_argument("--report-top", type=int, default=10)
    de.add_argument("--oos-months", type=int, default=60)
    de.add_argument("--min-dollar-volume", type=float, default=100_000.0,
                    help="mediana de 21 sesiones del volumen en dolares (defecto 100.000: "
                         "por debajo el cierre suele ser una operacion suelta)")
    de.add_argument("--min-price", type=float, default=0.0,
                    help="precio nominal minimo (defecto 0: penny stocks incluidos; el "
                         "coste estimado por spread decide en su lugar)")
    de.add_argument("--max-spread", type=float, default=0.10,
                    help="spread estimado maximo (ida y vuelta); 0 lo desactiva")
    de.add_argument("--max-lot-pct", type=float, default=0.05,
                    help="no fraccionables: precio de UNA accion como maximo este "
                         "porcentaje del patrimonio")
    de.add_argument("--lookback", type=int, default=756,
                    help="sesiones de historia del optimizador (756 = 3 anos)")
    de.add_argument("--spa-boot", type=int, default=1000)
    de.add_argument("--risk-aversion", type=_nonneg_float, default=1.0,
                    help="1 = Kelly (log); 0 = neutral al riesgo (todo a la de mayor retorno)")
    de.add_argument("--allow-cash", action="store_true",
                    help="permitir que el optimo deje parte en efectivo (defecto: todo invertido)")
    de.add_argument("--scenarios", type=int, default=3000)
    de.add_argument("--no-deep", action="store_true")
    de.add_argument("--broker", choices=["none", "alpaca", "etoro", "ibkr", "manual"],
                    default="none",
                    help="leer la cuenta PAPER/DEMO (posiciones, patrimonio, no operables); "
                         "manual = plataforma sin API (Lightyear) con --positions-csv y --cash")
    de.add_argument("--prices", default=None, help="broker=manual: JSON {simbolo: precio}")
    _add_broker_flags(de)
    de.add_argument("--plan", default=None, help="guardar plan de ordenes JSON")
    de.add_argument("--out", default=None, help="guardar la decision completa JSON")
    de.add_argument("--json", action="store_true")
    de.add_argument("--verbose", action="store_true")
    _add_drift_flags(de)
    _add_signal_flags(de)
    de.set_defaults(func=cmd_decide)

    ex = sub.add_parser("execute", help="ejecutar un plan de ordenes tras la apertura")
    ex.add_argument("plan")
    ex.add_argument("--broker", choices=["paper", "alpaca", "etoro", "ibkr", "manual"],
                    default="paper")
    ex.add_argument("--execute", action="store_true", help="enviar ordenes (defecto: simula)")
    ex.add_argument("--live", action="store_true",
                    help="cuenta REAL (exige SWARMTRADER_ALLOW_LIVE=1 y --confirm <plan_id>)")
    ex.add_argument("--confirm", default=None, help="identificador del plan que se confirma")
    ex.add_argument("--state-dir", default="runs")
    ex.add_argument("--mandate-file", default="runs/mandate.json",
                    help="mandato persistido (el mismo que usa `cycle`); si esta parado "
                         "por drawdown solo se permiten ventas")
    ex.add_argument("--capital", type=float, default=100.0, help="solo broker=paper")
    ex.add_argument("--prices", default=None, help="JSON {simbolo: precio} para broker=paper/manual")
    _add_broker_flags(ex)
    ex.set_defaults(func=cmd_execute)

    fu = sub.add_parser("fundamentals",
                        help="SEC companyfacts -> fundamentales y SUE point-in-time (CSV)")
    fu.add_argument("--companyfacts", required=True,
                    help="companyfacts.zip oficial de la SEC o volcado Parquet equivalente")
    fu.add_argument("--dei-shares", default=None,
                    help="solo Parquet: EntityCommonStockSharesOutstanding en Parquet")
    fu.add_argument("--tickers", required=True, help="company_tickers_exchange.json de la SEC")
    fu.add_argument("--out-dir", default="data")
    fu.set_defaults(func=cmd_fundamentals)

    un = sub.add_parser("universe", help="union de universos de varias plataformas")
    un.add_argument("--listed", default=None, help="nasdaqtraded.txt (cotizadas: Alpaca)")
    un.add_argument("--lightyear", default=None,
                    help="sitemap de acciones de Lightyear descargado, o 'web'")
    un.add_argument("--sec-tickers", default=None,
                    help="company_tickers_exchange.json: OTC con informes a la SEC (IBKR)")
    un.add_argument("--etoro", action="store_true", help="instrumentos de eToro (API, claves)")
    un.add_argument("--out", default="data/universe_union.csv")
    un.set_defaults(func=cmd_universe)

    an = sub.add_parser("analogy", help="casos historicos analogos para una posicion abierta")
    an.add_argument("input", nargs="?", default="universe.csv")
    an.add_argument("--symbol", required=True)
    an.add_argument("--as-of", default=None, help="YYYY-MM-DD (por defecto, ultima sesion)")
    an.add_argument("--years", type=float, default=4.0)
    an.add_argument("--k", type=int, default=50)
    an.add_argument("--min-analogues", type=int, default=20)
    an.add_argument("--min-run-up", type=float, default=0.15)
    an.add_argument("--metric", choices=["euclid", "fisher_rank", "fisher_gauss"],
                    default="euclid",
                    help="regla de distancia; las de Fisher son experimentales "
                         "(ver experiments/infogeo.py)")
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=cmd_analogy)

    mn = sub.add_parser("mandate", help="estado del mandato de evaluacion (meta y plazo)")
    mn.add_argument("action", choices=["start", "mark", "status"])
    mn.add_argument("--file", default="runs/mandate.json")
    mn.add_argument("--nav", type=float, default=None)
    mn.add_argument("--cash", type=float, default=0.0)
    mn.add_argument("--flow", type=float, default=0.0, help="ingreso (+) o retirada (-)")
    mn.add_argument("--target", type=float, default=0.75)
    mn.add_argument("--days", type=int, default=45)
    mn.add_argument("--max-drawdown", type=float, default=0.30)
    mn.add_argument("--cash-yield", type=float, default=0.0)
    mn.set_defaults(func=cmd_mandate)

    t = sub.add_parser("tournament"); add_data(t); add_swarm(t); _add_signal_flags(t)
    t.add_argument("--out-dir", default="runs/latest"); t.set_defaults(func=cmd_tournament)

    v = sub.add_parser("validate"); add_data(v); add_swarm(v); v.set_defaults(func=cmd_validate)

    s = sub.add_parser("signals"); add_data(s)
    s.add_argument("--champion", default="runs/latest/champion.json"); s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_signals)

    pa = sub.add_parser("paper"); add_data(pa)
    pa.add_argument("--champion", default="runs/latest/champion.json"); pa.add_argument("--config", default=None)
    pa.add_argument("--broker", choices=["sim", "alpaca"], default="sim"); pa.add_argument("--capital", type=float, default=50.0)
    pa.add_argument("--log", default="runs/live.jsonl"); pa.set_defaults(func=cmd_paper)

    lv = sub.add_parser("live-check"); lv.add_argument("--champion", default="runs/latest/champion.json")
    lv.add_argument("--log", default="runs/live.jsonl"); lv.set_defaults(func=cmd_live_check)

    c = sub.add_parser("certify"); c.add_argument("--db", default="runs/trials.db"); c.add_argument("--run-id", default=None)
    c.set_defaults(func=cmd_certify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
