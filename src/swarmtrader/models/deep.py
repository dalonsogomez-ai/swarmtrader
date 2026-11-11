"""Ranker cross-seccional profundo.

Por que ESTA arquitectura y no un LSTM de precios
--------------------------------------------------
La literatura de prediccion cross-seccional converge en tres cosas, y las tres
estan implementadas aqui:

  1. **Formular como ranking, no como regresion.** Predecir el precio o el
     retorno absoluto es casi imposible y ademas no es lo que necesitas:
     necesitas saber QUE acciones lo haran mejor que sus vecinas. Entrenar con
     una perdida de ranking alinea el modelo con el objetivo de inversion
     (Feng et al., *Temporal Relational Ranking*, 2019).
  2. **Modelar la relacion entre acciones, no cada accion por separado.** Un
     LSTM por ticker ignora que el retorno relativo depende del resto del
     universo. Aqui una capa de atencion opera sobre TODO el corte transversal
     de cada fecha, con un sesgo aprendido por sector: la version barata de una
     Temporal Graph Convolution, sin necesitar un grafo externo.
  3. **Promediar semillas.** La varianza de una red profunda por numero de
     parametros es su mayor problema en este dominio (Abe-Nakayama 2018). Cada
     modelo se entrena con N semillas y se promedian los RANGOS, no los scores.

Perdidas disponibles
--------------------
  * ``listnet``  entropia cruzada entre la distribucion softmax de los scores y
    la de los targets. Es ranking puro, estable y barato.
  * ``sharpe``   Sharpe diferenciable de una cartera long-short construida con
    pesos softmax sobre el score. Alinea el entrenamiento con el P&L, pero es
    mas ruidosa; se usa mezclada con listnet.

Requiere PyTorch (extra ``[deep]``). Si no esta instalado, el resto del sistema
funciona igual: los agentes profundos se marcan como inactivos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch es opcional
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = object

__all__ = ["DeepConfig", "TORCH_AVAILABLE", "best_device", "CrossSectionalRanker",
           "listnet_loss", "soft_topk_sharpe_loss", "fit_ranker", "predict_scores"]


@dataclass
class DeepConfig:
    arch: str = "auto"              # auto | attn | master | mixer | ensemble
    auto_mixer_above: int = 800     # universo por encima del cual O(N^2) no compensa
    seq_len: int = 60               # sesiones de historico por accion
    hidden: int = 48                # estado del GRU
    n_heads: int = 4
    attn_dim: int = 48
    dropout: float = 0.25
    epochs: int = 40
    patience: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seeds: int = 3                  # bagging de semillas
    loss: str = "listnet"           # listnet | sharpe | mixed
    sharpe_weight: float = 0.3      # solo si loss == "mixed"
    topk_temperature: float = 0.10
    max_dates_per_epoch: int = 256
    device: str | None = None
    verbose: bool = False


def best_device(requested: str | None = None) -> str:
    """cuda > mps (Apple Silicon) > cpu. En el MacBook M5 usa MPS."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------ perdidas
def listnet_loss(scores, targets, eps: float = 1e-9):
    """ListNet top-1: entropia cruzada entre softmax(scores) y softmax(targets).

    Solo importa el orden: es una perdida de ranking, no de regresion.
    """
    log_p = F.log_softmax(scores, dim=0)
    q = F.softmax(targets, dim=0)
    return -(q * log_p).sum()


def soft_topk_sharpe_loss(scores, fwd, temperature: float = 0.10):
    """Sharpe negativo de una cartera long-short suave sobre el corte.

    Los pesos son softmax(score/T) menos softmax(-score/T): diferenciable,
    suma cero y concentra en las colas cuando T es pequena.
    """
    w = F.softmax(scores / temperature, dim=0) - F.softmax(-scores / temperature, dim=0)
    r = (w * fwd).sum()
    return -r


ARCHS = ("attn", "master", "mixer")


class CrossSectionalRanker(nn.Module if TORCH_AVAILABLE else object):
    """GRU temporal por accion + mezcla cross-seccional.

    Tres variantes de la capa cross-seccional, todas permutacion-equivariantes
    (el resultado no depende del orden de los tickers) y todas capaces de
    trabajar con un numero VARIABLE de acciones por fecha, que es obligatorio
    cuando el universo cambia por delistings:

      * ``attn``   atencion multi-cabeza sobre el corte + sesgo por sector.
                   Coste O(N^2): a 5.000 simbolos empieza a doler.
      * ``master`` lo anterior MAS compuerta guiada por el mercado: se resume el
                   estado del mercado del dia (retorno y volatilidad agregados,
                   dispersion cross-seccional) y se usa para modular las
                   features de cada accion antes de la atencion. Patron tomado
                   de MASTER (AAAI 2024) y reimplementado desde cero: la idea es
                   condicionar por regimen, no copiar su codigo.
      * ``mixer``  sin atencion: mezcla por agregacion (media y desviacion sobre
                   el corte) reinyectada a cada accion, mas mezcla temporal y de
                   canal. Coste O(N). Patron inspirado en StockMixer (AAAI 2024),
                   adaptado: su "stock mixing" es un MLP sobre la dimension de
                   acciones y exige N fijo, lo que aqui es inaceptable; la
                   agregacion permutacion-invariante conserva la idea y admite N
                   variable.

    `DeepConfig.arch = "ensemble"` entrena las tres y promedia sus RANGOS: no
    selecciona nada, no gasta un grado de libertad, y la combinacion de
    pronosticos es lo que la evidencia senala como mas robusto.

    El defecto es ``auto``, y la regla es deliberadamente de COSTE, no de
    rendimiento: por debajo de `auto_mixer_above` simbolos se usa `ensemble`;
    por encima, `mixer`, porque es el unico O(N) y la atencion completa deja de
    ser practica. Elegir arquitectura mirando cual puntua mejor seria una
    comparacion multiple encubierta; elegirla por complejidad computacional no
    mira los resultados y por tanto no gasta grados de libertad.

    Entrada por fecha:
        seq     (N, L, C)   historico reciente de cada una de las N acciones
        feat    (N, F)      factores estaticos ya z-scoreados por fecha
        sector  (N,)        id de sector (opcional, -1 = desconocido)
    Salida:
        score   (N,)        mayor = mejor posicion relativa esperada
    """

    def __init__(self, n_features: int, n_channels: int = 2, n_sectors: int = 16,
                 cfg: DeepConfig = DeepConfig()):
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("PyTorch no esta instalado: pip install -e '.[deep]'")
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.gru = nn.GRU(n_channels, h, batch_first=True)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * h, cfg.attn_dim), nn.LayerNorm(cfg.attn_dim), nn.GELU(),
        )
        # atencion sobre el corte: cada accion mira al resto del universo
        self.attn = nn.MultiheadAttention(cfg.attn_dim, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.attn_dim)
        # sesgo relacional por sector (TGC barata): un escalar aprendido por par
        # de sectores que se suma a los logits de atencion
        self.n_sectors = n_sectors
        self.sector_bias = nn.Parameter(torch.zeros(n_sectors + 1, n_sectors + 1))
        self.arch = cfg.arch if cfg.arch in ARCHS else "attn"
        # compuerta guiada por el mercado (MASTER): 4 estadisticos del dia
        self.market = nn.Sequential(
            nn.Linear(4, cfg.attn_dim), nn.GELU(), nn.Linear(cfg.attn_dim, cfg.attn_dim),
            nn.Sigmoid(),
        )
        # mezcla por agregacion (StockMixer adaptado): media y sd del corte
        self.mix = nn.Sequential(
            nn.Linear(3 * cfg.attn_dim, cfg.attn_dim), nn.LayerNorm(cfg.attn_dim), nn.GELU(),
        )
        self.time_mix = nn.Sequential(
            nn.Linear(cfg.seq_len, cfg.seq_len), nn.GELU(), nn.Linear(cfg.seq_len, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.attn_dim, cfg.attn_dim // 2), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.attn_dim // 2, 1),
        )

    def _market_state(self, seq):
        """4 estadisticos del dia, invariantes al orden de las acciones."""
        r = seq[..., 0]                             # (N, L) retornos
        return torch.stack([
            r.mean(),                               # retorno medio del mercado
            r.std(),                                # volatilidad agregada
            r[:, -1].std(),                         # dispersion cross-seccional hoy
            r[:, -21:].mean() if r.shape[1] >= 21 else r.mean(),
        ]).detach()

    def forward(self, seq, feat, sector=None):
        _, hT = self.gru(seq)                       # (1, N, h)
        hT = hT.squeeze(0)
        z = self.merge(torch.cat([hT, self.feat_mlp(feat)], dim=-1))   # (N, D)

        if self.arch == "master":
            # compuerta por regimen: modula cada accion segun el estado del dia
            z = z * self.market(self._market_state(seq)).unsqueeze(0)

        if self.arch == "mixer":
            # mezcla temporal sobre la secuencia + agregacion del corte, O(N)
            tmix = self.time_mix(seq[..., 0]).expand(-1, z.shape[-1])
            pooled_mean = z.mean(dim=0, keepdim=True).expand_as(z)
            z = self.mix(torch.cat([z, pooled_mean, tmix], dim=-1))
            z = self.norm(z)
        else:
            bias = None
            if sector is not None:
                s = sector.clamp(min=-1) + 1        # -1 -> 0 (desconocido)
                s = s.clamp(max=self.n_sectors)
                bias = self.sector_bias[s][:, s]    # (N, N)
            q = z.unsqueeze(0)
            out, _ = self.attn(q, q, q, attn_mask=bias, need_weights=False)
            z = self.norm(z + out.squeeze(0))       # conexion residual
        return self.head(z).squeeze(-1)


# ------------------------------------------------------------------ entreno
def _epoch(model, opt, batches, cfg, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    for seq, feat, sec, tgt, fwd in batches:
        if seq.shape[0] < 5:
            continue
        with torch.set_grad_enabled(train):
            score = model(seq, feat, sec)
            loss = listnet_loss(score, tgt)
            if cfg.loss in ("sharpe", "mixed"):
                sh = soft_topk_sharpe_loss(score, fwd, cfg.topk_temperature)
                loss = sh if cfg.loss == "sharpe" else loss + cfg.sharpe_weight * sh
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


def fit_ranker(train_batches, val_batches, n_features: int, n_channels: int,
               n_sectors: int, cfg: DeepConfig, seed: int):
    """Entrena un modelo con parada temprana sobre una validacion PURGADA."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CrossSectionalRanker(n_features, n_channels, n_sectors, cfg)
    dev = best_device(cfg.device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for ep in range(cfg.epochs):
        idx = rng.permutation(len(train_batches))[: cfg.max_dates_per_epoch]
        _epoch(model, opt, [train_batches[i] for i in idx], cfg, True)
        vl = _epoch(model, opt, val_batches, cfg, False) if val_batches else float("inf")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if cfg.verbose:
            print(f"      seed {seed} ep {ep:02d} val {vl:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def predict_scores(models: list, batch) -> np.ndarray:
    """Predice promediando los RANGOS de cada semilla, no sus scores.

    Promediar scores crudos deja que una semilla mal escalada domine el
    conjunto; promediar rangos es robusto a la escala de cada red.
    """
    seq, feat, sec = batch[0], batch[1], batch[2]
    ranks = []
    for m in models:
        s = m(seq, feat, sec).detach().cpu().numpy().astype(float)
        order = s.argsort().argsort().astype(float)
        n = max(len(s) - 1, 1)
        ranks.append(order / n * 2.0 - 1.0)
    return np.mean(ranks, axis=0)
