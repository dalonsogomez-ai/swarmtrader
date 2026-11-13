"""EXPLORATORIO (posterior al preregistro): el control C1 fallo por falta de potencia con 80 fechas.
Aqui se promedia sobre 100 realizaciones del ruido sintetico para obtener la curva ESPERADA bajo
IC uniforme 0,075 (demostracion del mecanismo, no contraste). Tambien tasas base del universo."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import norm
rng = np.random.default_rng(7)
# uso: python experiments/universe_size_control_mc.py <oos.parquet> [salida.json]
o = pd.read_parquet(sys.argv[1]).dropna(subset=["score", "fwd"])
GRID = [100, 250, 500, 1000, 2000]; REAL, DRAWS, K = 100, 20, 5
acc = {N: [] for N in GRID + ["todo"]}; base25, base15 = [], []
for _, g in o.groupby("date"):
    r = g["fwd"].to_numpy(); n = len(r)
    base25.append((r > 0.25).mean()); base15.append((r < -0.15).mean())
    zr = norm.ppf((pd.Series(r).rank().to_numpy() - 0.5) / n)
    per = {N: [] for N in acc}
    for _ in range(REAL):
        s = 0.075 * zr + np.sqrt(1 - 0.075**2) * rng.standard_normal(n)
        top = np.argpartition(-s, K - 1)[:K]; per["todo"].append(r[top].mean() - r.mean())
        for N in GRID:
            for _ in range(DRAWS):
                idx = rng.choice(n, N, replace=False); t = idx[np.argpartition(-s[idx], K - 1)[:K]]
                per[N].append(r[t].mean() - r[idx].mean())
    for N in acc: acc[N].append(np.mean(per[N]))
res = {"curva_esperada_IC_uniforme_0075_K5": {str(N): round(float(np.mean(v)), 5) for N, v in acc.items()},
       "tasa_base_universo": {"p_mas25": round(float(np.mean(base25)), 4), "p_menos15": round(float(np.mean(base15)), 4)}}
print(json.dumps(res, indent=2)); json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "universe_size_control_mc.json", "w"), indent=2)
