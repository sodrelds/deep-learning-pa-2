"""Parte 0 no sintetico: figura da oclusao, testes das metricas, baseline no piso facil e girando
os botoes do gerador. Avalia contra o gt visivel (vis >= 0.3).

No botao da oclusao o IDF1 sobe, porque muita elipse entra atras do oclusor e nao sai ate o fim
do video. Por isso o painel mostra tambem a fracao dos buracos em que o id sobrevive.

  python parte0.py
"""
import json
import os
import runpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import avalia, iou, sobrevivencia
from synth import detector, gera
from tracker import Ingenuo, rastreia

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(AQUI, "figs"), os.path.join(AQUI, "resultados")
BASE = dict(iou_min=0.3, k=5, score_min=0.0)
SEEDS = range(20)
BOTOES = dict(n_obj=[5, 8, 11, 15], vel=[0.5, 1, 2, 3, 4, 6], oclusao=[0, 2, 5, 10, 20, 30])
PADRAO = dict(n_obj=8, vel=1.5, oclusao=0)


def roda(seed, T=60, **gen):
    """Metricas contra o gt visivel e a lista de buracos (duracao, id sobreviveu)."""
    _, gt = gera(T=T, seed=seed, **gen)
    pr = rastreia(detector(gt, ruido=0.03, seed=seed), Ingenuo(), **BASE)
    return avalia(gt[gt[:, 6] >= 0.3], pr), sobrevivencia(gt, pr)


def trechos(x):
    """(inicio, fim) de cada trecho seguido de True, fim exclusivo."""
    d = np.diff(np.r_[0, x.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def id_previsto(pr, g):
    """Id do rastreador em cima da caixa de gt g (frame, id, x, y, w, h), ou None."""
    p = pr[pr[:, 0] == g[0]]
    m = iou(g[2:6], p[:, 2:6])[0] if len(p) else []
    return int(p[np.argmax(m), 1]) if len(p) and max(m) >= 0.5 else None


def figura_oclusao(N=15):
    # procura um video em que uma elipse fica escondida pelo menos N quadros e volta
    for seed in range(200):
        video, gt = gera(n_obj=6, vel=1.5, oclusao=N, T=60, seed=seed)
        for i in np.unique(gt[:, 1]):
            g = gt[gt[:, 1] == i]
            buracos = [(a, b) for a, b in trechos(g[:, 6] < 0.3) if b - a >= N and a >= 3 and b <= 56]
            if buracos:
                break
        if buracos:
            break
    a, b = buracos[0]
    pr = rastreia(detector(gt, ruido=0.03, seed=seed), Ingenuo(), **BASE)
    antes, depois = id_previsto(pr, g[a - 1]), id_previsto(pr, g[b])
    quadros = [a - 3, a - 1, a, (a + b) // 2, b - 1, b, b + 2]

    fig = plt.figure(figsize=(14, 5.2))
    for k, t in enumerate(quadros):
        ax = fig.add_subplot(2, len(quadros), k + 1)
        ax.imshow(video[t], cmap="gray", vmin=0, vmax=1)
        x, y, w, h, vis = g[t, 2:7]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.5, color="lime" if vis >= 0.3 else "red",
                                   ls="-" if vis >= 0.3 else "--"))
        ax.set_title(f"quadro {t + 1}\nvis {vis:.2f}", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(g[:, 0], g[:, 6], "k.-")
    ax.axhspan(0, 0.3, color="red", alpha=0.08)
    ax.axvspan(a + 0.5, b + 0.5, color="gray", alpha=0.2)
    ax.set_xlabel("quadro")
    ax.set_ylabel("visibilidade")
    ax.set_title(f"elipse {int(i)} some por {b - a} quadros atras do oclusor (oclusao={N}) e volta. "
                 f"Ingenuo: id {antes} antes do buraco, id {depois} depois", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte0_oclusao.png"), dpi=130)
    plt.close(fig)
    return dict(seed=seed, elipse=int(i), quadros_escondida=int(b - a), id_antes=antes, id_depois=depois)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    runpy.run_path(os.path.join(AQUI, "metrics.py"), run_name="__main__")
    out = dict(figura=figura_oclusao())
    print("figura:", out["figura"])

    piso = [roda(s, n_obj=5, vel=0.5, oclusao=0)[0] for s in SEEDS]
    out["piso_facil"] = {m: float(np.mean([r[m] for r in piso])) for m in ["IDF1", "IDSW", "razao_ids"]}
    out["piso_facil"]["IDF1_min"] = float(min(r["IDF1"] for r in piso))
    print("piso facil:", out["piso_facil"])

    fig, eixos = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (botao, valores) in zip(eixos, BOTOES.items()):
        linhas = []
        for v in valores:
            rs = [roda(s, **{**PADRAO, botao: v}) for s in SEEDS]
            ms = [r[0] for r in rs]
            buracos = [ok for r in rs for _, ok in r[1]]
            linhas.append(dict(valor=v, **{m: (float(np.mean([r[m] for r in ms])), float(np.std([r[m] for r in ms])))
                                          for m in ["IDF1", "idsw_por_id", "razao_ids"]},
                               buracos=len(buracos), sobrevive=float(np.mean(buracos)) if buracos else None))
            print(botao, v, {k: (round(x[0], 3) if isinstance(x, tuple) else x) for k, x in linhas[-1].items()})
        out[botao] = linhas
        f1 = np.array([l["IDF1"] for l in linhas])
        ax.errorbar(valores, f1[:, 0], f1[:, 1], color="C0", marker="o", capsize=3)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel({"n_obj": "numero de elipses", "vel": "velocidade tipica (px/quadro)",
                       "oclusao": "duracao da oclusao (quadros)"}[botao])
        ax.set_ylabel("IDF1", color="C0")
        ax2 = ax.twinx()
        if botao == "oclusao":
            ax2.plot(valores, [l["sobrevive"] for l in linhas], color="C2", marker="^", ls="--")
            ax2.set_ylabel("fracao dos buracos em que o id sobrevive", color="C2")
            ax2.set_ylim(0, 1.05)
        else:
            sw = np.array([l["idsw_por_id"] for l in linhas])
            ax2.errorbar(valores, sw[:, 0], sw[:, 1], color="C1", marker="s", capsize=3, ls="--")
            ax2.set_ylabel("ID switches por id verdadeiro", color="C1")
            ax2.set_ylim(bottom=0)
    fig.suptitle(f"Baseline ingenuo no sintetico, media e desvio em {len(SEEDS)} videos. "
                 f"Fixo quando nao varia: {PADRAO}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte0_botoes.png"), dpi=130)
    plt.close(fig)
    json.dump(out, open(os.path.join(RES, "parte0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
