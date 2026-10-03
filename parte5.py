"""Parte 5: qualidade do detector, sem retreinar. As deteccoes SDP perdem caixas e ganham ruido e
falsos positivos em tres niveis, com o mesmo sorteio escalado entre eles. Usa o rastreador da
Parte 4 sem a correcao, que piorou a 10.

  python parte5.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import mot
from metrics import ap50, avalia, map50_quadros, tira_distratores
from model import PreditorRNN, carrega
from tracker import rastreia


CKPT = os.path.join(mot.AQUI, "checkpoints", "ablacao_regime", "teacher_clip1_s0.pt")
PARAMS = dict(iou_min=0.3, k=60, score_min=0.7)
NIVEIS = {
    "limpo": dict(p_drop=0.0, ruido=0.0, fp=0.0),
    "leve": dict(p_drop=0.10, ruido=0.03, fp=0.5),
    "medio": dict(p_drop=0.25, ruido=0.08, fp=1.5),
    "forte": dict(p_drop=0.50, ruido=0.15, fp=3.0),
}
SEQS = mot.TESTE
RES = os.path.join(mot.AQUI, "resultados", "parte5.json")
FIG = os.path.join(mot.AQUI, "figs", "parte5_detector.png")


def sorteios(seq, det):
    """Variaveis aleatorias compartilhadas pelos tres niveis numa sequencia."""
    rng = np.random.default_rng(20261001 + int(seq))
    info = mot.info(seq)
    n = info["n"]
    tentativas = np.repeat(np.arange(1, n + 1), rng.poisson(3.0, n))
    tam = rng.uniform([0.015, 0.04], [0.08, 0.25], size=(len(tentativas), 2))
    wh = tam * [info["W"], info["H"]]
    xy = rng.uniform(size=(len(tentativas), 2)) * (np.array([info["W"], info["H"]]) - wh)
    fp = np.column_stack([tentativas, -np.ones(len(tentativas)), xy, wh,
                          rng.uniform(0.7, 1.0, len(tentativas))])
    return dict(drop=rng.random(len(det)), desloc=rng.normal(size=(len(det), 4)),
                fp=fp, fp_ordem=rng.random(len(fp)))


def degrada(det, nivel, sorteado):
    """Ruido relativo ao tamanho de cada caixa."""
    cfg = NIVEIS[nivel]
    if nivel == "limpo":
        return det.copy()
    d = det[sorteado["drop"] >= cfg["p_drop"]].copy()
    z = sorteado["desloc"][sorteado["drop"] >= cfg["p_drop"]]
    d[:, 2:6] += cfg["ruido"] * z * d[:, [4, 5, 4, 5]]
    d[:, 4:6] = np.maximum(d[:, 4:6], 1.0)
    # afinar o Poisson(3) da fp falsos por quadro em media
    falsos = sorteado["fp"][sorteado["fp_ordem"] < cfg["fp"] / 3.0]
    return np.vstack([d, falsos])


def mede(seq, det, modelo):
    ped, dis = mot.gt(seq)
    d_ap = tira_distratores(det, ped, dis)
    pr = tira_distratores(rastreia(det, PreditorRNN(modelo), **PARAMS), ped, dis)
    m = avalia(ped, pr)
    return dict(mAP50_quadros=map50_quadros(ped, d_ap), AP50_seq=ap50(ped, d_ap),
                IDF1=m["IDF1"], IDSW=m["IDSW"], Frag=m["Frag"],
                ids_pred=m["ids_pred"], n_deteccoes=len(det))


def grafico(out):
    nomes = list(NIVEIS)
    x = np.arange(len(nomes))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for seq in SEQS:
        r = out["resultados"][seq]
        axs[0].plot(x, [r[n]["mAP50_quadros"] for n in nomes], "o-", label=f"MOT17-{seq}")
        axs[1].plot(x, [r[n]["IDF1"] for n in nomes], "o-", label=f"MOT17-{seq}")
    for ax, titulo in zip(axs, ("mAP50 por quadro", "IDF1")):
        ax.set_xticks(x, nomes)
        ax.set_ylim(0, 1)
        ax.set_title(titulo)
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("pontuacao")
    axs[0].legend()
    fig.suptitle("Parte 5: deterioracao da deteccao SDP, mesmo GRU e rastreador")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    modelo = carrega(CKPT)
    out = dict(experimento="qualidade do detector", checkpoint=os.path.relpath(CKPT, mot.AQUI),
               detector="SDP", tracker=PARAMS, seed=20261001, niveis=NIVEIS, resultados={})
    for seq in SEQS:
        det = mot.det(seq, "SDP")
        rand = sorteios(seq, det)
        out["resultados"][seq] = {}
        for nivel in NIVEIS:
            m = mede(seq, degrada(det, nivel, rand), modelo)
            out["resultados"][seq][nivel] = m
            print(seq, nivel, "mAP50", round(m["mAP50_quadros"], 4),
                  "IDF1", round(m["IDF1"], 4), flush=True)
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    with open(RES, "w") as f:
        json.dump(out, f, indent=2)
    grafico(out)


if __name__ == "__main__":
    main()
