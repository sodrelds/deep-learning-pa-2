"""Parte 2: o GRU contra o baseline ingenuo nas sequencias de teste (02 e 10), com as mesmas
deteccoes SDP e as mesmas metricas da Parte 1. O Kalman de velocidade constante entra so como
referencia.

A fonte de deteccao e o score minimo ficam congelados na Parte 1 (resultados/parte1.json). Entre
os metodos so muda o que acontece entre os quadros, a caixa prevista que entra no IoU da
associacao. O limiar de IoU e o k de cada metodo (e o q do Kalman) saem da media de IDF1 no
treino+val, como na Parte 1.

Mais duas medidas pra entender de onde vem a diferenca:
  horizonte  no gt de val e teste, sem detector: a track ve a pessoa por 10 quadros (com o ruido
             das deteccoes) e depois passa h quadros sem observacao, com a previsao andando
             sozinha, como numa oclusao. Mede o IoU entre a caixa prevista e a verdadeira em cada h.
  buracos    nas sequencias de teste, a fracao das vezes em que a pessoa some (visibilidade < 0.3)
             e volta com o mesmo id, separada pela duracao do buraco.

  python evaluate.py
  python evaluate.py --ckpt checkpoints/lstm_T32_s0.pt
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mot
from metrics import sobrevivencia
from model import PreditorRNN, carrega, previsao_livre
from tracker import Ingenuo, Kalman, roda_seq

FIG, RES = os.path.join(mot.AQUI, "figs"), os.path.join(mot.AQUI, "resultados")
GRADE = dict(iou_min=[0.2, 0.3, 0.4, 0.5], k=[1, 5, 15, 30, 60])
Q_KALMAN = [0.003, 0.01, 0.03]
METODOS = ["ingenuo", "kalman", "gru"]
FAIXAS = [(1, 5), (6, 15), (16, 30), (31, 10000)]
CORES = dict(ingenuo="C7", kalman="C0", gru="C3")


def preditor(metodo, ckpt, q):
    if metodo == "ingenuo":
        return Ingenuo()
    if metodo == "kalman":
        return Kalman(q=q)
    return PreditorRNN(carrega(ckpt))


def _idf1(tarefa):
    torch.set_num_threads(1)
    metodo, seq, ckpt, q, kw = tarefa
    return roda_seq(seq, preditor(metodo, ckpt, q), "SDP", **kw)[0]["IDF1"]


def ajusta(metodo, ckpt, score_min, pool):
    combos = [(q, dict(iou_min=i, k=k, score_min=score_min))
              for q in (Q_KALMAN if metodo == "kalman" else [None])
              for i, k in itertools.product(GRADE["iou_min"], GRADE["k"])]
    seqs = mot.TREINO + mot.VAL
    f1 = np.array(list(pool.map(_idf1, [(metodo, s, ckpt, q, kw) for q, kw in combos for s in seqs])))
    notas = f1.reshape(len(combos), len(seqs)).mean(1)
    q, kw = combos[int(np.argmax(notas))]
    print(metodo, "melhor no treino+val:", round(float(notas.max()), 3), kw, "q" if q else "", q or "", flush=True)
    return q, kw


def horizonte(metodo, ckpt, q, visto=10, H=30):
    """IoU medio entre caixa prevista e verdadeira h = 1..H quadros depois da ultima observacao,
    em pedacos de trajetoria do gt de val e teste (model.previsao_livre). As observacoes levam o
    mesmo ruido das deteccoes, igual pros tres metodos."""
    jan = []
    for s in mot.VAL + mot.TESTE:
        ped = mot.gt(s)[0]
        for i in np.unique(ped[:, 1]):
            g = ped[ped[:, 1] == i]
            g = g[np.argsort(g[:, 0]), 2:6]
            jan += [g[a:a + visto + H] for a in range(0, len(g) - visto - H + 1, 15)]
    return previsao_livre(preditor(metodo, ckpt, q), np.array(jan), visto).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(mot.AQUI, "checkpoints", "gru_T16_s0.pt"))
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    score_min = json.load(open(os.path.join(RES, "parte1.json")))["SDP"]["params"]["score_min"]

    res = dict(ckpt=os.path.relpath(args.ckpt, mot.AQUI), score_min=score_min)
    with ProcessPoolExecutor(args.procs) as pool:
        for m in METODOS:
            q, kw = ajusta(m, args.ckpt, score_min, pool)
            teste, buracos = {}, []
            for s in mot.TESTE:
                met, pr = roda_seq(s, preditor(m, args.ckpt, q), "SDP", **kw)
                teste[s] = {k: float(v) for k, v in met.items()}
                buracos += sobrevivencia(mot.gt(s)[0], pr)
            d = np.array(buracos)
            res[m] = dict(params=kw, q=q, teste=teste,
                          buracos={f"{a}-{b}": [float(d[(d[:, 0] >= a) & (d[:, 0] <= b), 1].mean()),
                                                int(((d[:, 0] >= a) & (d[:, 0] <= b)).sum())] for a, b in FAIXAS},
                          horizonte=horizonte(m, args.ckpt, q).tolist())
            print(m, {s: {k: round(v, 3) for k, v in t.items()} for s, t in teste.items()}, flush=True)
            print(m, "buracos (fracao que sobrevive, quantos):", res[m]["buracos"], flush=True)
    json.dump(res, open(os.path.join(RES, "parte2.json"), "w"), indent=1)
    graficos(res)


def graficos(res):
    fig, eixos = plt.subplots(1, 4, figsize=(17, 3.8))
    larg = 0.27
    for ax, (chave, nome) in zip(eixos, [("IDF1", "IDF1"), ("idsw_por_id", "ID switches por id verdadeiro"),
                                         ("razao_ids", "ids previstos / verdadeiros")]):
        for j, m in enumerate(METODOS):
            ax.bar(np.arange(len(mot.TESTE)) + (j - 1) * larg, [res[m]["teste"][s][chave] for s in mot.TESTE],
                   larg, color=CORES[m], label=m)
        ax.set_xticks(range(len(mot.TESTE)), [f"MOT17-{s}" for s in mot.TESTE])
        ax.set_title(nome, fontsize=10)
    eixos[2].axhline(1, color="gray", lw=0.8, ls=":")
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="upper left", ncol=3, fontsize=9)
    ax = eixos[3]
    for j, m in enumerate(METODOS):
        v = [res[m]["buracos"][f"{a}-{b}"][0] for a, b in FAIXAS]
        ax.bar(np.arange(len(FAIXAS)) + (j - 1) * larg, v, larg, color=CORES[m], label=m)
    n = [res["gru"]["buracos"][f"{a}-{b}"][1] for a, b in FAIXAS]
    ax.set_xticks(range(len(FAIXAS)), [f"{a}-{b}\n({k} buracos)" if b < 10000 else f"{a}+\n({k} buracos)"
                                       for (a, b), k in zip(FAIXAS, n)], fontsize=8)
    ax.set_title("fracao dos buracos em que o id sobrevive\n(02 e 10, duracao em quadros)", fontsize=10)
    ax.set_ylim(0, 1)
    fig.suptitle("Parte 2 nas sequencias de teste, mesmas deteccoes SDP", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "parte2_comparacao.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in METODOS:
        h = res[m]["horizonte"]
        ax.plot(range(1, len(h) + 1), h, color=CORES[m], marker=".", label=f"{m} (limiar de IoU {res[m]['params']['iou_min']})")
    ax.set_xlabel("quadros depois da ultima observacao")
    ax.set_ylabel("IoU entre caixa prevista e verdadeira")
    ax.set_title("Previsao andando sozinha depois de ver a pessoa por 10 quadros (gt de 09, 02 e 10)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte2_horizonte.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
