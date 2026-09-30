"""Parte 3, eixo 2: observacoes sempre presentes, scheduled sampling e previsao livre.

Cada configuracao usa as mesmas trajetorias, arquitetura GRU, janela de BPTT, ruido e regras
de associacao. O val 09 escolhe a melhor epoca por perda em previsao livre; as sequencias 02/10
so entram na avaliacao final. Sao tres seeds por configuracao e clipping ligado/desligado.

  python ablacao_regime.py
  python ablacao_regime.py --epocas 3 --seeds 0 --force  # verificacao rapida
"""
import argparse
import json
import os
from itertools import product

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mot
from model import PreditorRNN, salva, treina, trajetorias
from tracker import roda_seq


PARAMS = dict(iou_min=0.3, k=60, score_min=0.7)  # fixados antes da ablação, no resultado da Parte 2
REGIMES = ("teacher", "scheduled", "free")
CLIPS = (1.0, 0.0)
PASTA_CKPT = os.path.join(mot.AQUI, "checkpoints", "ablacao_regime")
CAMINHO_RES = os.path.join(mot.AQUI, "resultados", "parte3_regime.json")


def resumo(linhas):
    out = {}
    for regime, clip in product(REGIMES, CLIPS):
        rs = [r for r in linhas.values() if r["regime"] == regime and r["clip"] == clip]
        if not rs:
            continue
        chave = f"{regime}_clip{clip:g}"
        out[chave] = {"n_seeds": len(rs)}
        for nome, vals in {
            "val_IDF1": [r["metricas"]["09"]["IDF1"] for r in rs],
            "teste_IDF1": [np.mean([r["metricas"][s]["IDF1"] for s in mot.TESTE]) for r in rs],
            "teste_02_IDF1": [r["metricas"]["02"]["IDF1"] for r in rs],
            "teste_10_IDF1": [r["metricas"]["10"]["IDF1"] for r in rs],
            "teste_IDSW": [sum(r["metricas"][s]["IDSW"] for s in mot.TESTE) for r in rs],
            "val_perda_livre": [r["melhor_val_perda_livre"] for r in rs],
            "norma_grad_pico": [r["norma_grad_pico"] for r in rs],
        }.items():
            out[chave][nome] = {"media": float(np.mean(vals)), "desvio": float(np.std(vals, ddof=1))
                                if len(vals) > 1 else None}
    return out


def grafico(res):
    if not res["resumo"]:
        return
    nomes = list(res["resumo"])
    rotulos = [n.replace("_clip", "\nclip ") for n in nomes]
    x = np.arange(len(nomes))
    fig, (ax, ay) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for eixo, campo, titulo in [(ax, "teste_IDF1", "IDF1 medio nas sequencias 02 e 10"),
                                 (ay, "norma_grad_pico", "Maior norma media por epoca, antes do clipping")]:
        y = [res["resumo"][n][campo]["media"] for n in nomes]
        erro = [res["resumo"][n][campo]["desvio"] or 0 for n in nomes]
        eixo.bar(x, y, color=["C0" if "clip1" in n else "C3" for n in nomes], yerr=erro, capsize=4)
        eixo.set_title(titulo)
        eixo.set_ylabel("IDF1" if campo == "teste_IDF1" else "norma L2")
    ay.set_xticks(x, rotulos)
    fig.tight_layout()
    os.makedirs(os.path.join(mot.AQUI, "figs"), exist_ok=True)
    fig.savefig(os.path.join(mot.AQUI, "figs", "parte3_regime.png"), dpi=150)
    plt.close(fig)


def salvar(res):
    res["resumo"] = resumo(res["runs"])
    os.makedirs(os.path.dirname(CAMINHO_RES), exist_ok=True)
    with open(CAMINHO_RES, "w") as f:
        json.dump(res, f, indent=2)
    grafico(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(PASTA_CKPT, exist_ok=True)
    res = {"descricao": "Eixo 2: observacoes sempre presentes, omissao crescente, previsao livre",
           "epocas": a.epocas, "T": 16, "L": 64, "visto": 10, "params_tracker": PARAMS,
           "split": dict(treino=mot.TREINO, val=mot.VAL, teste=mot.TESTE), "runs": {}, "resumo": {}}
    if os.path.exists(CAMINHO_RES) and not a.force:
        anterior = json.load(open(CAMINHO_RES))
        if anterior.get("epocas") == a.epocas and anterior.get("T") == 16:
            res["runs"] = anterior["runs"]
    treino, val = trajetorias(mot.TREINO), trajetorias(mot.VAL)
    for regime, clip, seed in product(REGIMES, CLIPS, a.seeds):
        chave = f"{regime}_clip{clip:g}_s{seed}"
        if chave in res["runs"]:
            print("ja pronto:", chave, flush=True)
            continue
        print("treinando:", chave, flush=True)
        modelo, hist = treina(T=16, L=64, epocas=a.epocas, clip=clip, seed=seed,
                             trajs_treino=treino, trajs_val=val, regime=regime,
                             validacao_regime="free", visto=10,
                             log=lambda s: print(chave, s, flush=True) if any(
                                 f"epoca {n:3d} " in s for n in (1, 10, 20, 30, 40, 50, a.epocas)) else None)
        ckpt = os.path.join(PASTA_CKPT, f"{chave}.pt")
        salva(modelo, ckpt, T=16, L=64, seed=seed, regime=regime, clip=clip,
              visto=10, validacao_regime="free", hist=hist)
        metricas = {}
        for s in mot.VAL + mot.TESTE:
            m, _ = roda_seq(s, PreditorRNN(modelo), "SDP", **PARAMS)
            metricas[s] = {k: float(v) for k, v in m.items()}
        res["runs"][chave] = dict(regime=regime, clip=clip, seed=seed,
                                   checkpoint=os.path.relpath(ckpt, mot.AQUI),
                                   melhor_epoca=int(np.argmin([h[1] for h in hist]) + 1),
                                   melhor_val_perda_livre=float(min(h[1] for h in hist)),
                                   norma_grad_pico=float(max(h[2] for h in hist)),
                                   historico=hist, metricas=metricas)
        salvar(res)
        print("concluido:", chave, "IDF1 teste", [round(metricas[s]["IDF1"], 3) for s in mot.TESTE], flush=True)
    salvar(res)
    print(json.dumps(res["resumo"], indent=2), flush=True)


if __name__ == "__main__":
    main()
