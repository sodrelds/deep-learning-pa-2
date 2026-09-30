"""Parte 1: baseline por quadro no MOT17.

Duas fontes de deteccao, nenhuma treinada por nos:
  SDP  deteccao publica que vem no MOT17. Dos tres detectores publicos e o de maior mAP50 por
       quadro nas 7 sequencias, por isso e a fonte padrao do
       resto do PA. Com deteccao boa, o que quebra e a identidade, que e o assunto do PA. O DPM
       ainda tem score em outra escala (de -0.5 a 4.8), o que complica escolher limiar.
  TV   Faster R-CNN do torchvision pre-treinado no COCO, classe person (detect.py)

AP e IDF1 usam o mesmo gt, com as pessoas totalmente escondidas contando. Por isso o AP tem teto de
recall nas sequencias com muita oclusao: na 02, 29% das caixas do gt tem visibilidade 0.
O mAP50 por quadro e a media do AP50 dos quadros com pedestres, sem ponderar pelo numero de pessoas.
O AP50 agregado da sequencia tambem fica salvo, com outro nome, para permitir a comparacao.

Associacao ingenua (tracker.Ingenuo): IoU entre a ultima caixa vista de cada track e as deteccoes
do quadro, Hungarian, limiar fixo, id novo quando nada casa, e a track morre depois de k quadros
sem observacao. O limiar de IoU, o k e o score minimo da deteccao saem da media de IDF1 nas
sequencias de treino e validacao (04 05 09 11 13), nunca nas de teste, e valem pras 7.

As sequencias do grafico vao em ordem de oclusao: fracao das caixas de pedestre com menos de metade
visivel, tirada do campo visibility do gt.

  python parte1.py              # SDP e TV
  python parte1.py --fontes SDP
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mot
from metrics import ap50, map50_quadros, tira_distratores
from tracker import Ingenuo, roda_seq

FIG, RES = os.path.join(mot.AQUI, "figs"), os.path.join(mot.AQUI, "resultados")
GRADE = dict(iou_min=[0.2, 0.3, 0.4, 0.5], k=[1, 5, 15, 30, 60])
SCORES = dict(SDP=[0.4, 0.7, 0.9, 0.99], TV=[0.3, 0.5, 0.7, 0.9])


def ap(seq, fonte):
    ped, dis = mot.gt(seq)
    det = tira_distratores(mot.det(seq, fonte), ped, dis)
    return map50_quadros(ped, det), ap50(ped, det)


def _idf1(tarefa):
    fonte, s, kw = tarefa
    return roda_seq(s, Ingenuo(), fonte, **kw)[0]["IDF1"]


def ajusta(fonte, seqs, pool):
    """Parametros do ingenuo com maior IDF1 medio nas seqs."""
    combos = [dict(iou_min=i, k=k, score_min=sc)
              for i, k, sc in itertools.product(GRADE["iou_min"], GRADE["k"], SCORES[fonte])]
    f1 = np.array(list(pool.map(_idf1, [(fonte, s, kw) for kw in combos for s in seqs])))
    notas = f1.reshape(len(combos), len(seqs)).mean(1)
    kw = combos[int(np.argmax(notas))]
    print(fonte, "melhor no treino+val:", round(float(notas.max()), 3), kw, flush=True)
    return kw


def grafico(res, fontes):
    ordem = sorted(mot.TODAS, key=mot.oclusao)
    x = np.arange(len(ordem))
    rotulo = [f"{s}{' (teste)' if s in mot.TESTE else ''}\n{'parada' if s in mot.CAMERA_PARADA else 'movel'}\n"
              f"ocl. {100 * mot.oclusao(s):.0f}%" for s in ordem]
    fig, (cima, baixo) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for fonte, ls in zip(fontes, ["-", "--"]):
        r = res[fonte]["seqs"]
        cima.plot(x, [r[s]["mAP50_quadros"] for s in ordem], ls, color="C0", marker="o", label=f"mAP50 por quadro ({fonte})")
        cima.plot(x, [r[s]["IDF1"] for s in ordem], ls, color="C1", marker="s", label=f"IDF1 ({fonte})")
        baixo.plot(x, [r[s]["razao_ids"] for s in ordem], ls, color="C2", marker="^", label=f"ids previstos / verdadeiros ({fonte})")
        baixo.plot(x, [r[s]["idsw_por_id"] for s in ordem], ls, color="C3", marker="v", label=f"ID switches por id verdadeiro ({fonte})")
    cima.set_ylim(0, 1)
    cima.legend(fontsize=8, loc="lower left")
    cima.set_title("Descolamento: detectar bem quadro a quadro nao garante manter a identidade (baseline ingenuo)")
    baixo.axhline(1, color="gray", lw=0.8, ls=":")
    baixo.set_ylim(bottom=0)
    baixo.legend(fontsize=8, loc="upper left")
    baixo.set_xticks(x, rotulo, fontsize=8)
    baixo.set_xlabel("sequencias em ordem de oclusao (fracao das caixas com menos de metade visivel)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte1_descolamento.png"), dpi=130)
    plt.close(fig)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--fontes", nargs="+", default=["SDP", "TV"])
    ap_.add_argument("--procs", type=int, default=8)
    args = ap_.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    medidas = {f: {s: ap(s, f) for s in mot.TODAS} for f in ["DPM", "FRCNN", "SDP"]}
    res = dict(map50_quadros_publicas={f: {s: v[0] for s, v in d.items()} for f, d in medidas.items()},
               ap50_seq_publicas={f: {s: v[1] for s, v in d.items()} for f, d in medidas.items()})
    for f, v in res["map50_quadros_publicas"].items():
        print(f"mAP50 por quadro medio {f}: {np.mean(list(v.values())):.3f}", {s: round(a, 3) for s, a in v.items()})

    for fonte in args.fontes:
        with ProcessPoolExecutor(args.procs) as pool:
            kw = ajusta(fonte, mot.TREINO + mot.VAL, pool)
        seqs = {}
        for s in mot.TODAS:
            m, _ = roda_seq(s, Ingenuo(), fonte, **kw)
            seqs[s] = {k: float(v) for k, v in m.items()}
            seqs[s]["mAP50_quadros"], seqs[s]["AP50_seq"] = ap(s, fonte)
            seqs[s]["guloso_IDF1"] = float(roda_seq(s, Ingenuo(), fonte, guloso=True, **kw)[0]["IDF1"])
            print(fonte, s, {k: round(v, 3) for k, v in seqs[s].items()}, flush=True)
        res[fonte] = dict(params=kw, seqs=seqs)
    json.dump(res, open(os.path.join(RES, "parte1.json"), "w"), indent=1)
    grafico(res, args.fontes)


if __name__ == "__main__":
    main()
