"""Parte 4: horizonte de memoria (gradiente e empirico), tres falhas e correcao antes/depois.
Checkpoint: teacher com clipping, seed 0, o melhor na validacao 09 da Parte 3.

  python parte4.py --sem-galeria    # so as medidas
  python parte4.py                  # com as tiras (quadros via fetch_frames.py)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_rgb
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import mot
from metrics import avalia, casamentos, iou, tira_distratores
from model import PreditorRNN, carrega, desloca, deslocamento, poe_ruido
from tracker import rastreia


CKPT = os.path.join(mot.AQUI, "checkpoints", "ablacao_regime", "teacher_clip1_s0.pt")
BASE = dict(iou_min=0.3, k=60, score_min=0.7)
# na 09 o GT10 volta depois de 71 quadros com IoU 0.227 contra a deteccao certa, abaixo de 0.3
CORRECAO = dict(iou_retorno=0.2, idade_retorno=10)
SEQS = mot.VAL + mot.TESTE
RES = os.path.join(mot.AQUI, "resultados", "parte4.json")
FIG = os.path.join(mot.AQUI, "figs")
GALERIA = [("09", 21, 106, 173), ("02", 9, 182, 222), ("10", 14, 500, 534)]
FAIXAS = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 10000)]


def rastreia_seq(seq, modelo, **extra):
    ped, dis = mot.gt(seq)
    bruto, trace = rastreia(mot.det(seq, "SDP"), PreditorRNN(modelo), com_trace=True,
                           **BASE, **extra)
    pr = tira_distratores(bruto, ped, dis)
    return ped, pr, trace, avalia(ped, pr)


def eventos_oclusao(seq, ped, pr, trace):
    """Cada buraco completo no GT e o tempo que o ID anterior permaneceu disponivel e correto."""
    cas = casamentos(ped, pr)
    dono = {(int(f), int(pid)): int(gid) for (f, gid), pid in cas.items()}
    ativos = {(int(r[0]), int(r[1])): r for r in trace}
    det = mot.det(seq, "SDP")
    eventos = []
    for gid in np.unique(ped[:, 1]).astype(int):
        g = ped[ped[:, 1] == gid]
        g = g[np.argsort(g[:, 0])]
        d = np.diff(np.r_[0, (g[:, 6] < 0.3).astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if a == 0 or b == len(g):
                continue
            inicio, retorno, dur = int(g[a, 0]), int(g[b, 0]), int(b - a)
            antes = [int(cas[(int(f), gid)]) for f in g[max(0, a - 5):a, 0]
                     if (int(f), gid) in cas]
            depois = [int(cas[(int(f), gid)]) for f in g[b:min(len(g), b + 5), 0]
                      if (int(f), gid) in cas]
            pid = antes[-1] if antes else None
            novo = depois[0] if depois else None
            morte = next((f for f in range(inicio, retorno + 1) if pid is not None and
                          (f, pid) not in ativos), None) if pid is not None else None
            contaminacao = next((f for f in range(inicio, retorno + 1) if pid is not None and
                                 dono.get((f, pid)) not in (None, gid)), None) if pid is not None else None
            primeira_falha = min([x for x in (morte, contaminacao) if x is not None], default=retorno)
            caixa_prev = ativos.get((retorno, pid)) if pid is not None else None
            pred_iou = float(iou(caixa_prev[2:6], g[b, 2:6])[0, 0]) if caixa_prev is not None else None
            ds = det[(det[:, 0] == retorno) & (det[:, 6] >= BASE["score_min"])]
            sobre_det = iou(ds[:, 2:6], g[b, 2:6]).reshape(-1) if len(ds) else np.array([])
            melhor_det = int(np.argmax(sobre_det)) if len(ds) else None
            det_iou = float(sobre_det[melhor_det]) if len(ds) else 0.0
            pred_det_iou = (float(iou(caixa_prev[2:6], ds[melhor_det, 2:6])[0, 0])
                            if caixa_prev is not None and len(ds) else None)
            igual = bool(pid is not None and novo == pid)
            if pid is None:
                causa = "sem_id_antes"
            elif igual:
                causa = "mantida"
            elif morte is not None and morte <= retorno:
                causa = "morte"
            elif contaminacao is not None and contaminacao <= retorno:
                causa = "contaminacao"
            elif det_iou < 0.5:
                causa = "detector"
            elif pred_iou is not None and pred_iou < BASE["iou_min"]:
                causa = "fora_gate"
            else:
                causa = "associacao"
            eventos.append(dict(seq=seq, gt_id=gid, inicio=inicio, retorno=retorno, duracao=dur,
                                id_antes=pid, id_depois=novo, manteve_id=igual,
                                morte=morte, contaminacao=contaminacao,
                                tempo_sem_falha=max(0, min(dur, primeira_falha - inicio)) if pid is not None else None,
                                iou_prev_retorno=pred_iou, iou_det_retorno=det_iou,
                                iou_prev_det_retorno=pred_det_iou,
                                causa=causa))
    return eventos


def gradiente_memoria(modelo, K=64, aquecimento=16, n_janelas=96):
    """Norma de dL_t/dh_(t-k) em janelas reais do gt, com o grafo inteiro e com o detach a cada
    16 passos do treino."""
    total = K + aquecimento + 1
    candidatos = []
    for seq in SEQS:
        ped = mot.gt(seq)[0]
        for gid in np.unique(ped[:, 1]):
            g = ped[ped[:, 1] == gid]
            g = g[np.argsort(g[:, 0])]
            for a in range(0, len(g) - total + 1, 8):
                if g[a, 6] >= 0.3:
                    candidatos.append((seq, g[a:a + total, 2:7]))
    rng = np.random.default_rng(4)
    inds = rng.choice(len(candidatos), min(n_janelas, len(candidatos)), replace=False)
    janelas = [candidatos[i][1] for i in inds]
    x = torch.tensor(np.array(janelas)[:, :, :4], dtype=torch.float32)
    vis = torch.tensor(np.array(janelas)[:, :, 4], dtype=torch.float32)
    obs = poe_ruido(x, torch.Generator().manual_seed(4))
    def curva(truncar):
        p, s, estados = obs[:, 0], None, []
        for t in range(total - 1):
            if truncar and t > 0 and t % 16 == 0:
                s = s.detach()
            z = deslocamento(p, obs[:, t]) * (vis[:, t] >= 0.3)[:, None]
            d, s = modelo(z, s)
            estados.append(s)
            if t == total - 2:
                perda = F.smooth_l1_loss(d, deslocamento(p, x[:, t + 1]), reduction="none").sum(1).mean()
            p = desloca(p, d.detach())
        grad = torch.autograd.grad(perda, estados[-K:], allow_unused=truncar)
        normas = np.array([float(g.norm(dim=1).mean()) if g is not None else 0.0
                           for g in grad[::-1]])
        return normas, float(perda.detach())

    normas, loss = curva(False)
    normas_treino, _ = curva(True)
    relativo = normas / normas[0]
    relativo_treino = normas_treino / normas_treino[0]
    abaixo = np.flatnonzero(relativo <= 0.05)
    return dict(k=list(range(K)), norma=normas.tolist(), relativa=relativo.tolist(),
                norma_treino=normas_treino.tolist(), relativa_treino=relativo_treino.tolist(),
                limiar_20x=int(abaixo[0]) if len(abaixo) else None,
                n_janelas=len(janelas), loss=loss,
                seqs=SEQS, BPTT=16)


def figura_gradiente(r):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(r["k"], r["relativa"], "o-", ms=2, label="GRU, GT MOT17 + ruido de deteccao")
    ax.semilogy(r["k"][:16], r["relativa_treino"][:16], "s--", ms=3,
                label="gradiente usado no treino (zero apos T)")
    ax.axvline(r["BPTT"], color="C3", ls="--", label=f"truncamento BPTT T={r['BPTT']}")
    ax.axhline(0.05, color="gray", ls=":", label="queda de 20 vezes")
    ax.set(xlabel="k quadros entre o estado e a perda", ylabel="norma relativa de dL_t/dh_(t-k)",
           title=f"Horizonte analitico, {r['n_janelas']} janelas de MOT17-09/02/10")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte4_gradiente.png"), dpi=150)
    plt.close(fig)


def resumo_empirico(eventos):
    todos = [e for e in eventos if e["id_antes"] is not None]
    out = {}
    for a, b in FAIXAS:
        grupo = [e for e in todos if a <= e["duracao"] <= b]
        out[f"{a}-{b}"] = dict(n=len(grupo), fracao_mesmo_id=float(np.mean([e["manteve_id"] for e in grupo]))
                                if grupo else None,
                                tempo_sem_falha_medio=float(np.mean([e["tempo_sem_falha"] for e in grupo]))
                                if grupo else None)
    return dict(n_buracos_completos=len(eventos), n_com_id_antes=len(todos), faixas=out,
                duracoes_gt=[e["duracao"] for e in eventos],
                tempos_sem_falha=[e["tempo_sem_falha"] for e in todos])


def figura_empirica(r, corrigido=None):
    fig, (ax, ay) = plt.subplots(2, 1, figsize=(8, 7))
    h = np.arange(0, max(r["duracoes_gt"]) + 1)
    ax.step(h, [np.mean(np.array(r["duracoes_gt"]) >= t) for t in h], where="post",
            label=f"duracao da oclusao no GT (n={r['n_buracos_completos']})")
    ax.step(h, [np.mean(np.array(r["tempos_sem_falha"]) >= t) for t in h], where="post",
            label=f"tempo com ID intacto, censurado na volta (n={r['n_com_id_antes']})")
    ax.axvline(60, color="C3", ls="--", label="vida maxima da track k=60")
    ax.set(xlabel="h (quadros)", ylabel="fracao de eventos com duracao >= h",
           title="Distribuicoes de duracao e horizonte empirico")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    chaves = list(r["faixas"])
    taxas = [r["faixas"][k]["fracao_mesmo_id"] or 0 for k in chaves]
    ns = [r["faixas"][k]["n"] for k in chaves]
    x = np.arange(len(chaves))
    if corrigido is None:
        ay.bar(x, taxas, color="C0")
    else:
        taxas_corr = [corrigido["faixas"][k]["fracao_mesmo_id"] or 0 for k in chaves]
        ay.bar(x - 0.2, taxas, 0.4, color="C0", label="IoU fixo 0.3")
        ay.bar(x + 0.2, taxas_corr, 0.4, color="C1", label="IoU 0.2 apos 10 quadros")
        ay.legend(fontsize=8)
    ay.set_xticks(x, [f"{'121+' if k.startswith('121-') else k}\n(n={n})"
                                       for k, n in zip(chaves, ns)])
    ay.set(xlabel="duracao da oclusao no GT", ylabel="fracao com o mesmo ID na volta",
           title="Identidade preservada por duracao")
    ay.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte4_empirico.png"), dpi=150)
    plt.close(fig)


def figura_correcao(base, corrigido):
    seqs = SEQS
    x = np.arange(len(seqs))
    fig, (ax, ay) = plt.subplots(1, 2, figsize=(9, 3.7))
    for eixo, metrica, titulo in [(ax, "IDF1", "IDF1"), (ay, "IDSW", "ID switches")]:
        eixo.bar(x - 0.18, [base[s][metrica] for s in seqs], 0.35, label="IoU fixo 0.3")
        eixo.bar(x + 0.18, [corrigido[s][metrica] for s in seqs], 0.35,
                 label="IoU 0.2 apos 10 quadros")
        eixo.set_xticks(x, [f"MOT17-{s}" for s in seqs])
        eixo.set_title(titulo)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parte4_correcao.png"), dpi=150)
    plt.close(fig)


def quadros_galeria(inicio, retorno):
    return sorted(set([inicio - 2, inicio - 1, inicio, (inicio + retorno) // 2,
                       retorno - 1, retorno, retorno + 2]))


def cor_id(i):
    return hsv_to_rgb(((int(i) * 0.61803398875) % 1, 0.8, 0.95))


def desenha_caixa(ax, caixa, origem, cor, estilo, texto, lw=1.5):
    x0, y0 = origem
    x, y, w, h = caixa
    ax.add_patch(Rectangle((x - x0, y - y0), w, h, fill=False, ec=cor, lw=lw, ls=estilo))
    largura = max(ax.get_xlim())
    altura = max(ax.get_ylim())
    tx = float(np.clip(x - x0, 2, largura - 50))
    ty = float(np.clip(y - y0 - 2, 10, altura - 8))
    ax.text(tx, ty, texto, fontsize=7, color="black" if np.mean(to_rgb(cor)) > 0.55 else "white",
            clip_on=True, bbox=dict(facecolor=cor, alpha=0.85, pad=1, edgecolor="none"))


def figura_falha(seq, gid, inicio, retorno, evento, ped, pr, trace):
    frames = quadros_galeria(inicio, retorno)
    g = ped[ped[:, 1] == gid]
    trace_por_chave = {(int(r[0]), int(r[1])): r for r in trace}
    fig, axes = plt.subplots(2, len(frames), figsize=(3.1 * len(frames), 5.8))
    for j, f in enumerate(frames):
        caminho = mot.imagem(seq, f)
        imagem = Image.open(caminho).convert("RGB")
        alvo = g[g[:, 0] == f]
        if not len(alvo):
            continue
        alvo = alvo[0]
        W, H = imagem.size
        cx, cy = alvo[2] + alvo[4] / 2, alvo[3] + alvo[5] / 2
        w, h = min(500, W), min(360, H)
        x0 = int(np.clip(cx - w / 2, 0, W - w))
        y0 = int(np.clip(cy - h / 2, 0, H - h))
        imagem = imagem.crop((x0, y0, x0 + w, y0 + h))
        for linha, ax in enumerate(axes[:, j]):
            ax.imshow(imagem, alpha=1.0 if linha == 0 else 0.55)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.axis("off")
            if linha == 0:
                for caixa in ped[ped[:, 0] == f]:
                    if caixa[6] < 0.3 and int(caixa[1]) != gid:
                        continue
                    if caixa[2] + caixa[4] < x0 or caixa[2] > x0 + w or caixa[3] + caixa[5] < y0 or caixa[3] > y0 + h:
                        continue
                    alvo_gt = int(caixa[1]) == gid
                    desenha_caixa(ax, caixa[2:6], (x0, y0), cor_id(caixa[1]),
                                 "--" if caixa[6] < 0.3 else "-", f"GT {int(caixa[1])}",
                                 2.5 if alvo_gt else 1.1)
                for caixa in pr[pr[:, 0] == f]:
                    if caixa[2] + caixa[4] < x0 or caixa[2] > x0 + w or caixa[3] + caixa[5] < y0 or caixa[3] > y0 + h:
                        continue
                    desenha_caixa(ax, caixa[2:6], (x0, y0), cor_id(caixa[1]), ":",
                                 f"P {int(caixa[1])}", 1.1)
                ax.set_title(f"quadro {f} | vis {alvo[6]:.2f}", fontsize=9)
            else:
                desenha_caixa(ax, alvo[2:6], (x0, y0), "lime", "-", f"GT {gid}", 2)
                pid = evento["id_antes"]
                estado = trace_por_chave.get((f, pid)) if pid is not None else None
                if estado is not None:
                    ciou = float(iou(estado[2:6], alvo[2:6])[0, 0])
                    desenha_caixa(ax, estado[2:6], (x0, y0), "yellow", "--",
                                 f"prev P{pid} IoU {ciou:.2f}", 2)
                    ax.set_title(f"caixa prevista | sem ver {int(estado[6])} q", fontsize=9)
                else:
                    ax.set_title(f"sem estado para P{pid}", fontsize=9)
    fig.suptitle(f"MOT17-{seq}, GT {gid}: {evento['duracao']} quadros oculto, "
                 f"P{evento['id_antes']} -> P{evento['id_depois']} ({evento['causa']})\n"
                 "Topo: GT solido, track pontilhada. Base: GT verde, previsao GRU amarela. Cores por ID.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    caminho = os.path.join(FIG, f"parte4_falha_{seq}_gt{gid}.png")
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-galeria", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    torch.set_num_threads(2)
    modelo = carrega(CKPT)
    base, corr, detalhes, eventos, eventos_corr = {}, {}, {}, [], []
    for seq in SEQS:
        ped, pr, trace, met = rastreia_seq(seq, modelo)
        ped_corr, pr_corr, trace_corr, met_corr = rastreia_seq(seq, modelo, **CORRECAO)
        detalhes[seq] = (ped, pr, trace)
        base[seq], corr[seq] = met, met_corr
        eventos.extend(eventos_oclusao(seq, ped, pr, trace))
        eventos_corr.extend(eventos_oclusao(seq, ped_corr, pr_corr, trace_corr))
        print(seq, "base", round(met["IDF1"], 4), "corrigido", round(met_corr["IDF1"], 4), flush=True)
    # so a 09 escolhe o portao
    grade = {"0.3 (fixo)": base["09"]["IDF1"], "0.2": corr["09"]["IDF1"]}
    for gate in (0.25, 0.15):
        _, _, _, met = rastreia_seq("09", modelo, iou_retorno=gate, idade_retorno=10)
        grade[str(gate)] = met["IDF1"]
    assert grade["0.2"] > grade["0.3 (fixo)"]
    print("gate na validacao:", grade, flush=True)
    grad = gradiente_memoria(modelo)
    emp, emp_corr = resumo_empirico(eventos), resumo_empirico(eventos_corr)
    escolhidos, escolhidos_corr = [], []
    for seq, gid, inicio, retorno in GALERIA:
        e = next(e for e in eventos if (e["seq"], e["gt_id"], e["inicio"], e["retorno"]) ==
                 (seq, gid, inicio, retorno))
        escolhidos.append(e)
        e_corr = next(e for e in eventos_corr if (e["seq"], e["gt_id"], e["inicio"], e["retorno"]) ==
                      (seq, gid, inicio, retorno))
        escolhidos_corr.append(e_corr)
    diagnostico_gate = next(e for e in eventos if e["seq"] == "09" and e["gt_id"] == 10
                            and e["inicio"] == 309 and e["retorno"] == 380)
    assert 0.2 <= diagnostico_gate["iou_prev_det_retorno"] < 0.25
    out = dict(checkpoint=os.path.relpath(CKPT, mot.AQUI), base_params=BASE,
               correcao=CORRECAO, ajuste_gate_val=grade, base=base, corrigido=corr,
               gradiente=grad, empirico=emp, empirico_corrigido=emp_corr,
               diagnostico_gate=diagnostico_gate,
               eventos=eventos, galeria=escolhidos,
               galeria_corrigida=escolhidos_corr)
    def numero_numpy(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(f"tipo nao serializavel: {type(x)}")
    with open(RES, "w") as f:
        json.dump(out, f, indent=2, default=numero_numpy, allow_nan=False)
    figura_gradiente(grad)
    figura_empirica(emp, emp_corr)
    figura_correcao(base, corr)
    print("gradiente cai 20x em k=", grad["limiar_20x"], flush=True)
    print("buracos completos", emp["n_buracos_completos"], "com ID antes", emp["n_com_id_antes"], flush=True)
    if args.sem_galeria:
        return
    faltam = [(seq, f) for seq, _, inicio, retorno in GALERIA
              for f in quadros_galeria(inicio, retorno) if not os.path.exists(mot.imagem(seq, f))]
    if faltam:
        partes = [f"{seq}:{','.join(str(f) for s, f in faltam if s == seq)}" for seq in sorted(set(s for s, _ in faltam))]
        raise FileNotFoundError("Imagens ausentes. Rode: python fetch_frames.py " + " ".join(partes))
    for seq, gid, inicio, retorno in GALERIA:
        e = next(e for e in escolhidos if (e["seq"], e["gt_id"]) == (seq, gid))
        caminho = figura_falha(seq, gid, inicio, retorno, e, *detalhes[seq])
        print("galeria:", caminho, flush=True)


if __name__ == "__main__":
    main()
