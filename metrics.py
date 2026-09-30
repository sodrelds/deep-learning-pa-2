"""Metricas de rastreamento feitas na mao (motmetrics e TrackEval sao proibidos no PA).

Tudo aqui usa arrays com uma caixa por linha: frame, id, x, y, w, h, ... com (x, y) no canto de
cima a esquerda, igual ao gt.txt do MOT17. Deteccao sem id usa -1 na coluna do id e o score na
coluna 6, igual ao det.txt.

IDF1: atribuicao global um-pra-um entre ids verdadeiros e previstos na sequencia inteira. Pra cada
par (gt i, previsto j) conta em quantos quadros as caixas se sobrepoem com IoU >= 0.5 e o
Hungarian escolhe os pares que maximizam a soma. Essa soma e o IDTP, e
IDF1 = 2 IDTP / (caixas do gt + caixas previstas).

ID switch e fragmentacao saem de um casamento quadro a quadro. Um par que casou no quadro anterior
e ainda passa do limiar continua casado, o resto vai pro Hungarian. Sem essa regra, dois gts lado
a lado trocam de par por causa de ruido e aparecem switches que nao aconteceram. Switch e o gt
casar com um id previsto diferente do ultimo com que tinha casado, mesmo depois de um buraco.
Fragmentacao e o gt que estava rastreado ficar sem casar e depois voltar a casar.

NMS e o matching tambem ficam aqui porque usam o mesmo IoU.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    """IoU entre todas as caixas de a (n, 4) e de b (m, 4), no formato x, y, w, h."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    iw = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    uniao = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.divide(inter, uniao, out=np.zeros_like(inter), where=uniao > 0)


def casa(m, limiar, guloso=False):
    """Pares (i, j) um-pra-um de uma matriz de IoU, so os que passam do limiar.

    O Hungarian maximiza a soma dos IoU dos pares validos. O guloso e o do PA1: vai do maior IoU
    pro menor enquanto os dois lados estiverem livres.
    """
    if m.size == 0:
        return []
    if guloso:
        pares, usou_i, usou_j = [], set(), set()
        for i, j in sorted(zip(*np.nonzero(m >= limiar)), key=lambda p: -m[p]):
            if i not in usou_i and j not in usou_j:
                pares.append((i, j))
                usou_i.add(i)
                usou_j.add(j)
        return pares
    # par abaixo do limiar entra com custo 0, nao ajuda na soma e e filtrado depois
    r, c = linear_sum_assignment(np.where(m >= limiar, -m, 0.0))
    return [(i, j) for i, j in zip(r, c) if m[i, j] >= limiar]


def nms(caixas, scores, limiar=0.5):
    """NMS dos slides: pega a caixa de maior score, joga fora quem tem IoU >= limiar com ela e
    repete com o que sobrou. Devolve os indices das caixas que ficam."""
    ordem = np.argsort(-np.asarray(scores), kind="stable")
    fica = []
    while len(ordem):
        i = ordem[0]
        fica.append(i)
        ordem = ordem[1:][iou(caixas[i], caixas[ordem[1:]])[0] < limiar]
    return np.array(fica, int)


def por_quadro(a):
    """dict frame -> linhas daquele frame."""
    a = a[np.argsort(a[:, 0], kind="stable")]
    f, ini = np.unique(a[:, 0].astype(int), return_index=True)
    return dict(zip(f.tolist(), np.split(a, ini[1:])))


def tira_distratores(pr, ped, dis, limiar=0.5):
    """Tira as caixas previstas que casam com um distrator do gt (pessoa parada, pessoa em veiculo,
    reflexo...). Detectar essas coisas nao conta nem como acerto nem como erro. O casamento e feito
    junto com os pedestres, entao uma caixa em cima de um pedestre de verdade nao some so porque
    tem um distrator do lado."""
    if len(dis) == 0 or len(pr) == 0:
        return pr
    todos = np.vstack([np.column_stack([ped[:, :6], np.zeros(len(ped))]),
                       np.column_stack([dis[:, :6], np.ones(len(dis))])])
    gq = por_quadro(todos)
    fora = []
    for f, p in por_quadro(np.column_stack([pr, np.arange(len(pr))])).items():
        g = gq.get(f)
        if g is None:
            continue
        fora += [int(p[j, -1]) for i, j in casa(iou(g[:, 2:6], p[:, 2:6]), limiar) if g[i, 6] == 1]
    return np.delete(pr, fora, axis=0)


def ap50(gt, det, limiar=0.5):
    """AP da classe pessoa com IoU >= 0.5, do jeito dos slides: ordena as deteccoes por score,
    monta a curva precisao x recall, usa a precisao interpolada (a maior precisao em qualquer
    recall acima) e faz a media nos niveis de recall. Junta todos os quadros da sequencia, porque
    AP por quadro fica instavel em quadro com duas pessoas."""
    if len(det) == 0:
        return 0.0
    gq = por_quadro(gt)
    usado = {f: np.zeros(len(g), bool) for f, g in gq.items()}
    det = det[np.argsort(-det[:, 6], kind="stable")]
    tp = np.zeros(len(det))
    for k, d in enumerate(det):
        g = gq.get(int(d[0]))
        if g is None:
            continue
        m = iou(d[2:6], g[:, 2:6])[0]
        m[usado[int(d[0])]] = 0
        j = m.argmax()
        if m[j] >= limiar:
            tp[k] = 1
            usado[int(d[0])][j] = True
    acertos = np.cumsum(tp)
    rec = acertos / len(gt)
    prec = np.maximum.accumulate((acertos / np.arange(1, len(det) + 1))[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def idf1(gt, pr, limiar=0.5):
    """Devolve IDF1 e IDTP."""
    if len(pr) == 0:
        return 0.0, 0.0
    gids, pids = np.unique(gt[:, 1]), np.unique(pr[:, 1])
    sobre = np.zeros((len(gids), len(pids)))
    pq = por_quadro(pr)
    for f, g in por_quadro(gt).items():
        p = pq.get(f)
        if p is None:
            continue
        i, j = np.nonzero(iou(g[:, 2:6], p[:, 2:6]) >= limiar)
        np.add.at(sobre, (np.searchsorted(gids, g[i, 1]), np.searchsorted(pids, p[j, 1])), 1)
    r, c = linear_sum_assignment(-sobre)
    idtp = sobre[r, c].sum()
    return 2 * idtp / (len(gt) + len(pr)), idtp


def casamentos(gt, pr, limiar=0.5):
    """Casamento quadro a quadro com a regra de continuidade. Devolve {(frame, id gt): id previsto}."""
    pq = por_quadro(pr) if len(pr) else {}
    vazio = np.zeros((0, pr.shape[1] if pr.ndim == 2 else 6))
    anterior, cas = {}, {}
    for f, g in por_quadro(gt).items():
        p = pq.get(f, vazio)
        m = iou(g[:, 2:6], p[:, 2:6])
        col = {pid: j for j, pid in enumerate(p[:, 1])}
        pares = {}
        for i, gid in enumerate(g[:, 1]):
            j = col.get(anterior.get(gid))
            if j is not None and m[i, j] >= limiar:
                pares[i] = j
        li = [i for i in range(len(g)) if i not in pares]
        lj = [j for j in range(len(p)) if j not in pares.values()]
        for a, b in casa(m[np.ix_(li, lj)], limiar):
            pares[li[a]] = lj[b]
        anterior = {g[i, 1]: p[j, 1] for i, j in pares.items()}
        cas.update({(f, gid): pid for gid, pid in anterior.items()})
    return cas


def trocas(gt, pr, limiar=0.5):
    """ID switches, fragmentacoes e a lista de switches (frame, id gt, id antigo, id novo)."""
    cas = casamentos(gt, pr, limiar)
    ultimo, rastreado = {}, {}
    idsw, frag, eventos = 0, 0, []
    for f, gid in gt[np.argsort(gt[:, 0], kind="stable"), :2]:
        pid = cas.get((int(f), gid))
        if pid is not None:
            if gid in ultimo and ultimo[gid] != pid:
                idsw += 1
                eventos.append((int(f), gid, ultimo[gid], pid))
            if rastreado.get(gid) is False:
                frag += 1
            ultimo[gid] = pid
            rastreado[gid] = True
        elif gid in rastreado:
            rastreado[gid] = False
    return idsw, frag, eventos


def sobrevivencia(gt, pr, vis_min=0.3, janela=5, limiar=0.5):
    """Pra cada vez que um objeto do gt fica escondido (visibilidade < vis_min) e depois aparece de
    novo, confere se o rastreador deu o mesmo id antes e depois do buraco. Procura o casamento ate
    `janela` quadros antes e depois, porque o detector pode demorar a pegar a pessoa de volta.
    Buraco sem casamento dos dois lados fica de fora. Devolve uma lista de (duracao, sobreviveu)."""
    cas = casamentos(gt, pr, limiar)
    out = []
    for gid in np.unique(gt[:, 1]):
        g = gt[gt[:, 1] == gid]
        g = g[np.argsort(g[:, 0])]
        d = np.diff(np.r_[0, (g[:, 6] < vis_min).astype(int), 0])
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            antes = [cas[k] for k in ((int(f), gid) for f in g[max(a - janela, 0):a, 0]) if k in cas]
            depois = [cas[k] for k in ((int(f), gid) for f in g[b:b + janela, 0]) if k in cas]
            if antes and depois:
                out.append((int(b - a), bool(antes[-1] == depois[0])))
    return out


def avalia(gt, pr, limiar=0.5):
    f1, idtp = idf1(gt, pr, limiar)
    sw, fr, _ = trocas(gt, pr, limiar)
    ng = len(np.unique(gt[:, 1]))
    npr = len(np.unique(pr[:, 1])) if len(pr) else 0
    return dict(IDF1=f1, IDP=idtp / max(len(pr), 1), IDR=idtp / len(gt), IDSW=sw, Frag=fr,
                ids_gt=ng, ids_pred=npr, razao_ids=npr / ng, idsw_por_id=sw / ng,
                erro_contagem=(npr - ng) / ng)


if __name__ == "__main__":
    def trilha(i, x0, quadros):
        f = np.asarray(quadros, float)
        return np.column_stack([f, np.full(len(f), i), x0 + f, np.full(len(f), 100.0),
                                np.full(len(f), 50.0), np.full(len(f), 100.0)])

    # dois pedestres de 100 quadros, longe um do outro
    t = np.arange(1, 101)
    gt = np.vstack([trilha(1, 10, t), trilha(2, 500, t)])

    assert abs(iou([0, 0, 10, 10], [5, 0, 10, 10])[0, 0] - 1 / 3) < 1e-9

    # (a) previsao igual ao gt
    r = avalia(gt, gt.copy())
    assert r["IDF1"] == 1 and r["IDSW"] == 0 and r["Frag"] == 0 and r["erro_contagem"] == 0

    # (b) os dois ids trocam a partir do quadro 51. Cada gt troca de id uma vez, entao sao 2
    # switches. No IDF1 o melhor que da e casar cada gt com metade dele: IDTP = 100 das 200 caixas
    troca = gt.copy()
    depois = troca[:, 0] >= 51
    troca[depois, 1] = 3 - troca[depois, 1]
    r = avalia(gt, troca)
    assert r["IDSW"] == 2 and r["Frag"] == 0 and abs(r["IDF1"] - 0.5) < 1e-9, r

    # (c) a track do pedestre 1 parte em duas no meio e o 2 fica intacto: 1 switch so, mas a
    # metade que ficou com o id novo e toda IDFN e IDFP. IDTP = 50 + 100, IDF1 = 300/400
    parte = gt.copy()
    parte[(parte[:, 1] == 1) & (parte[:, 0] >= 51), 1] = 3
    r = avalia(gt, parte)
    assert r["IDSW"] == 1 and r["Frag"] == 0 and abs(r["IDF1"] - 0.75) < 1e-9, r
    assert r["ids_pred"] == 3 and abs(r["erro_contagem"] - 0.5) < 1e-9

    # (c) com um buraco de 10 quadros entre os pedacos, a troca vira tambem uma fragmentacao
    buraco = np.vstack([trilha(1, 10, range(1, 51)), trilha(3, 10, range(61, 101)), trilha(2, 500, t)])
    r = avalia(gt, buraco)
    assert r["IDSW"] == 1 and r["Frag"] == 1 and abs(r["IDF1"] - 300 / 390) < 1e-9, r

    # regra de continuidade: o previsto 7 segue o gt 1 e o 8 segue o gt 2, mas no quadro 50 as
    # caixas previstas trocam de lugar. O Hungarian puro trocaria os pares (soma 2.0 contra 1.33)
    # e contaria 4 switches. Como os pares antigos ainda passam de 0.5, ficam
    a, b = trilha(1, 10, t), trilha(2, 20, t)
    pa, pb = a.copy(), b.copy()
    pa[:, 1], pb[:, 1] = 7, 8
    pa[49, 2], pb[49, 2] = b[49, 2], a[49, 2]
    assert trocas(np.vstack([a, b]), np.vstack([pa, pb]))[0] == 0

    # AP: tudo detectado da 1; so metade do gt da 0.5; um FP com o maior score quase nao muda
    det = np.column_stack([gt[:, 0], np.full(len(gt), -1), gt[:, 2:6], np.ones(len(gt))])
    assert abs(ap50(gt, det) - 1) < 1e-9
    assert abs(ap50(gt, det[det[:, 2] < 400]) - 0.5) < 1e-9
    fp = np.array([[1, -1, 900, 900, 50, 100, 2.0]])
    assert abs(ap50(gt, np.vstack([fp, det])) - 200 / 201) < 1e-9

    # NMS: a segunda caixa tem IoU 0.81 com a primeira e cai, a terceira esta longe e fica
    cx = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [50, 50, 10, 10]], float)
    assert nms(cx, [0.9, 0.8, 0.7]).tolist() == [0, 2]

    # sobrevivencia: o pedestre 1 fica escondido nos quadros 41 a 50. Com o mesmo id dos dois
    # lados do buraco sobreviveu; com id novo depois, nao
    vis = np.ones(len(gt))
    vis[(gt[:, 1] == 1) & (gt[:, 0] >= 41) & (gt[:, 0] <= 50)] = 0
    gv = np.column_stack([gt, vis])
    visto = gt[vis == 1]
    assert sobrevivencia(gv, visto) == [(10, True)]
    novo = visto.copy()
    novo[(novo[:, 1] == 1) & (novo[:, 0] > 50), 1] = 9
    assert sobrevivencia(gv, novo) == [(10, False)]

    # distrator: a caixa em cima da pessoa parada sai, a do pedestre fica
    dis = np.array([[1, 99, 900, 100, 50, 100, 1.0]])
    pr = np.array([[1, 5, 11, 100, 50, 100], [1, 6, 900, 100, 50, 100]], float)
    assert tira_distratores(pr, gt, dis)[:, 1].tolist() == [5]

    print("ok, metricas batem nos casos feitos na mao")
