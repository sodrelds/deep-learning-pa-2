"""Parte 0: gerador de video sintetico e simulador de detector.

O gerador desenha elipses em movimento num quadro 128x128 em ordem de profundidade, de tras pra
frente, entao quem esta atras some de verdade quando passa atras de outra. A visibilidade de cada
elipse e a fracao dos pixels dela que sobrou no desenho final, o mesmo significado do campo
visibility do MOT17, e o gt continua existindo durante a oclusao, igual la.

Pra controlar a duracao da oclusao tem um oclusor, uma elipse grande e parada na frente de todo
mundo, no meio do quadro. Ela e bem mais alta que o quadro (vira uma faixa vertical) e a largura e
calculada pra uma elipse de tamanho medio, cruzando na horizontal na velocidade tipica, ficar
inteira escondida por `oclusao` quadros. Os objetos andam mais na horizontal (ate 30 graus) e
quicam nas bordas. O buraco na deteccao sai um pouco maior que `oclusao`, porque a visibilidade
cai abaixo de 0.3 antes da elipse sumir inteira e as elipses mais lentas demoram mais pra cruzar.
"""
import numpy as np

LADO = 128
SEMI_EIXO = (3.0, 8.0)   # semi-eixos sorteados nesse intervalo, media 5.5


def gera(n_obj=8, vel=1.5, oclusao=0, T=45, ruido=0.05, contraste=(0.3, 0.8), seed=0):
    """Devolve o video (T, 128, 128) em [0, 1] e o gt com frame, id, x, y, w, h, visibilidade.

    n_obj      numero de elipses
    vel        velocidade tipica em pixels por quadro (cada elipse sorteia de 0.5 a 1.5 vezes isso)
    oclusao    duracao da oclusao em quadros; 0 tira o oclusor
    ruido      desvio do ruido gaussiano somado no video
    contraste  diferenca de intensidade entre elipse e fundo, sorteada nesse intervalo
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:LADO, :LADO]
    ax, ay = rng.uniform(*SEMI_EIXO, n_obj), rng.uniform(*SEMI_EIXO, n_obj)
    pos = np.column_stack([rng.uniform(ax, LADO - 1 - ax), rng.uniform(ay, LADO - 1 - ay)])
    ang = rng.uniform(-np.pi / 6, np.pi / 6, n_obj) + np.pi * rng.integers(0, 2, n_obj)
    v = vel * rng.uniform(0.5, 1.5, n_obj)[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
    ordem = rng.permutation(n_obj)   # desenha nessa ordem, o ultimo fica na frente
    fundo = rng.uniform(0.2, 0.5)
    tom = np.clip(fundo + rng.choice([-1, 1], n_obj) * rng.uniform(*contraste, n_obj), 0, 1)
    rx = oclusao * vel / 2 + np.mean(SEMI_EIXO)
    oclusor = ((xx - (LADO - 1) / 2) / rx) ** 2 + ((yy - (LADO - 1) / 2) / 400) ** 2 <= 1

    video = np.empty((T, LADO, LADO), np.float32)
    gt = []
    for t in range(T):
        img = np.full((LADO, LADO), fundo)
        dono = np.full((LADO, LADO), -1)
        area = np.zeros(n_obj)
        for i in ordem:
            m = ((xx - pos[i, 0]) / ax[i]) ** 2 + ((yy - pos[i, 1]) / ay[i]) ** 2 <= 1
            img[m], dono[m], area[i] = tom[i], i, m.sum()
        if oclusao > 0:
            img[oclusor], dono[oclusor] = min(fundo + 0.35, 1), -2
        for i in range(n_obj):
            vis = (dono == i).sum() / area[i]
            gt.append([t + 1, i + 1, pos[i, 0] - ax[i], pos[i, 1] - ay[i], 2 * ax[i], 2 * ay[i], vis])
        video[t] = np.clip(img + rng.normal(0, ruido, img.shape), 0, 1)
        pos += v
        for d, a in ((0, ax), (1, ay)):
            bateu = (pos[:, d] < a) | (pos[:, d] > LADO - 1 - a)
            v[bateu, d] *= -1
            pos[:, d] = np.clip(pos[:, d], a, LADO - 1 - a)
    return video, np.array(gt)


def detector(gt, p_drop=0.0, ruido=0.0, fp=0.0, vis_min=0.3, seed=0):
    """Estraga as caixas verdadeiras de proposito. Devolve frame, -1, x, y, w, h, score.

    vis_min  elipse com menos visibilidade que isso nao e detectada, e assim que a oclusao vira
             buraco na deteccao
    p_drop   fracao das caixas visiveis descartadas ao acaso
    ruido    desvio do ruido nas coordenadas, relativo ao tamanho da caixa
    fp       media de falsos positivos por quadro, em lugar aleatorio
    """
    rng = np.random.default_rng(seed)
    g = gt[gt[:, 6] >= vis_min]
    g = g[rng.random(len(g)) >= p_drop]
    cx = g[:, 2:6] + rng.normal(0, ruido, (len(g), 4)) * g[:, [4, 5, 4, 5]]
    cx[:, 2:] = np.maximum(cx[:, 2:], 1)
    dets = np.column_stack([g[:, 0], -np.ones(len(g)), cx, rng.uniform(0.5, 1, len(g))])
    quadro = np.repeat(np.arange(1, int(gt[:, 0].max()) + 1), rng.poisson(fp, int(gt[:, 0].max())))
    wh = 2 * rng.uniform(*SEMI_EIXO, (len(quadro), 2))
    xy = rng.uniform(0, 1, (len(quadro), 2)) * (LADO - wh)
    falsos = np.column_stack([quadro, -np.ones(len(quadro)), xy, wh, rng.uniform(0.3, 0.8, len(quadro))])
    return np.vstack([dets, falsos])


if __name__ == "__main__":
    # sem oclusor tem elipse inteira visivel, e com oclusor alguma tem que sumir de vez (vis 0)
    _, g = gera(n_obj=5, vel=1.0, oclusao=0, T=30, seed=1)
    assert g.shape == (150, 7) and g[:, 6].max() == 1
    _, g = gera(n_obj=10, vel=1.5, oclusao=15, T=60, seed=1)
    assert (g[:, 6] == 0).any()
    d = detector(g, p_drop=0.5, seed=0)
    assert len(d) < (g[:, 6] >= 0.3).sum()
    print("ok, gerador e detector")
