"""Rastreamento por deteccao. O laco e o mesmo pra todos, so muda quem preve onde cada track vai
estar no quadro seguinte:

  Ingenuo       a caixa prevista e a ultima caixa observada (Parte 1)
  Kalman        filtro de Kalman de velocidade constante, so como baseline de comparacao
  PreditorRNN   o modelo recorrente da Parte 2, no model.py

Regra de associacao: IoU entre a caixa prevista de cada track viva e as deteccoes do quadro, com
Hungarian (ou guloso) e limiar fixo. Deteccao com score acima do limiar que nao casa com nenhuma
track vira track nova, com id novo. Track que passa k quadros seguidos sem casar morre. A saida so
tem caixa nos quadros em que a track casou com uma deteccao, e a caixa e a da deteccao.
"""
import numpy as np

import mot
from metrics import avalia, casa, iou, por_quadro, tira_distratores


def centro(c):
    """x, y, w, h -> cx, cy, w, h"""
    return np.array([c[0] + c[2] / 2, c[1] + c[3] / 2, c[2], c[3]])


def canto(z):
    """cx, cy, w, h -> x, y, w, h (tamanho no minimo 1 px, o Kalman pode encolher a caixa)"""
    w, h = max(z[2], 1.0), max(z[3], 1.0)
    return np.array([z[0] - w / 2, z[1] - h / 2, w, h])


class Ingenuo:
    """A caixa prevista pro quadro seguinte e a ultima que a track viu."""

    def novo(self):
        return {}

    def passo(self, estados, obs):
        """obs[i] e a caixa com que a track i casou nesse quadro, ou None. Devolve a caixa
        prevista de cada track pro quadro seguinte."""
        for e, o in zip(estados, obs):
            if o is not None:
                e["caixa"] = o
        return [e["caixa"] for e in estados]


class Kalman:
    """Estado cx, cy, w, h e as quatro velocidades. Os ruidos sao proporcionais a altura da caixa:
    r e o desvio da medida e q o do processo, os dois como fracao da altura."""
    F = np.block([[np.eye(4), np.eye(4)], [np.zeros((4, 4)), np.eye(4)]])
    H = np.eye(4, 8)

    def __init__(self, q=0.01, r=0.03):
        self.q, self.r = q, r

    def novo(self):
        return {}

    def passo(self, estados, obs):
        prev = []
        for e, o in zip(estados, obs):
            if o is not None:
                z = centro(o)
                R = np.eye(4) * (self.r * z[3]) ** 2
                if "x" not in e:
                    # velocidade desconhecida no nascimento, variancia grande nela
                    e["x"] = np.r_[z, np.zeros(4)]
                    e["P"] = np.diag(np.r_[np.diag(R), np.full(4, (0.1 * z[3]) ** 2)])
                else:
                    S = self.H @ e["P"] @ self.H.T + R
                    K = e["P"] @ self.H.T @ np.linalg.inv(S)
                    e["x"] = e["x"] + K @ (z - self.H @ e["x"])
                    e["P"] = (np.eye(8) - K @ self.H) @ e["P"]
            e["x"] = self.F @ e["x"]
            e["P"] = self.F @ e["P"] @ self.F.T + np.eye(8) * (self.q * e["x"][3]) ** 2
            prev.append(canto(e["x"]))
        return prev


def rastreia(dets, preditor, iou_min=0.3, k=30, score_min=0.5, guloso=False):
    """dets: frame, -1, x, y, w, h, score. Devolve frame, id, x, y, w, h."""
    dets = dets[dets[:, 6] >= score_min]
    if len(dets) == 0:
        return np.zeros((0, 6))
    quadros, vazio = por_quadro(dets), np.zeros((0, 7))
    vivas, saida, prox = [], [], 1
    for f in range(1, int(dets[:, 0].max()) + 1):
        d = quadros.get(f, vazio)
        prev = np.array([t["prev"] for t in vivas]).reshape(-1, 4)
        pares = casa(iou(prev, d[:, 2:6]), iou_min, guloso)
        obs = [None] * len(vivas)
        for i, j in pares:
            obs[i] = d[j, 2:6]
            saida.append([f, vivas[i]["id"], *d[j, 2:6]])
        for t, o in zip(vivas, obs):
            t["parada"] = 0 if o is not None else t["parada"] + 1
        fica = [i for i, t in enumerate(vivas) if t["parada"] < k]
        vivas, obs = [vivas[i] for i in fica], [obs[i] for i in fica]
        casadas = {j for _, j in pares}
        for j in range(len(d)):
            if j not in casadas:
                vivas.append(dict(id=prox, parada=0, estado=preditor.novo()))
                obs.append(d[j, 2:6])
                saida.append([f, prox, *d[j, 2:6]])
                prox += 1
        for t, p in zip(vivas, preditor.passo([t["estado"] for t in vivas], obs)):
            t["prev"] = p
    return np.array(saida).reshape(-1, 6)


def roda_seq(seq, preditor, fonte="SDP", **kw):
    """Rastreia uma sequencia do MOT17 e avalia contra os pedestres do gt, depois de tirar as
    caixas que casam com distrator. Devolve as metricas e as caixas rastreadas."""
    ped, dis = mot.gt(seq)
    pr = tira_distratores(rastreia(mot.det(seq, fonte), preditor, **kw), ped, dis)
    return avalia(ped, pr), pr


if __name__ == "__main__":
    # duas pessoas andando lado a lado, sem ruido: o ingenuo e o Kalman seguram os dois ids
    f = np.arange(1, 41)
    a = np.column_stack([f, -np.ones(40), 100 + 3 * f, np.full(40, 50.0), np.full(40, 40.0), np.full(40, 80.0), np.ones(40)])
    b = a.copy()
    b[:, 3] = 300
    for p in [Ingenuo(), Kalman()]:
        out = rastreia(np.vstack([a, b]), p, iou_min=0.3, k=5)
        assert len(np.unique(out[:, 1])) == 2 and len(out) == 80
    # buraco de 10 quadros no meio andando a 3 px/quadro: a caixa parada do ingenuo nao alcanca
    # mais a pessoa quando ela volta (id novo), o Kalman continua andando e acha
    buraco = a[(a[:, 0] <= 15) | (a[:, 0] > 25)]
    assert len(np.unique(rastreia(buraco, Ingenuo(), iou_min=0.3, k=20)[:, 1])) == 2
    assert len(np.unique(rastreia(buraco, Kalman(), iou_min=0.3, k=20)[:, 1])) == 1
    print("ok, rastreador")
