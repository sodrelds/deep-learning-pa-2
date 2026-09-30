"""Parte 2, trilha A: RNN como modelo de movimento.

Um estado recorrente por track. Cada track tem a caixa que o modelo previu pra ela no quadro atual.
Quando a track casa com uma deteccao, a celula recebe essa observacao escrita no referencial da
propria previsao, ou seja, quanto a pessoa apareceu fora de onde o modelo esperava. Quando nao
casa com nada (oclusao, deteccao perdida), a entrada e zero e o estado roda sozinho. A saida e o
deslocamento da previsao do quadro seguinte em relacao a previsao atual. Quem guarda de onde a
pessoa vinha e o estado da recorrencia, porque a entrada so diz o quanto a previsao errou.

O deslocamento e parametrizado do jeito que os slides de deteccao escrevem a caixa em relacao a
ancora (aqui a ancora e a caixa prevista): dx = (cx' - cx)/w, dy = (cy' - cy)/h, dw = log(w'/w),
dh = log(h'/h), vezes ESCALA pra ficar perto de 1.

A observacao entra relativa a previsao por causa da escala. Com a caixa absoluta normalizada pelo
tamanho da imagem, a mudanca entre dois quadros e de uns 0.001, e o GRU precisaria de pesos enormes
pra tirar velocidade disso. A gente testou assim e ele quase nao aprendeu movimento em 60 epocas.
Relativa a previsao, a entrada fica na ordem de 1.

Perda smooth-L1 entre o deslocamento previsto e o deslocamento ate a caixa verdadeira do quadro
seguinte, nas trajetorias do gt das sequencias de treino. As observacoes levam um ruido com o
desvio medido das deteccoes SDP contra o gt (RUIDO), porque no rastreamento a entrada e uma
deteccao, que nunca cai exatamente em cima do gt. O alvo e o gt limpo, entao o modelo tambem aprende
a nao seguir o ruido.

BPTT truncado como nos slides: a janela tem L quadros e o estado atravessa a janela inteira
(k1 = L), mas o gradiente so volta T passos (k2 = T), com detach a cada T. A caixa prevista entra
sem gradiente. Um passo do otimizador por lote, pra todo T ter o mesmo numero de atualizacoes.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mot
from metrics import iou

ESCALA = 10.0
# desvio robusto (mediana do erro absoluto / 0.6745) das deteccoes SDP casadas com o gt nas
# sequencias de treino e val, em dx, dy, dw, dh
RUIDO = np.array([0.06, 0.02, 0.08, 0.04])


class Movimento(nn.Module):
    def __init__(self, celula="gru", oculto=64):
        super().__init__()
        self.celula, self.oculto = celula, oculto
        self.rec = dict(rnn=nn.RNNCell, gru=nn.GRUCell, lstm=nn.LSTMCell)[celula](4, oculto)
        self.saida = nn.Linear(oculto, 4)
        self.dim_estado = 2 * oculto if celula == "lstm" else oculto

    def forward(self, z, s=None):
        """Um passo. z (B, 4) e a observacao relativa a previsao (zero se nao teve observacao);
        s e o estado (no LSTM, h e c concatenados). Devolve o deslocamento da proxima previsao."""
        if self.celula == "lstm":
            h, c = self.rec(z, None if s is None else s.chunk(2, 1))
            s = torch.cat([h, c], 1)
        else:
            h = s = self.rec(z, s)
        return self.saida(h), s


def desloca(c, d):
    """Aplica o deslocamento d (em unidades de ESCALA) na caixa c, as duas xywh."""
    d = d / ESCALA
    cx = c[..., :2] + c[..., 2:] / 2 + d[..., :2] * c[..., 2:]
    wh = c[..., 2:] * torch.exp(d[..., 2:])
    return torch.cat([cx - wh / 2, wh], -1)


def deslocamento(c, c2):
    """O d tal que desloca(c, d) = c2."""
    return ESCALA * torch.cat([(c2[..., :2] + c2[..., 2:] / 2 - c[..., :2] - c[..., 2:] / 2) / c[..., 2:],
                               torch.log(c2[..., 2:] / c[..., 2:])], -1)


def poe_ruido(c, gen):
    r = torch.randn(c.shape, generator=gen) * torch.tensor(RUIDO, dtype=c.dtype)
    cx = c[..., :2] + c[..., 2:] / 2 + r[..., :2] * c[..., 2:]
    wh = c[..., 2:] * torch.exp(r[..., 2:])
    return torch.cat([cx - wh / 2, wh], -1)


def trajetorias(seqs):
    """Caixas xywh de cada pedestre do gt. O gt do MOT17 nao tem buraco dentro de um id, a pessoa
    continua anotada quando fica escondida."""
    out = []
    for s in seqs:
        ped = mot.gt(s)[0]
        for i in np.unique(ped[:, 1]):
            g = ped[ped[:, 1] == i]
            out.append(g[np.argsort(g[:, 0]), 2:6])
    return out


def janelas(trajs, L, rng):
    """Corta janelas de L+1 caixas, com inicio sorteado e mais ou menos uma janela a cada L/4
    quadros de trajetoria. Trajetoria curta vira uma janela so, com o fim mascarado."""
    X, M = [], []
    for c in trajs:
        n = len(c)
        if n < 3:
            continue
        inicios = [0] if n <= L + 1 else rng.integers(0, n - L, max(1, 4 * n // L))
        for i in inicios:
            w = c[i:i + L + 1]
            X.append(np.pad(w, ((0, L + 1 - len(w)), (0, 0)), mode="edge"))
            M.append(np.arange(L) < len(w) - 1)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(M), dtype=torch.float32)


def perda_janelas(modelo, X, M, T, gen, treino=True, regime="teacher", prob_livre=0.0, visto=10):
    """BPTT truncado. Apos `visto` passos, escolhe observacao ou previsao livre.

    teacher: recebe observacao ruidosa em todos os passos; scheduled: a observacao de cada
    passo e omitida com probabilidade crescente; free: so recebe as primeiras `visto`
    observacoes. Na omissao, z=0 e a caixa prevista segue a saida anterior da rede.
    A caixa prevista sempre e realimentada, como na inferencia.
    """
    if regime not in ("teacher", "scheduled", "free"):
        raise ValueError(f"regime desconhecido: {regime}")
    obs = poe_ruido(X, gen)
    p = obs[:, 0]   # no nascimento a previsao e a propria deteccao
    s, total, L = None, 0.0, X.shape[1] - 1
    for t0 in range(0, L, T):
        perda = 0.0
        for t in range(t0, min(t0 + T, L)):
            if regime == "teacher" or t < visto:
                usar = torch.ones(len(X), dtype=torch.bool)
            elif regime == "free":
                usar = torch.zeros(len(X), dtype=torch.bool)
            else:
                usar = torch.rand(len(X), generator=gen) >= prob_livre
            z = deslocamento(p, obs[:, t]) * usar[:, None]
            d, s = modelo(z, s)
            perda = perda + (F.smooth_l1_loss(d, deslocamento(p, X[:, t + 1]), reduction="none").sum(1) * M[:, t]).sum()
            p = desloca(p, d.detach())
        if treino:
            (perda / M.sum()).backward()
        s = s.detach()
        total += float(perda.detach())
    return total / float(M.sum())


def treina(celula="gru", oculto=64, T=16, L=64, epocas=60, lote=128, lr=1e-3, clip=1.0, seed=0,
           trajs_treino=None, trajs_val=None, log=print, regime="teacher", validacao_regime="teacher",
           visto=10):
    """Treina e devolve (modelo com a melhor perda na validacao, historico)."""
    torch.manual_seed(seed)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    trajs_treino = trajs_treino if trajs_treino is not None else trajetorias(mot.TREINO)
    trajs_val = trajs_val if trajs_val is not None else trajetorias(mot.VAL)
    val = janelas(trajs_val, L, np.random.default_rng(123))
    modelo = Movimento(celula, oculto)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    melhor, estado, hist = float("inf"), None, []
    for ep in range(epocas):
        X, M = janelas(trajs_treino, L, rng)
        perm = torch.randperm(len(X), generator=gen)
        modelo.train()
        tr, normas = [], []
        prob_livre = ep / max(epocas - 1, 1) if regime == "scheduled" else 0.0
        for b in range(0, len(X), lote):
            i = perm[b:b + lote]
            opt.zero_grad()
            tr.append(perda_janelas(modelo, X[i], M[i], T, gen, regime=regime,
                                    prob_livre=prob_livre, visto=visto))
            if clip:
                norma = nn.utils.clip_grad_norm_(modelo.parameters(), clip)
            else:
                norma = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in modelo.parameters()
                                                               if p.grad is not None]))
            normas.append(float(norma))
            opt.step()
        modelo.eval()
        with torch.no_grad():
            v = perda_janelas(modelo, *val, T, torch.Generator().manual_seed(123), treino=False,
                              regime=validacao_regime, visto=visto)
        hist.append((float(np.mean(tr)), v, float(np.mean(normas))))
        if v < melhor:
            melhor, estado = v, {k: x.clone() for k, x in modelo.state_dict().items()}
        log(f"epoca {ep + 1:3d}  treino {hist[-1][0]:.4f}  val {v:.4f}  norma_grad {hist[-1][2]:.3f}")
    modelo.load_state_dict(estado)
    return modelo.eval(), hist


def salva(modelo, caminho, **extra):
    torch.save(dict(celula=modelo.celula, oculto=modelo.oculto, pesos=modelo.state_dict(), **extra), caminho)


def carrega(caminho):
    ck = torch.load(caminho, weights_only=False)
    m = Movimento(ck["celula"], ck["oculto"])
    m.load_state_dict(ck["pesos"])
    return m.eval()


class PreditorRNN:
    """Liga o modelo no tracker.rastreia. O estado de cada track guarda o estado da recorrencia e a
    caixa prevista pro quadro atual."""

    def __init__(self, modelo):
        self.m = modelo.eval()

    def novo(self):
        return {}

    @torch.no_grad()
    def passo(self, estados, obs):
        if not estados:
            return []
        p = torch.tensor(np.array([e.get("prev", o) for e, o in zip(estados, obs)]), dtype=torch.float32)
        z = torch.zeros(len(estados), 4)
        vistos = [i for i, o in enumerate(obs) if o is not None]
        if vistos:
            z[vistos] = deslocamento(p[vistos], torch.tensor(np.array([obs[i] for i in vistos]), dtype=torch.float32))
        zero = torch.zeros(self.m.dim_estado)
        d, s = self.m(z, torch.stack([e.get("s", zero) for e in estados]))
        prev = desloca(p, d)
        for e, si, pi in zip(estados, s, prev):
            e["s"], e["prev"] = si, pi.numpy()
        return list(prev.numpy())


def previsao_livre(preditor, jan, visto=10, seed=0):
    """jan (n, visto + H, 4): pedacos de trajetoria do gt. A track ve os primeiros `visto` quadros
    com o ruido das deteccoes e depois anda sozinha. Devolve (n, H) com o IoU entre a caixa prevista
    e a verdadeira; h = 1 e a previsao normal do quadro seguinte a ultima observacao."""
    obs = poe_ruido(torch.tensor(jan[:, :visto], dtype=torch.float32), torch.Generator().manual_seed(seed)).numpy()
    est = [preditor.novo() for _ in jan]
    for t in range(visto):
        prev = preditor.passo(est, list(obs[:, t]))
    out = []
    for h in range(jan.shape[1] - visto):
        if h:
            prev = preditor.passo(est, [None] * len(jan))
        out.append([iou(a, b)[0, 0] for a, b in zip(prev, jan[:, visto + h])])
    return np.array(out).T


if __name__ == "__main__":
    # confere as contas da parametrizacao e que o modelo aprende a andar: treina rapido nas
    # trajetorias das elipses sinteticas e, vendo 10 quadros com ruido e depois andando sozinho
    # por 10, tem que acompanhar a elipse melhor que repetir a ultima caixa
    from synth import gera
    from tracker import Ingenuo
    c = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    c2 = torch.tensor([[13.0, 18.0, 33.0, 57.0]])
    assert torch.allclose(desloca(c, deslocamento(c, c2)), c2, atol=1e-4)

    # Nos extremos do schedule, as mascaras devem reproduzir teacher e free.
    x = torch.stack([torch.stack([c[0] + torch.tensor([float(t), 0, 0, 0])
                                  for t in range(9)]) for _ in range(2)])
    mascara = torch.ones(2, 8)
    pequena = Movimento()
    def perda(regime, prob_livre=0.0):
        return perda_janelas(pequena, x, mascara, 4, torch.Generator().manual_seed(42),
                             treino=False, regime=regime, prob_livre=prob_livre, visto=3)
    assert np.isclose(perda("teacher"), perda("scheduled", 0.0))
    assert np.isclose(perda("free"), perda("scheduled", 1.0))

    trajs = []
    for seed in range(60):
        _, g = gera(n_obj=8, vel=2.0, T=60, seed=seed)
        trajs += [g[g[:, 1] == i, 2:6] for i in np.unique(g[:, 1])]
    modelo, _ = treina(epocas=15, L=32, trajs_treino=trajs[:400], trajs_val=trajs[400:], log=lambda *a: None)
    jan = np.array([t[:20] for t in trajs[400:]])
    rnn = previsao_livre(PreditorRNN(modelo), jan).mean(0)
    ing = previsao_livre(Ingenuo(), jan).mean(0)
    print("IoU h=1, 5, 10  rnn", rnn[[0, 4, 9]].round(2), " ultima caixa", ing[[0, 4, 9]].round(2))
    assert rnn[0] > ing[0] and rnn[9] > ing[9]
    print("ok, modelo")
