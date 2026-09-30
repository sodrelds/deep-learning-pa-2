"""Treina o modelo de movimento da Parte 2 nas trajetorias do gt de treino (04 05 11 13), escolhe
a epoca pela perda na validacao (09) e salva o checkpoint em checkpoints/.

  python train.py                                  # o modelo final: GRU, 64 de estado, T = 16
  python train.py --celula lstm --T 32 --seed 2    # uma configuracao do Eixo 1
"""
import argparse
import os
import time
import torch

import mot
from model import salva, treina


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celula", default="gru", choices=["rnn", "gru", "lstm"])
    ap.add_argument("--oculto", type=int, default=64)
    ap.add_argument("--T", type=int, default=16, help="janela do BPTT truncado")
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", choices=["teacher", "scheduled", "free"], default="teacher")
    ap.add_argument("--clip", type=float, default=1.0, help="norma maxima; 0 desliga o clipping")
    ap.add_argument("--validacao-regime", choices=["teacher", "free"], default="teacher")
    ap.add_argument("--visto", type=int, default=10, help="observacoes iniciais antes de omitir entradas")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--saida")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sufixo = "" if a.regime == "teacher" and a.clip == 1.0 else f"_{a.regime}_clip{a.clip:g}"
    saida = a.saida or os.path.join(mot.AQUI, "checkpoints", f"{a.celula}_T{a.T}_s{a.seed}{sufixo}.pt")
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    t0 = time.time()
    modelo, hist = treina(a.celula, a.oculto, a.T, epocas=a.epocas, seed=a.seed, regime=a.regime,
                         clip=a.clip, validacao_regime=a.validacao_regime, visto=a.visto)
    salva(modelo, saida, T=a.T, seed=a.seed, hist=hist, regime=a.regime, clip=a.clip,
          validacao_regime=a.validacao_regime, visto=a.visto)
    print(f"salvo em {saida} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
