"""Leitura do MOT17 (MOT17Labels.zip e MOT17Det.zip extraidos em data/MOT17 ou em MOT17_DIR).

Split por sequencia inteira, so nas 7 que tem gt: treino 04 05 11 13, val 09 (limiar e epoca),
teste 02 (parada, muita oclusao) e 10 (camera andando, de noite).
"""
import configparser
import os
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MOT17_DIR", os.path.join(AQUI, "data", "MOT17"))
TREINO, VAL, TESTE = ["04", "05", "11", "13"], ["09"], ["02", "10"]
TODAS = ["02", "04", "05", "09", "10", "11", "13"]
CAMERA_PARADA = {"02", "04", "09"}
DISTRATORES = [2, 7, 8, 12]   # pessoa em veiculo, pessoa parada, distrator, reflexo


def pasta(seq, fonte="SDP"):
    return os.path.join(ROOT, "train", f"MOT17-{seq}-{fonte}")


def info(seq):
    c = configparser.ConfigParser()
    c.read(os.path.join(pasta(seq), "seqinfo.ini"))
    s = c["Sequence"]
    return dict(fps=int(s["frameRate"]), n=int(s["seqLength"]), W=int(s["imWidth"]), H=int(s["imHeight"]))


def gt(seq):
    """Pedestres (classe 1 com conf 1) e distratores, com colunas frame, id, x, y, w, h, visibilidade."""
    g = np.loadtxt(os.path.join(pasta(seq), "gt", "gt.txt"), delimiter=",")
    cols = [0, 1, 2, 3, 4, 5, 8]
    return g[(g[:, 6] == 1) & (g[:, 7] == 1)][:, cols], g[np.isin(g[:, 7], DISTRATORES)][:, cols]


def det(seq, fonte="SDP"):
    """frame, -1, x, y, w, h, score. Fonte DPM, FRCNN, SDP ou TV (detect.py)."""
    if fonte == "TV":
        caminho = os.path.join(AQUI, "dets", f"MOT17-{seq}-TV.txt")
    else:
        caminho = os.path.join(pasta(seq, fonte), "det", "det.txt")
    return np.loadtxt(caminho, delimiter=",", ndmin=2)[:, :7]


def imagem(seq, frame):
    return os.path.join(ROOT, "train", f"MOT17-{seq}", "img1", f"{frame:06d}.jpg")


def oclusao(seq):
    """Fracao das caixas de pedestre com menos de metade visivel (eixo dos graficos)."""
    return float(np.mean(gt(seq)[0][:, 6] < 0.5))
