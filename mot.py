"""Leitura do MOT17.

Os labels vem do MOT17Labels.zip (gt e det dos tres detectores publicos) e as imagens do
MOT17Det.zip, que tem cada sequencia uma vez so (o MOT17.zip repete as imagens pros tres
detectores e pesa 5.5 GB). Os dois vao extraidos na mesma pasta, que por padrao e data/MOT17 e
pode ser trocada pela variavel MOT17_DIR.

So as 7 sequencias de treino do MOTChallenge tem gt, entao o split sai delas, sempre por
sequencia inteira:

  treino  04 05 11 13   parada e densa (04), camera andando a 14 fps (05), shopping com a camera
                        andando (11), onibus a 25 fps (13)
  val     09            parada e com pouca gente, usada pra escolher limiar e epoca
  teste   02 10         o modelo nunca ve: uma parada com muita oclusao (02) e uma com a camera
                        andando de noite (10)
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
    """Deteccoes com colunas frame, -1, x, y, w, h, score. A fonte e DPM, FRCNN, SDP ou TV (a do
    torchvision, gerada pelo detect.py)."""
    if fonte == "TV":
        caminho = os.path.join(AQUI, "dets", f"MOT17-{seq}-TV.txt")
    else:
        caminho = os.path.join(pasta(seq, fonte), "det", "det.txt")
    return np.loadtxt(caminho, delimiter=",", ndmin=2)[:, :7]


def imagem(seq, frame):
    return os.path.join(ROOT, "train", f"MOT17-{seq}", "img1", f"{frame:06d}.jpg")


def oclusao(seq):
    """Fracao das caixas de pedestre com menos de metade visivel. E o eixo de dificuldade dos graficos."""
    return float(np.mean(gt(seq)[0][:, 6] < 0.5))
