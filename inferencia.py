"""Inferencia em uma sequencia de imagens: caixas com IDs persistentes e contagem unica.

Aceita uma pasta com img1/ (ou a propria pasta das imagens). Se houver det/det.txt,
usa essas deteccoes; caso contrario usa o Faster R-CNN do detect.py, sem treino.
O rastreador usa o checkpoint e os parametros finais documentados na Parte 4.
"""
import colorsys
from pathlib import Path

import cv2
import numpy as np
import torch

from model import PreditorRNN, carrega
from tracker import rastreia


RAIZ = Path(__file__).resolve().parent
CKPT = RAIZ / "checkpoints" / "ablacao_regime" / "teacher_clip1_s0.pt"
PARAMS = dict(iou_min=0.3, k=60, score_min=0.7)
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp"}


def imagens_e_frames(caminho):
    seq = Path(caminho).expanduser().resolve()
    if not seq.is_dir():
        raise FileNotFoundError(f"pasta da sequencia ausente: {seq}")
    pasta = seq / "img1" if (seq / "img1").is_dir() else seq
    imagens = sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES)
    if not imagens:
        raise ValueError(f"nenhuma imagem em {pasta}")
    if all(p.stem.isdecimal() for p in imagens):
        frames = [int(p.stem) for p in imagens]
        if len(set(frames)) != len(frames) or min(frames) < 1:
            raise ValueError("nomes numericos dos quadros devem ser unicos e positivos")
    else:
        frames = list(range(1, len(imagens) + 1))
    return seq, imagens, frames


def obter_deteccoes(seq, imagens, frames, arquivo=None):
    caminho = Path(arquivo).expanduser().resolve() if arquivo else seq / "det" / "det.txt"
    if caminho.is_file():
        if caminho.stat().st_size == 0:
            return np.zeros((0, 7)), f"arquivo {caminho}"
        linhas = np.loadtxt(caminho, delimiter=",", ndmin=2)
        if linhas.shape[1] < 7:
            raise ValueError("det.txt precisa ter ao menos 7 colunas MOT: frame,id,x,y,w,h,score")
        det = linhas[:, :7].copy()
        det[:, 1] = -1
        det = det[np.isin(det[:, 0].astype(int), frames)]
        return det, f"arquivo {caminho}"
    if arquivo is not None:
        raise FileNotFoundError(f"arquivo de deteccoes ausente: {caminho}")
    from detect import detector, detecta
    modelo = detector()
    linhas = []
    for i, (img, f) in enumerate(zip(imagens, frames), 1):
        ds = detecta(modelo, img)
        if len(ds):
            linhas.extend([f, -1, *d] for d in ds)
        if i % 50 == 0:
            print(f"detector: {i}/{len(imagens)} quadros", flush=True)
    return np.asarray(linhas, float).reshape(-1, 7), "Faster R-CNN torchvision"


def cor(pid):
    rgb = colorsys.hsv_to_rgb((int(pid) * 0.61803398875) % 1, 0.82, 0.98)
    return tuple(int(255 * x) for x in rgb[::-1])


def gerar_video(sequencia, saida="inferencia.mp4", deteccoes=None, fps=25,
                largura_max=1280, checkpoint=CKPT):
    """Devolve (caminho do MP4, numero de IDs unicos), sem retreinar.

    `sequencia` aponta para pasta com img1/ ou diretamente para imagens. `deteccoes`
    opcional aponta para det.txt no formato MOT. Sem ele, procura seq/det/det.txt;
    na ausencia desse arquivo executa o detector torchvision quadro a quadro.
    """
    seq, imagens, frames = imagens_e_frames(sequencia)
    det, fonte = obter_deteccoes(seq, imagens, frames, deteccoes)
    torch.set_num_threads(2)
    pr = rastreia(det, PreditorRNN(carrega(checkpoint)), **PARAMS)
    por_frame = {}
    for linha in pr:
        por_frame.setdefault(int(linha[0]), []).append(linha)
    ids = set(int(i) for i in pr[:, 1]) if len(pr) else set()
    vistos = set()
    primeiro = cv2.imread(str(imagens[0]))
    if primeiro is None:
        raise ValueError(f"imagem ilegivel: {imagens[0]}")
    h, w = primeiro.shape[:2]
    escala = min(1.0, largura_max / w)
    destino = (round(w * escala), round(h * escala))
    saida = Path(saida).expanduser().resolve()
    saida.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, destino)
    if not writer.isOpened():
        raise RuntimeError(f"nao foi possivel abrir MP4 para escrita: {saida}")
    try:
        for img, f in zip(imagens, frames):
            quadro = cv2.imread(str(img))
            if quadro is None:
                raise ValueError(f"imagem ilegivel: {img}")
            if quadro.shape[:2] != (h, w):
                raise ValueError(f"tamanho de quadro inconsistente: {img}")
            if escala != 1.0:
                quadro = cv2.resize(quadro, destino, interpolation=cv2.INTER_AREA)
            for linha in por_frame.get(f, []):
                pid = int(linha[1])
                vistos.add(pid)
                x, y, bw, bh = linha[2:6] * escala
                x1, y1 = round(x), round(y)
                x2, y2 = round(x + bw), round(y + bh)
                cv2.rectangle(quadro, (x1, y1), (x2, y2), cor(pid), 2)
                cv2.putText(quadro, f"ID {pid}", (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor(pid), 2, cv2.LINE_AA)
            cv2.rectangle(quadro, (0, 0), (min(destino[0], 440), 34), (0, 0, 0), -1)
            cv2.putText(quadro, f"Quadro {f} | IDs unicos: {len(vistos)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(quadro)
    finally:
        writer.release()
    print(f"{saida} | {len(imagens)} quadros | {len(ids)} IDs unicos | deteccoes: {fonte}")
    return str(saida), len(ids)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sequencia", help="pasta com img1/ ou pasta de imagens")
    ap.add_argument("--saida", default="inferencia.mp4")
    ap.add_argument("--deteccoes", help="det.txt no formato MOT; se omitido, detecta automaticamente")
    ap.add_argument("--fps", type=float, default=25)
    args = ap.parse_args()
    gerar_video(args.sequencia, args.saida, args.deteccoes, args.fps)
