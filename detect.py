"""Parte 1: Faster R-CNN do torchvision (COCO, classe person), sem treino.

box_nms_thresh=1.0 desliga a NMS final do torchvision e a nms() do metrics.py roda no lugar (a
da RPN fica, faz parte do modelo). Grava score >= 0.05 porque o AP precisa dos baixos. Na CPU
leva umas 9 h, entao roda no Kaggle (kaggle_run.py).

  python detect.py                 # as 7 sequencias de treino, grava em dets/
  python detect.py --seqs 02 10
"""
import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

import mot
from metrics import nms


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def detector():
    return fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_nms_thresh=1.0,
                                      box_score_thresh=0.05, box_detections_per_img=1000).eval().to(DEV)


@torch.no_grad()
def detecta(modelo, caminho_img, limiar_nms=0.5):
    """Linhas x, y, w, h, score das pessoas de uma imagem, ja com a nossa NMS."""
    out = modelo([to_tensor(Image.open(caminho_img).convert("RGB")).to(DEV)])[0]
    pessoa = (out["labels"] == 1).cpu().numpy()
    b, sc = out["boxes"].cpu().numpy()[pessoa], out["scores"].cpu().numpy()[pessoa]
    xywh = np.column_stack([b[:, :2], b[:, 2:] - b[:, :2]])
    fica = nms(xywh, sc, limiar_nms)
    return np.column_stack([xywh[fica], sc[fica]]).reshape(-1, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=mot.TODAS)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    modelo = detector()
    os.makedirs(os.path.join(mot.AQUI, "dets"), exist_ok=True)
    for s in args.seqs:
        n, linhas, t0 = mot.info(s)["n"], [], time.time()
        for f in range(1, n + 1):
            for x, y, w, h, sc in detecta(modelo, mot.imagem(s, f)):
                linhas.append([f, -1, x, y, w, h, sc])
            if f % 50 == 0:
                print(f"MOT17-{s} {f}/{n}  {(time.time() - t0) / f:.2f} s/quadro", flush=True)
        np.savetxt(os.path.join(mot.AQUI, "dets", f"MOT17-{s}-TV.txt"), np.array(linhas),
                   fmt="%d,%d,%.1f,%.1f,%.1f,%.1f,%.3f")
        print(f"MOT17-{s}: {len(linhas)} deteccoes em {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
