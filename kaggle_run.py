"""Roda o detect.py na GPU do Kaggle. Na CPU daqui seriam umas 9 h.

Os .py que o detector usa vao em base64 dentro de um notebook, igual no PA1. O notebook baixa o
MOT17 do site do MOTChallenge, roda o detector nas 7 sequencias e a pasta dets/ volta no output.

  python kaggle_run.py            # manda o notebook pro Kaggle
  python kaggle_run.py --status
  python kaggle_run.py --baixar   # copia os dets/*.txt do output pra ca
"""
import argparse
import base64
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

AQUI = os.path.dirname(os.path.abspath(__file__))
KERNEL = "joaovtaf/pa2-deteccoes"

PREPARO = '''
import base64, io, os, subprocess, sys, tarfile, zipfile
os.makedirs("/kaggle/working/pa2", exist_ok=True)
os.chdir("/kaggle/working/pa2")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(SRC)), mode="r:gz").extractall(".")

import torch
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma", cap, flush=True)
if torch.cuda.is_available() and cap[0] < 7:
    # a P100 e sm_60 e o torch do Kaggle so cobre sm_70+, o mesmo problema do PA1
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "torchvision==0.20.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

# os dados ficam em /kaggle/temp, que nao volta no output
DADOS = "/kaggle/temp/MOT17"
os.makedirs(DADOS, exist_ok=True)
for z in ["MOT17Labels.zip", "MOT17Det.zip"]:
    # o servidor do MOTChallenge derruba download grande as vezes, o -C - continua de onde parou
    subprocess.run(f"cd {DADOS} && for i in 1 2 3 4 5 6; do curl -s -C - -o {z} https://motchallenge.net/data/{z} && break; sleep 10; done",
                   shell=True, check=True)
    zipfile.ZipFile(os.path.join(DADOS, z)).extractall(DADOS)
    os.remove(os.path.join(DADOS, z))
print(sorted(os.listdir(os.path.join(DADOS, "train"))), flush=True)
'''

RODA = '''
r = subprocess.run("python detect.py", shell=True, env={**os.environ, "MOT17_DIR": DADOS})
print("detect.py saiu com", r.returncode, flush=True)
'''


def celula(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def monta(pasta):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in ["mot.py", "metrics.py", "detect.py"]:
            tf.add(os.path.join(AQUI, p), arcname=p)
    src = base64.b64encode(buf.getvalue()).decode()
    nb = {"cells": [celula(f'SRC = "{src}"'), celula(PREPARO), celula(RODA)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(pasta, exist_ok=True)
    json.dump(nb, open(os.path.join(pasta, "pa2-deteccoes.ipynb"), "w", encoding="utf-8"))
    json.dump({"id": KERNEL, "title": "pa2-deteccoes", "code_file": "pa2-deteccoes.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
               "enable_internet": True, "dataset_sources": [], "competition_sources": [],
               "kernel_sources": []}, open(os.path.join(pasta, "kernel-metadata.json"), "w"), indent=1)


def kaggle(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--baixar", action="store_true")
    args = ap.parse_args()
    pasta = os.path.join(AQUI, "kaggle")
    if args.status:
        print(kaggle("kernels", "status", KERNEL))
    elif args.baixar:
        saida = os.path.join(pasta, "saida")
        print(kaggle("kernels", "output", KERNEL, "-p", saida))
        os.makedirs(os.path.join(AQUI, "dets"), exist_ok=True)
        for f in glob.glob(os.path.join(saida, "**", "MOT17-*-TV.txt"), recursive=True):
            shutil.copy(f, os.path.join(AQUI, "dets"))
            print("copiado", os.path.basename(f))
    else:
        monta(pasta)
        print(kaggle("kernels", "push", "-p", pasta))
        print(f"acompanhar: https://www.kaggle.com/code/{KERNEL}")


if __name__ == "__main__":
    main()
