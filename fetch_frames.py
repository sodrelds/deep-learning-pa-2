"""Baixa do MOT17Det.zip, por HTTP Range, so os quadros da galeria da Parte 4 (vao pra data/MOT17).

  python fetch_frames.py 02:155,156,157 09:308,309,380 10:187,188,291
"""
import argparse
import io
import os
import urllib.request
import zipfile
from collections import OrderedDict

import mot

URL = "https://motchallenge.net/data/MOT17Det.zip"


class RemoteZip(io.RawIOBase):
    BLOCO = 1 << 20

    def __init__(self, url=URL):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            self.tamanho = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.tamanho + offset
        else:
            raise ValueError("whence invalido")
        return self.pos

    def _bloco(self, indice):
        if indice in self.cache:
            self.cache.move_to_end(indice)
            return self.cache[indice]
        inicio = indice * self.BLOCO
        fim = min(inicio + self.BLOCO, self.tamanho) - 1
        pedido = urllib.request.Request(self.url, headers={"Range": f"bytes={inicio}-{fim}"})
        with urllib.request.urlopen(pedido, timeout=120) as r:
            if r.status != 206:
                raise RuntimeError("O servidor nao retornou HTTP 206 para o intervalo solicitado")
            data = r.read()
        if len(data) != fim - inicio + 1:
            raise IOError("Intervalo incompleto")
        self.cache[indice] = data
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n < 0:
            n = self.tamanho - self.pos
        n = min(n, self.tamanho - self.pos)
        partes = []
        while n > 0:
            indice, dentro = divmod(self.pos, self.BLOCO)
            parte = self._bloco(indice)[dentro:dentro + n]
            partes.append(parte)
            self.pos += len(parte)
            n -= len(parte)
        return b"".join(partes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quadros", nargs="*", help="SEQ:frame,frame,...")
    ap.add_argument("--list", action="store_true", help="lista os primeiros arquivos do ZIP")
    args = ap.parse_args()
    with zipfile.ZipFile(RemoteZip()) as z:
        if args.list:
            print("\n".join(z.namelist()[:40]))
        for item in args.quadros:
            seq, frames = item.split(":")
            for frame in map(int, frames.split(",")):
                nome = f"train/MOT17-{seq}/img1/{frame:06d}.jpg"
                destino = os.path.join(mot.ROOT, nome)
                if os.path.exists(destino):
                    print("ja existe", destino, flush=True)
                    continue
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                with z.open(nome) as fonte, open(destino, "wb") as saida:
                    saida.write(fonte.read())
                print("baixado", destino, flush=True)


if __name__ == "__main__":
    main()
