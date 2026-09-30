# deep-learning-pa-2

Programming Assignment 2 de Aprendizado Profundo (FGV): rastrear pedestres no MOT17 mantendo a
identidade de cada um ao longo do vídeo, sem rastreador pronto. Até agora estão feitas a Parte 0
(gerador sintético, simulador de detector, métricas e baseline), a Parte 1 (baseline por quadro
com as duas fontes de detecção) e a Parte 2 na trilha A, com um GRU como modelo de movimento.

## Ambiente

Python 3.14 e PyTorch 2.12 na CPU. Instalação:

```
pip install torch torchvision numpy scipy matplotlib pillow
```

Tudo roda na CPU menos o Faster R-CNN do torchvision, que aqui leva uns 6 s por quadro. Ele rodou
no Kaggle pelo `kaggle_run.py`, que embarca os `.py` num notebook, baixa o MOT17 lá e devolve as
detecções. Elas já estão em `dets/`, então não precisa rodar de novo pra reproduzir o resto.

## Dados

Do site do MOTChallenge só usamos dois arquivos: o `MOT17Labels.zip` (10 MB, gt e detecções
públicas) e o `MOT17Det.zip` (1.9 GB, as imagens, cada sequência uma vez só). O `MOT17.zip`
completo repete as mesmas imagens pros três detectores e pesa 5.5 GB.

```
curl -O https://motchallenge.net/data/MOT17Labels.zip
curl -O https://motchallenge.net/data/MOT17Det.zip
```

Os dois vão extraídos na mesma pasta. O código procura em `data/MOT17`, ou onde a variável
`MOT17_DIR` apontar. As imagens só são usadas pelo detector do torchvision, o resto roda só com os
labels.

Só as 7 sequências de treino do MOTChallenge têm gt, então o split sai delas, sempre por sequência
inteira: treino 04, 05, 11 e 13, validação 09, teste 02 e 10. No teste ficou uma sequência de
câmera parada com muita oclusão (02) e uma com a câmera andando de noite (10), e o treino ainda
tem câmera parada e densa (04), 14 fps (05) e 25 fps (13).

## Treinar e avaliar

```
python train.py       # treina o GRU e salva em checkpoints/gru_T16_s0.pt
python evaluate.py    # ingênuo, Kalman e GRU nas sequências de teste
```

O `evaluate.py` usa o limiar de score que a Parte 1 congelou em `resultados/parte1.json`, então na
primeira vez tem que rodar o `python parte1.py` antes. O checkpoint é pequeno e já está no repo.

## O resto

O `parte0.py` roda os testes das métricas, faz a figura do objeto que some atrás do oclusor e mede
o baseline no sintético. A Parte 1 inteira sai do `parte1.py`: AP dos detectores, limiares do
baseline escolhidos nas sequências de treino e o gráfico do descolamento. O Faster R-CNN com a nossa
NMS fica no `detect.py`. As figuras vão pra `figs/` e os números pra `resultados/`.

A regra de associação é a mesma pra todos os métodos (`tracker.py`): IoU entre a caixa prevista de
cada track e as detecções do quadro, Hungarian com limiar fixo, detecção sem par vira track nova e
a track morre depois de k quadros sem casar. Entre o baseline, o Kalman e o GRU só muda a caixa
prevista. Cada arquivo começa com um texto explicando as escolhas dele.

`metrics.py`, `synth.py`, `tracker.py` e `model.py` têm um teste rápido no `__main__`, por exemplo
`python metrics.py`.
