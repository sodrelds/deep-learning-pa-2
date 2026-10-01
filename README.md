# deep-learning-pa-2

Programming Assignment 2 de Aprendizado Profundo (FGV): rastrear pedestres no MOT17 mantendo a
identidade de cada um ao longo do vídeo, sem rastreador pronto. Estão feitas a Parte 0
(gerador sintético, simulador de detector, métricas e baseline), a Parte 1 (baseline por quadro
com as duas fontes de detecção), a Parte 2 na trilha A (GRU como modelo de movimento), a ablação
da Parte 3 no eixo do regime de treino e a Parte 4 (galeria, horizonte de memória e correção).

## Ambiente

Testado com Python 3.13 e PyTorch 2.11 na CPU. Instalação:

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
`MOT17_DIR` apontar. As imagens são usadas pelo detector do torchvision e pelas três tiras de
falhas da Parte 4; as métricas e os treinos rodam só com as anotações.

Para reproduzir a avaliação da Parte 1 e a ablação da Parte 3, basta o pacote de anotações:

```
mkdir -p data/MOT17
unzip MOT17Labels.zip -d data/MOT17
```

Só as 7 sequências de treino do MOTChallenge têm gt, então o split sai delas, sempre por sequência
inteira: treino 04, 05, 11 e 13, validação 09, teste 02 e 10. No teste ficou uma sequência de
câmera parada com muita oclusão (02) e uma com a câmera andando de noite (10), e o treino ainda
tem câmera parada e densa (04), 14 fps (05) e 25 fps (13).

## Treinar e avaliar

```
python train.py       # treina o GRU e salva em checkpoints/gru_T16_s0.pt
python evaluate.py    # ingênuo, Kalman e GRU nas sequências de teste
python ablacao_regime.py  # Parte 3: 3 regimes x clipping ligado/desligado x 3 seeds
python parte4.py       # horizonte de memória, galeria de falhas e correção
```

O `evaluate.py` usa o limiar de score que a Parte 1 congelou em `resultados/parte1.json`, então na
primeira vez tem que rodar o `python parte1.py` antes. O checkpoint é pequeno e já está no repo.
`ablacao_regime.py` salva cada checkpoint em `checkpoints/ablacao_regime/`, os números por seed e
média ± desvio em `resultados/parte3_regime.json` e o gráfico em `figs/parte3_regime.png`. Pode ser
interrompido e retomado sem repetir configurações já concluídas.

Para a Parte 4, as curvas e métricas rodam só com o `MOT17Labels.zip` (`python parte4.py --sem-galeria`).
Para gerar as três tiras sem baixar o ZIP inteiro de imagens, busque apenas os
21 quadros usados nas figuras e então rode o script completo:

```
python fetch_frames.py '09:104,105,106,139,172,173,175' '02:180,181,182,202,221,222,224' '10:498,499,500,517,533,534,536'
python parte4.py
```

`fetch_frames.py` lê os JPEGs do `MOT17Det.zip` oficial por HTTP Range e os coloca em
`data/MOT17/train/MOT17-XX/img1/`. Se as imagens completas já estiverem extraídas, esse passo é
desnecessário. Todos os números e eventos da Parte 4 ficam em `resultados/parte4.json`.

## O resto

O `parte0.py` roda os testes das métricas, faz a figura do objeto que some atrás do oclusor e mede
o baseline no sintético. A Parte 1 inteira sai do `parte1.py`: AP dos detectores, limiares do
baseline escolhidos nas sequências de treino e o gráfico do descolamento. O Faster R-CNN com a nossa
NMS fica no `detect.py`. As figuras vão pra `figs/` e os números pra `resultados/`.

Na Parte 1, `mAP50_quadros` é a média do AP50 dos quadros que têm pelo menos um pedestre verdadeiro;
cada quadro tem o mesmo peso. `AP50_seq` junta todos os quadros antes de integrar a curva de
precisão e recall. Quadros vazios ficam fora da primeira média porque AP não tem recall definido
neles; seus falsos positivos entram em `AP50_seq`. O gráfico obrigatório usa `mAP50_quadros`.

Na ablação da Parte 3, a caixa prevista sempre alimenta o passo seguinte. O que muda é a entrada
de observações depois dos dez primeiros quadros: `teacher` recebe todas, `scheduled` omite cada
observação com probabilidade que cresce de 0 a 1 ao longo das épocas e `free` não recebe mais
nenhuma. Quando uma observação falta, a entrada da GRU é zero. Cada regime roda com norma de
clipping 1 e com clipping desligado, em três seeds. A melhor época é escolhida pela perda de
previsão livre na sequência de validação 09; o rastreador usa os mesmos limiares congelados em
todas as configurações, e as sequências 02 e 10 são usadas apenas para a avaliação final.

A regra de associação de base é a mesma pra todos os métodos (`tracker.py`): IoU entre a caixa
prevista de cada track e as detecções do quadro, Hungarian com limiar fixo, detecção sem par vira
track nova e a track morre depois de k quadros sem casar. Entre o baseline, o Kalman e o GRU só
muda a caixa prevista. A correção da Parte 4 altera explicitamente o limiar após uma ausência.
Cada arquivo começa com um texto explicando as escolhas dele.

`metrics.py`, `synth.py`, `tracker.py` e `model.py` têm um teste rápido no `__main__`, por exemplo
`python metrics.py`.

## Resultados reproduzidos

Na Parte 1, a média do mAP50 por quadro nas sete sequências foi 0,434 (DPM), 0,555 (FRCNN),
0,666 (SDP) e 0,685 (torchvision). SDP continua sendo a melhor das três fontes públicas; ela é a
fonte congelada nas Partes 2 e 3. Os números por sequência, incluindo o AP50 agregado, estão em
`resultados/parte1.json`.

Na Parte 3, todos os valores abaixo são média ± desvio amostral de três seeds. O IDF1 de teste é
a média das sequências 02 e 10 em cada seed; IDSW é a soma das duas sequências. O clipping usa
norma máxima 1 quando ligado.

| Regime | Clipping | IDF1 val 09 | IDF1 teste 02+10 | IDSW teste 02+10 |
| --- | --- | ---: | ---: | ---: |
| teacher | ligado | 0,617 ± 0,013 | 0,455 ± 0,015 | 838 ± 27 |
| teacher | desligado | 0,609 ± 0,012 | 0,456 ± 0,005 | 820 ± 33 |
| scheduled | ligado | 0,603 ± 0,031 | 0,453 ± 0,010 | 844 ± 13 |
| scheduled | desligado | 0,603 ± 0,023 | 0,450 ± 0,002 | 833 ± 9 |
| free | ligado | 0,590 ± 0,042 | 0,480 ± 0,009 | 780 ± 37 |
| free | desligado | 0,603 ± 0,022 | 0,470 ± 0,007 | 764 ± 43 |

No teste, o ganho do treino livre com clipping sobre teacher com clipping veio sobretudo da
sequência 10: IDF1 médio 0,522 contra 0,484; na 02, 0,437 contra 0,427. Na validação 09, teacher
teve IDF1 maior. Portanto, o resultado de teste é uma análise da ablação, não uma escolha de
hiperparâmetro feita nas sequências de teste. Sem clipping, nenhum treino divergiu; os resultados
de IDF1 ficaram próximos nas três seeds, com leve queda no treino livre. A maior média por época
da norma do gradiente antes de clipping foi cerca de 2,0 em teacher/scheduled e 0,72 em free.

## Parte 4: horizonte, falhas e correção

O grupo `teacher` com clipping teve o maior IDF1 médio na validação 09 na Parte 3; a seed 0 foi
a melhor desse grupo na mesma sequência. Por isso a Parte 4 usa
`checkpoints/ablacao_regime/teacher_clip1_s0.pt`, sem escolher pelo teste. Uma oclusão é um
trecho completo com visibilidade do GT abaixo de 0,3.

Na medição analítica, a norma relativa de ∂L_t/∂h_(t-k) foi 0,459 em k=15 e 0,171 em k=63,
em 96 janelas reais de MOT17-09/02/10 com observações ruidosas. Portanto, a GRU não teve queda
de 20 vezes dentro dos 64 passos medidos. O treino, porém, corta o grafo a cada T=16: o
gradiente **usado para atualizar os pesos é zero além desse limite**, mesmo quando o estado da
GRU ainda carrega informação. A curva e os valores estão em `figs/parte4_gradiente.png` e
`resultados/parte4.json`.

Na medição empírica, há 300 oclusões completas no GT; 232 tinham um ID previsto antes do buraco.
O mesmo ID voltou em 7/34 buracos de 31–60 quadros e em 2/23 de 61–120. A distribuição de
duração do GT, o tempo observado até morte/troca de ID e as frações por faixa estão em
`figs/parte4_empirico.png`. Eventos sem ID antes da oclusão não entram na taxa de sobrevivência;
tempos sem falha são censurados na volta da pessoa.

Três falhas reais estão em `figs/parte4_falha_09_gt21.png`, `figs/parte4_falha_02_gt9.png` e
`figs/parte4_falha_10_gt14.png`. Na 09, o GT 21 fica oculto por 67 quadros e a track P3 morre
no quadro 165, oito quadros antes do retorno. Na 02, o GT 9 fica oculto por 40 quadros; a track
continua viva, mas sua previsão chega com IoU 0 contra o GT. Na 10, o GT 14 fica oculto por 34
quadros; a IoU entre a previsão e a detecção correta na volta é 0,249, abaixo do portão 0,3.
As três tiras incluem GT e predições por ID e a caixa prevista pela GRU em cada quadro.

Para corrigir o último tipo de falha, a associação passa a aceitar IoU 0,2 após dez quadros sem
observação. A escolha se baseou num caso da validação 09 com IoU previsão–detecção de 0,227;
0,25 teve IDF1 de validação ligeiramente maior, mas ainda excluiria essa detecção. As sequências
de teste não entraram nessa escolha. O antes/depois com o mesmo checkpoint e as
mesmas detecções é:

| Sequência | IDF1 antes | IDF1 depois | ID switches antes | ID switches depois |
| --- | ---: | ---: | ---: | ---: |
| 09 (validação) | 0,626 | 0,647 | 48 | 53 |
| 02 (teste) | 0,429 | 0,434 | 345 | 319 |
| 10 (teste) | 0,508 | 0,495 | 467 | 462 |

A mudança recuperou o mesmo ID nos casos GT 9 da sequência 02 e GT 14 da 10, mas reduziu o IDF1
agregado da 10. Isso indica que abrir o portão recupera algumas pessoas e também cria associações
erradas numa cena com câmera móvel. A comparação completa está em `figs/parte4_correcao.png` e
`resultados/parte4.json`.
