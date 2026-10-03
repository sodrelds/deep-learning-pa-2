# AI_LOG

Usamos Claude (Claude Code no VS Code) no PA2 como apoio, principalmente nas partes mais difíceis.
As escolhas de projeto foram nossas. Abaixo os episódios em que a IA fez parte da solução.

## Distratores do MOT17

Além dos pedestres, o gt do MOT17 marca pessoa parada, pessoa em veículo, reflexo e outros
distratores (classes 2, 7, 8 e 12), e detectar essas coisas não deveria contar nem como acerto nem
como erro. Quem chamou atenção pra isso foi a IA, e foi dela também a regra de tirar a caixa
prevista só quando ela casa com um distrator num casamento feito junto com os pedestres. Se o
distrator fosse tirado sozinho, uma caixa em cima de um pedestre de verdade podia sumir só porque
tinha um distrator do lado. Isso fica no
`tira_distratores()` do `metrics.py` e vale pro AP e pro rastreamento.

## Contagem de ID switch

Na primeira versão da contagem, cada quadro fazia o Hungarian do zero, e duas pessoas lado a lado
trocavam de par por ruído na caixa. O switch entrava na conta sem ter acontecido no rastreamento.
A correção veio da IA: um par gt/previsto que casou no quadro anterior e ainda passa do limiar
continua casado, e só o resto vai pro Hungarian.

## A primeira versão do modelo não aprendia movimento

A primeira versão do GRU recebia a caixa absoluta normalizada pelo tamanho da imagem. Rodamos 60
épocas e ele quase não aprendeu movimento. A IA achou o problema: nessa escala a mudança entre
dois quadros é de uns 0,001, e o GRU precisaria de pesos enormes pra tirar velocidade disso. A
correção dela foi passar a observação relativa à própria previsão, no mesmo formato de
deslocamento em relação à âncora dos slides de detecção, e aí a entrada fica na ordem de 1. Ela
também chegou a sugerir dar a velocidade direto como entrada, e isso a gente recusou: na trilha A
quem tem que guardar de onde a pessoa vinha é o estado da recorrência.

## Horizonte de memória analítico

A Parte 4 pede a norma de ∂L_t/∂h_(t-k) em função de k. A IA sugeriu guardar os estados da GRU e
usar `torch.autograd.grad` pra medir duas curvas no mesmo modelo: uma com o grafo inteiro e outra
com o detach a cada 16 passos, igual ao treino. Em 96 janelas reais das sequências 09, 02 e 10, a
curva com o grafo inteiro cai devagar, com norma relativa de 0,459 em k=15 e 0,171 em k=63, então
a GRU não tem a queda de 20 vezes dentro de 64 passos. Com o detach do treino, o gradiente além de
16 passos é zero, então o sinal de supervisão nunca atravessou mais que 16 quadros, mesmo a GRU
conseguindo passar gradiente bem além disso.

## Horizonte de memória empírico

Pra medir quanto tempo o estado sobrevive a uma oclusão, cada buraco completo no gt (visibilidade
abaixo de 0,3) virou um evento, e a IA ajudou a definir o que conta como falha e o que fica de
fora. Falha é a track morrer ou ser contaminada, que é quando o id dela passa a casar com outra
pessoa durante o buraco, mesmo com a track ainda viva. Buraco em que a pessoa não tinha id antes
de sumir fica fora da taxa, e o tempo de quem não falhou é censurado na volta da pessoa, porque
não dá pra saber quanto mais ele aguentaria. Dos 300 buracos completos, 232 tinham id antes, e o
mesmo id voltou em 7 de 34 buracos de 31 a 60 quadros e em 2 de 23 de 61 a 120.

## Quadros da galeria

Pras tiras da galeria da Parte 4 precisávamos de só 21 quadros, e baixar o ZIP de 1,9 GB pra isso
não fazia sentido. O Claude sugeriu ler o ZIP oficial por HTTP Range, já que o servidor aceita, e
é isso que o `fetch_frames.py` faz.

## Gráficos

Todos os gráficos em `figs/`, da Parte 0 à Parte 5, foram feitos pelo Claude. Os números que eles
mostram saem dos scripts de cada parte e ficam salvos em `resultados/`.
