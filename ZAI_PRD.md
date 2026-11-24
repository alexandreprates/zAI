
Criar um jogo estilo sobrevivencia humanos vs zumbis para treinamento de dois modelos de aprendizado neural, um para controlar o humano e outro para controlar os zumbis.

# Jogo

## Objetivos
    Humano: Sobreviver o maximo de tempo possível
    Zumbis: Encontrar e atacar o Humano

## Mapa
    Area de 500 x 500 metros
    10 casas com duas portas uma na frente e outra nos fundos, as cadas devem ser distribuidas de maneira uniforme pelo mapa;
    Espalhado pelo mapa devem existir pacotes de alimentos que os humanos precisam consumir para se manter vivos, a quantidade de pacotes de comida do mapa deve ser configurável via variavel (padrão 120);

## Regras
                
### Humano

    Ao colidir com um zumbi ele morre;
    Apenas 1 humano por partida;
    Raio de visão de 100 metros, com campo de visão de 120 graus à frente (na direção em que anda);
    Velocidade caminhada 1 metro por segundo;
    Para permanecer vivo deve consumir 1 pacote de comida a cada 90 segundos;
    Pode carregar infinitos pacotes de comida;
    Para cada pacote de comida carregado a velocidade de movimento diminui em 20% da velocidade base (redução linear, não composta: com 5 ou mais pacotes o humano não consegue se mover);
    Para cada pacote de comida coletado no mapa deve disparar um evento de ruido em um raio de 100m;
    Caminhar deve fazer ruido em um raio de 10m;
    Pode fechar a porta de uma casa, esta ação faz ruido raio de 200m;
    Ao colidir com um zumbi ele morre;

### Zumbis

    Colidir com o humano para ataca-lo;
    O numero de zumbis por partida deve ser definido por variavel de ambiente (padrão 20);
    Raio de audição de 300m;
    Velocidade de caminhada de 0.7 metros por segundo;
    Não é capaz de abrir ou fechar portas

### Casas

    Só podem ser acessadas pelas portas;
    Cada casa possui duas portas, sempre que uma for fechada a outra será aberta automáticamente


# Software

Para gerenciar a partida deve existir um servidor que controla as mecanicas do jogo, arranjo do mapa, distribuição dos pacotes de alimentos, eventos de ruidos, notificação de humanos e zumbis do ambiente

Para que a partida possa ser observada o servidor deve disponibilizar uma pagina web com dois tipos de visualização: Visão superior mostrando todo o mapa e todos os elementos ou primeira pessoa (3D) pela perspectiva do jogador;
Na página html alem de observar o mapa deve ser exibido os dados de treinamento/avaliação tanto do modelo do humano quando do modelo dos zumbis

Ambos modelos (controle do humano e controle do zumbi) devem se comumicar com o servidor do jogo para receber as informações do ambiente e movimentar seus personagens


# Infrastrutura

Tanto o server do jogo quando o treniador do modelo devem rodar em containers sem instalar nada na maquina local, os treinadores dos modelos devem ser otimizado para utilizar os recursos de GPU para o treinamento;

# Conclusão

Treinar modelos de aprendizado de máquina que sejam capazes de criar táticas de sobrevivencia (para humanos) e caça em grupo (para zumbis), até que eles desenvolvam táticas que vão alem de ficar vagando no mapa, quero que eles se organizem e aprendam táticas para alcançar seus respectivos objetivos.
