# Topicos-ADCS-2022-1_Attitude-Determination

![](https://img.shields.io/badge/version-v0.1-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

Este repositório foi desenvolvido no contexto da disciplina de Tópicos Especiais em Engenharia Elétrica da Universidade de Brasília (UnB) e implementa uma versão adaptada da rede neural desenvolvida por *Dos Santos et al.* para determinação da atitude.

> Dos Santos, G. H., Seman, L. O., Bezerra, E. A., Leithardt, V., Mendes, A. S., & Stefenon, S. F. (2021). ["Static Attitude Determination Using Convolutional Neural Networks"](https://www.mdpi.com/1424-8220/21/19/6419/pdf). Sensors (Basel, Switzerland), 21(19), 6419. https://doi.org/10.3390/s21196419

## Configurações

- Versão do Python: 3.9.5
- Versão do PyTorch: 1.12.1

### Ambiente de desenvolvimento

1. Crie o ambiente

```bash
python -m venv pTAD
```

2. Ative o ambiente

    - Em ambiente Windows (bash do git)
    
    ```bash
    source pTAD/Scripts/activate
    ```
    
    - Em ambiente Linux
    
    ```bash
    source pTAD/bin/activate
    ```

3. Instale as dependências

```bash
pip install -r requirements.txt
```

## Como usar

### Para treinar ou testar o modelo

```bash
(pTAD)$ python src/main.py -h
usage: main.py [-h] -m /path/to/weights.pth [-e first_epoch-last_epoch]
               <command>

Train B-Swish on Attitude profile dataset

positional arguments:
  <command>             Set the mode to 'training' or'test'.

options:
  -h, --help            show this help message and exit
  -m /path/to/weights.pth, --model /path/to/weights.pth
                        Path to weights (.pth file), 'last' or 'first'.
  -e first_epoch-last_epoch, --epochs first_epoch-last_epoch
                        Range of epochs to training.
```

- Exemplo de comando de treino

```bash
python src/main.py training --model first --epochs 1-200
```

- Exemplo de comando de teste

```bash
python src/main.py test --model last
```

### Para avaliar o modelo com outros dados

```bash
(pTAD)$ python eval/evaluate.py -h
usage: evaluate.py [-h] -i /path/to/input.json -m /path/to/weights.pth
                   [-o /path/to/output.json] [-w a1, a2, a3]
                   [-p {axis-angle,quaternion}]

Evaluate B-Swish model

options:
  -h, --help            show this help message and exit
  -i /path/to/input.json, --input /path/to/input.json
                        Path to input file (.json file).
  -m /path/to/weights.pth, --model /path/to/weights.pth
                        Path to weights (.pth file), 'last' or 'first'.
  -o /path/to/output.json, --output /path/to/output.json
                        Path to input file (.json file).
  -w a1, a2, a3, --wahba_weights a1, a2, a3
                        Coma separeted weights for Wahba's problem. (Default:
                        1.0, 1.0, 1.0)
  -p {axis-angle,quaternion}, --parameterization {axis-angle,quaternion}
                        Parameterization of output attitude matrix. (Default:
                        axis-angle)
```

- Exemplo de comando

```bash
python eval/evaluate.py --input data/arquivo/entrada.json --model last
```

#### Formato do arquivo de entrada

O arquivo de entrada deve ser um ".json" com o seguinte formato
```bash
{
    "n": 2,
    "r": {
        "B_eci": "2435.1449234364, 9209.57092801303, 18414.3477578444",
        "Sun_vec_eci": "-0.692699291225791, 0.667793036145862, 0.272433758573305"
    },
    "b": {
        "B_magnetometer": "25538.4613037109, 5461.53869628906, -33307.6934814453",
        "Sun_vec_body_frame": "-0.372567440599065, 0.228045590381382, 0.899549170925675"
    }
}
```

- n: Número de vetores em cada campo;
- r: Vetores no sistema de referência;
- b: Vetores medidos no corpo.

O nome dos vetores é livre, mas devem estar no formato indicado com a separação dos valores por vírgula e espaço.

#### Formato do arquivo de saída

O arquivo de saída será um ".json" com o seguinte formato
```bash
{
    "Parameterization": "axis-angle",
    "Line 1": "0.47368383407592773, -0.2572217881679535, 0.8422948718070984",
    "Line 2": "0.2843320667743683, 0.949848473072052, 0.13016605377197266",
    "Line 3": "-0.8335339426994324, 0.17783388495445251, 0.5230643153190613"
}
```

- Parameterization: Parametrização da matriz de atitude;
- Line 1: Primeira linha da matriz de atitude;
- Line 2: Segunda linha da matriz de atitude;
- Line 3: Terceira linha da matriz de atitude.

## Resultados

### Treino

![](data/imagem/erro_treinamento.png)

Resultados ao final do treinamento

|   Etapa   | Erro geodésico |
|:---------:|:--------------:|
|   Treino  |     5,278º     |
| Validação |     1,863º     |

### Teste

|      Erro      | Valor do erro |
|:--------------:|:-------------:|
| Erro geodésico |     2,098º    |
|  Erro de Wahba |     0,002     |