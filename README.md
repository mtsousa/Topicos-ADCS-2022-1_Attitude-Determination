# Topicos-ADCS-2022-1_Attitude-Determination

## Configurações

- Versão do Python: 3.9.5

### Crie um ambiente de desenvolvimento virtual

- No ambiente Windows, no bash do git, execute
```bash
python -m venv pTAD
```

- No ambiente Linux, execute
```bash
python3.9 -m venv pTAD
```

### Ative o ambiente de desenvolvimento virtual

- No ambiente Windows, no bash do git, execute
```bash
source pTAD/Scripts/activate
```

- No ambiente Linux, execute
```bash
source pTAD/bin/activate
```

### Instale as dependências

```bash
pip install -r requirements.txt
```

## Como usar

### Para treinar o modelo

- Com o ambiente de desenvolvimento ativo, execute
```bash
python src/main.py training --model [WEIGHTS] --epochs [INICIO]-[FIM]
```

Em que:
- [INICIO]: Primeira época de treino;
- [FIM]: Última época de treino; e
- E [WEIGHTS] deve ser substituído por
* Caminho dos pesos, como 'model/bswish_model_0005.pth';
* O peso do último treino, ou seja, 'last'; ou
* 'first' para indicar que não tem pesos prévios.

### Para avaliar o modelo

- Com o ambiente de desenvolvimento ativo, execute
```bash
python src/main.py evaluate --model [WEIGHTS]
```

Em que [WEIGHTS] deve ser substituído por
* Caminho dos pesos, como 'model/bswish_model_0005.pth'; ou
* O peso do último treino, ou seja, 'last'.
