# bdtech.ai
Desafio Técnico – Hackathon Forecast Big Data 2025

Objetivo:
Desenvolver um modelo de previsão de vendas (forecast) para apoiar o varejo na reposição de produtos. A tarefa é prever a quantidade semanal de vendas por PDV (Ponto de Venda) /SKU Stock Keeping Unit

(ou Unidade de Manutenção de Estoque) para as cinco semanas de janeiro/2023, utilizando como base o histórico de vendas de 2022.

##script:
forecast_pipeline_v1.py

## Estrtutura:

**Autor:** Wanderlei Soares  
**Email:** wanderlei.junior@gmail.com

## Descrição
Este script implementa um pipeline de previsão utilizando técnicas de aprendizado de máquina, com foco no modelo LightGBM para regressão. O objetivo é processar dados, treinar modelos, realizar previsões e salvar os resultados de forma eficiente.

## Funcionalidades Principais
- Carregamento e pré-processamento dos dados
- Treinamento do modelo LightGBM
- Avaliação de desempenho do modelo
- Geração de previsões
- Salvamento dos resultados e logs

## Estrutura do Código
1. **Importação de bibliotecas**: Importa pacotes essenciais para manipulação de dados, modelagem e avaliação.
2. **Configuração de parâmetros**: Define parâmetros do modelo e caminhos para arquivos de entrada/saída.
3. **Carregamento dos dados**: Lê os dados de treino e teste a partir de arquivos Parquet.
4. **Pré-processamento**: Realiza limpeza, transformação e seleção de variáveis relevantes.
5. **Treinamento do modelo**: Ajusta o modelo LightGBM aos dados de treino.
6. **Avaliação**: Calcula métricas de desempenho como RMSE, MAE, etc.
7. **Previsão**: Aplica o modelo treinado aos dados de teste.
8. **Exportação dos resultados**: Salva previsões e logs em arquivos CSV e TXT.

## Como Executar


1. Certifique-se de que todas as dependências estão instaladas (consulte o 
`requirements.txt`).
    1.1 Shell
        python -m pip install -r requirements.txt


2. Execute o script via terminal:
   
   ```bash
   forecast_pipeline_v1.py --data-dir dataset/train/parquet --output-csv results/jan2023_v1_sample.csv --validation-weeks 5 --forecast-weeks 5
   ```

3. Os resultados serão salvos na pasta especificada no código.

## Estrutura de Pastas Necessária

```
├── dataset/
│   └── train/
│       └── parquet/
│           ├── pdvs.parquet
│           ├── produtos.parquet
│           └── transacoes_2022.parquet
├── results/
│   └── jan2023_v1_sample.csv
├── src/
│   └── forecast_pipeline_v1.py
├── requirements.txt
|
```

### Descrição de cada item

- `dataset/train/parquet/`: Pasta onde ficam os dados de entrada do projeto, em formato Parquet.
    - `pdvs.parquet`: Dados dos pontos de venda.
    - `produtos.parquet`: Dados dos produtos.
    - `transacoes_2022.parquet`: Histórico de transações para treinamento.
- `results/`: Pasta onde os resultados das previsões serão salvos.
    - `jan2023_v1_sample.csv`: Exemplo de arquivo gerado com as previsões para janeiro de 2023.
- `src/`: Pasta dos scripts do projeto.
    - `forecast_pipeline_v1.py`: Script principal do pipeline de previsão.
    
- `requirements.txt`: Lista de dependências Python necessárias para rodar o projeto.


