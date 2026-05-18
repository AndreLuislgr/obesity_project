# 🏥 ObesityAI — Sistema Preditivo de Obesidade
**Tech Challenge — FIAP Pós-Tech Data Analytics**

## 📋 Sobre o Projeto

Sistema de apoio à decisão médica para previsão do nível de obesidade, desenvolvido com Machine Learning. O sistema classifica pacientes em 7 categorias — de Abaixo do Peso a Obesidade Grau III — com base em dados físicos, hábitos alimentares e estilo de vida.

## 🎯 Resultados do Modelo

| Métrica | Valor |
|---------|-------|
| Acurácia (Teste) | **97,4%** |
| Validação Cruzada (5-fold) | **98,3% ± 0,4%** |
| F1-Score Macro | **0,97** |
| Requisito mínimo | ~~75%~~ ✅ |

## 🏗️ Arquitetura da Pipeline

```
Obesity.csv
    │
    ▼
1. EDA & Análise Exploratória
    │
    ▼
2. Feature Engineering
   ├── BMI = Weight / Height²
   ├── age_group (faixa etária)
   ├── inactive (flag sedentarismo)
   └── risk_score (score composto)
    │
    ▼
3. Pré-processamento (ColumnTransformer)
   ├── StandardScaler → variáveis numéricas
   └── OneHotEncoder → variáveis categóricas
    │
    ▼
4. Modelo: Random Forest (300 estimadores)
   ├── class_weight='balanced'
   └── Stratified Train/Test Split (80/20)
    │
    ▼
5. Avaliação
   ├── Accuracy Score
   ├── Classification Report
   ├── Confusion Matrix
   └── Stratified K-Fold CV (5 folds)
    │
    ▼
6. Deploy (Streamlit)
   ├── Sistema Preditivo
   └── Dashboard Analítico
```

## 📁 Estrutura do Projeto

```
obesity_project/
├── app.py                    # Aplicação Streamlit
├── pipeline_obesity.ipynb    # Notebook com a pipeline completa de ML
├── Obesity.csv               # Dataset original
├── model.pkl                 # Modelo treinado (Random Forest Pipeline)
├── classes.json              # Labels das classes
├── feature_importance.json   # Importância das features
├── model_info.json           # Métricas do modelo
├── eda_stats.json            # Estatísticas para o dashboard
├── requirements.txt          # Dependências
└── README.md                 # Este arquivo
```

## 🚀 Como Rodar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/obesity-predictor
cd obesity-predictor

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o notebook para treinar o modelo (gera os artefatos .pkl e .json)
jupyter notebook pipeline_obesity.ipynb

# 4. Rode a aplicação Streamlit
streamlit run app.py
```

## 🌐 Deploy no Streamlit Cloud

1. Faça push de todos os arquivos para o GitHub (incluindo `model.pkl` e os `.json`)
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório GitHub
4. Configure: `Main file path = app.py`
5. Clique em **Deploy!**

> ⚠️ O arquivo `model.pkl` deve estar no repositório. Se for muito grande (>100MB), use Git LFS.

## 📊 Funcionalidades do Sistema

### 🔮 Sistema Preditivo
- Formulário interativo para entrada de dados do paciente
- Diagnóstico com 7 classes de peso corporal
- Probabilidades por classe com visualização gráfica
- Recomendações clínicas personalizadas por diagnóstico
- Fatores de risco identificados automaticamente

### 📊 Dashboard Analítico
- KPIs de prevalência da obesidade na base
- Distribuição por nível de obesidade
- IMC médio por categoria
- Análise de histórico familiar × obesidade
- Frequência de atividade física por categoria
- Importância das features do modelo
- 6 insights acionáveis para a equipe médica

## 🔬 Features Utilizadas

| Feature | Tipo | Descrição |
|---------|------|-----------|
| Gender | Categórica | Gênero biológico |
| Age | Numérica | Idade em anos |
| Height | Numérica | Altura em metros |
| Weight | Numérica | Peso em kg |
| family_history | Binária | Histórico familiar de excesso de peso |
| FAVC | Binária | Consumo frequente de alimentos calóricos |
| FCVC | Ordinal (1–3) | Frequência de vegetais nas refeições |
| NCP | Ordinal (1–4) | Número de refeições principais/dia |
| CAEC | Categórica | Consumo entre refeições |
| SMOKE | Binária | Tabagismo |
| CH2O | Ordinal (1–3) | Consumo diário de água |
| SCC | Binária | Monitora calorias |
| FAF | Ordinal (0–3) | Frequência de atividade física |
| TUE | Ordinal (0–2) | Tempo em dispositivos eletrônicos |
| CALC | Categórica | Consumo de álcool |
| MTRANS | Categórica | Meio de transporte |
| **BMI** | Numérica (eng.) | **Índice de Massa Corporal** |
| **age_group** | Categórica (eng.) | **Faixa etária** |
| **inactive** | Binária (eng.) | **Flag de sedentarismo total** |
| **risk_score** | Ordinal (eng.) | **Score de risco comportamental** |

## ⚠️ Disclaimer

Este sistema é uma ferramenta de **apoio à decisão médica** baseada em dados. O diagnóstico final é de responsabilidade exclusiva do médico. O modelo foi treinado em dados sintéticos/augmentados e deve ser validado clinicamente antes de uso em ambiente real.

## 👥 Equipe

Desenvolvido como Tech Challenge da Pós-Tech em Data Analytics — FIAP.
