# Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns

# Importar Bibliotecas e Bases de Dados
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import time

caminho = pathlib.Path('dataset')

baseAirbnb = pd.DataFrame()
meses = {'jan': 1, 'fev':2, 'mar':3, 'abr':4, 'mai':5, 'jun':6, 'jul':7, 'ago':8, 'set':9, 'out':10, 'nov':11, 'dez':12}

for arquivo in caminho.iterdir():
    mes = meses[arquivo.name[:3]] # extrair o mês do nome do arquivo
    ano = int(arquivo.name[-8:-4]) # extrair o ano do nome do arquivo
    tabela = pd.read_csv(caminho / arquivo.name)
    tabela['Mes'] = mes # adicionar una coluna com o mês
    tabela['Ano'] = ano # adicionar una coluna com o ano
    baseAirbnb = baseAirbnb.append(tabela) # adicionar no df principal

# Consolidar Base de Dados
baseAirbnb.to_csv('BaseAirbnb.csv', sep=';', index = False) # Salvar no arquivo .csv o df principal completa

# Identificar quais colunas vamos manter
colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','Mes','Ano']

baseAirbnb = baseAirbnb.loc[:, colunas]
display(baseAirbnb)

# Tratar Valores Faltando
for coluna in baseAirbnb:
    if baseAirbnb[coluna].isnull().sum() > 300000:
        baseAirbnb = baseAirbnb.drop(coluna, axis=1)

baseAirbnb = baseAirbnb.dropna()

display(baseAirbnb.isnull().sum())

#Verificar Tipos de Dados em cada coluna

# baseAirbnb = baseAirbnb.reset_index(drop=True)

display(baseAirbnb.dtypes)
print('-'*60)
display(baseAirbnb.loc[0])

# Limpando e formatando dados monetarios

baseAirbnb['price'] = baseAirbnb['price'].str.replace('$', '')
baseAirbnb['price'] = baseAirbnb['price'].str.replace(',', '')
baseAirbnb['price'] = baseAirbnb['price'].astype(np.float32, copy=False)

baseAirbnb['extra_people'] = baseAirbnb['extra_people'].str.replace('$', '')
baseAirbnb['extra_people'] = baseAirbnb['extra_people'].str.replace(',', '')
baseAirbnb['extra_people'] = baseAirbnb['extra_people'].astype(np.float32, copy=False)

display(baseAirbnb.dtypes)

#Análise Exploratória e Tratar Outliers

plt.figure(figsize=(20,15))
sns.heatmap(baseAirbnb.corr(), annot=True, cmap='Blues')

# Definir funções

def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(20, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    
def excluir_outliers(df, coluna):
    qtLinhas = df.shape[0]
    limInf, limSup = limites(df[coluna])
    df = df.loc[ (df[coluna] >= limInf) & (df[coluna] <= limSup), :]
    return df, qtLinhas - df.shape[0]

def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


diagrama_caixa(baseAirbnb['price'])
histograma(baseAirbnb['price'])

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'price')
print(f'{removidas} linhas removidas')

histograma(baseAirbnb['price'])


# Analizar coluna extra_people

diagrama_caixa(baseAirbnb['extra_people'])
histograma(baseAirbnb['extra_people'])

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'extra_people')
print(f'{removidas} linhas removidas')

# ANalizar coluna hosting_listing_count

diagrama_caixa(baseAirbnb['host_listings_count'])
grafico_barra(baseAirbnb['host_listings_count'])

# podemos escluir os outliers. Por que hosting com mais de 6 imoveis não são o objetivo do projeto

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'host_listings_count')
print(f'{removidas} linhas removidas')


# Analizar coluna accommodates

diagrama_caixa(baseAirbnb['accommodates'])
grafico_barra(baseAirbnb['accommodates'])

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'accommodates')
print(f'{removidas} linhas removidas')


# Analizar coluna bathrooms

diagrama_caixa(baseAirbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=baseAirbnb['bathrooms'].value_counts().index, y=baseAirbnb['bathrooms'].value_counts())

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'bathrooms')
print(f'{removidas} linhas removidas')


# Analizar coluna bedrooms

diagrama_caixa(baseAirbnb['bedrooms'])
grafico_barra(baseAirbnb['bedrooms'])

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'bedrooms')
print(f'{removidas} linhas removidas')


# ANalizar coluna beds

diagrama_caixa(baseAirbnb['beds'])
grafico_barra(baseAirbnb['beds'])

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'beds')
print(f'{removidas} linhas removidas')


# Analizar coluna guests_included

#diagrama_caixa(baseAirbnb['guests_included'])
#grafico_barra(baseAirbnb['guests_included'])

plt.figure(figsize=(15, 5))
sns.barplot(x=baseAirbnb['guests_included'].value_counts().index, y=baseAirbnb['guests_included'].value_counts())
# Essa coluna sera removida da analise. O valor é insuficiente. Além pelas metricas atuais teria que excluir uma grande quantidade de dados
baseAirbnb = baseAirbnb.drop('guests_included', axis=1)
display(baseAirbnb.shape)


#Analizar coluna minimum_nights

diagrama_caixa(baseAirbnb['minimum_nights'])
grafico_barra(baseAirbnb['minimum_nights'])

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'minimum_nights')
print(f'{removidas} linhas removidas')


# Analizar coluna maximum_nights

diagrama_caixa(baseAirbnb['maximum_nights'])
grafico_barra(baseAirbnb['maximum_nights'])

baseAirbnb = baseAirbnb.drop('maximum_nights', axis=1)
display(baseAirbnb.shape)


# Analizar coluna number_of_reviews

diagrama_caixa(baseAirbnb['number_of_reviews'])
grafico_barra(baseAirbnb['number_of_reviews'])

baseAirbnb = baseAirbnb.drop('number_of_reviews', axis=1)
display(baseAirbnb.shape)


# Analizar colunas de texto

# property_type
plt.figure(figsize=(15,5))
grafico = sns.countplot('property_type', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=85)
tabela = baseAirbnb['property_type'].value_counts()

for i in tabela.index:
    if tabela[i] < 2000:
        baseAirbnb.loc[baseAirbnb['property_type'] == i,  'property_type'] = 'Other'
        
print(baseAirbnb['property_type'].value_counts())
plt.figure(figsize=(15,5))
grafico = sns.countplot('property_type', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=15)


# room_type
plt.figure(figsize=(15,5))
grafico = sns.countplot('room_type', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=15)


# bed_type
plt.figure(figsize=(15,5))
grafico = sns.countplot('bed_type', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=15)
print(baseAirbnb['bed_type'].value_counts())
tabela = baseAirbnb['bed_type'].value_counts()

for i in tabela.index:
    if tabela[i] < 10000:
        baseAirbnb.loc[baseAirbnb['bed_type'] == i,  'bed_type'] = 'Others'
        
plt.figure(figsize=(15,5))
grafico = sns.countplot('bed_type', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=15)
print(baseAirbnb['bed_type'].value_counts())

# cancellation_policy
plt.figure(figsize=(15,5))
grafico = sns.countplot('cancellation_policy', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=15)
print(baseAirbnb['cancellation_policy'].value_counts())
tabela = baseAirbnb['cancellation_policy'].value_counts()

for i in tabela.index:
    if tabela[i] < 10000:
        baseAirbnb.loc[baseAirbnb['cancellation_policy'] == i,  'cancellation_policy'] = 'strict'
        
print(baseAirbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15,5))
grafico = sns.countplot('cancellation_policy', data=baseAirbnb)
grafico.tick_params(axis='x', rotation=15)
print(baseAirbnb['cancellation_policy'].value_counts())

# amenities
# Existem uma diversidade muito grande desse valor, e pode não ser um valor exato. Por isso tomamremos como base a quantidade de amenities e não o tipo de amenities.
# Sera criada uma nova coluna com o novo valor e excluida a coluna 'amenities'

print(baseAirbnb['amenities'].iloc[0].split(','))
print(len(baseAirbnb['amenities'].iloc[0].split(',')))

baseAirbnb['n_amenities'] = baseAirbnb['amenities'].str.split(',').apply(len)
baseAirbnb = baseAirbnb.drop('amenities', axis=1)
baseAirbnb.shape

diagrama_caixa(baseAirbnb['n_amenities'])
grafico_barra(baseAirbnb['n_amenities'])
print(baseAirbnb['n_amenities'].value_counts())

baseAirbnb, removidas = excluir_outliers(baseAirbnb, 'n_amenities')
print(f'{removidas} linhas removidas')

# Visualizando mapas com lonigude e latitude

amostra = baseAirbnb.sample(n=5000)
centroMapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}

mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
                        center=centroMapa, zoom=10, mapbox_style='stamen-terrain')
mapa.show()

# Encoding
# Variaveis True ou False

colunasTF = ['host_is_superhost','instant_bookable', 'is_business_travel_ready']

baseAirbnb_cod = baseAirbnb.copy()
for coluna in colunasTF:
    baseAirbnb_cod.loc[baseAirbnb_cod[coluna] == 't' , coluna] = 1
    baseAirbnb_cod.loc[baseAirbnb_cod[coluna] == 'f' , coluna] = 0
        
display(baseAirbnb_cod.iloc[0])

# Variavies de descrição
colunas_cat = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
baseAirbnb_cod = pd.get_dummies(data=baseAirbnb_cod, columns = colunas_cat)

display(baseAirbnb_cod.iloc[0])

# Modelo de Previsão
def avaliar_modelo(nmModelo, yTeste, previsao):
    r2 = r2_score(yTeste, previsao)
    RSME = np.sqrt(mean_squared_error(yTeste, previsao))
    return f'Modelo {nmModelo}:\nr2={r2:.2%}\nRSME={RSME:.2f}'

# modelos
modelo_LR = LinearRegression()
modelo_RF = RandomForestRegressor()
modelo_ET = ExtraTreesRegressor()

modelos = {'LinearRegressor': modelo_LR,
          'RandomForest': modelo_RF,
          'ExtraTrees': modelo_ET,
          }

y = baseAirbnb_cod['price']
x = baseAirbnb_cod.drop(['price'], axis=1)

# Separar dados em treino e testes e treino dos modelos

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

for nmModelo, modelo in modelos.items():
    # treinar
    modelo.fit(x_train, y_train)
    # testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nmModelo, y_test, previsao))
        


# Análise do Melhor Modelo

for nmModelo, modelo in modelos.items():
    # testar
    inicio = time.time()
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nmModelo, y_test, previsao))
    fim = time.time()
    tempo = fim - inicio
    print(f'Tempo de execução: {tempo:,.3f}')
    print('\n')


# - Melhor modelo: ExtraTreesRegressor
#     Modelo com maior valor de R2 e ao mesmo tempo menor valor de RSME

# Ajustes e Melhorias no Melhor Modelo

print(modelo_ET.feature_importances_)
print(x_train.columns)

importancia = pd.DataFrame(modelo_ET.feature_importances_, x_train.columns)
importancia = importancia.sort_values(by=0, ascending=False)
display(importancia)

plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia.index, y=importancia[0])
ax.tick_params(axis='x', rotation=90)

# Ajustes de presição no modelo
# is_business_travel_ready tem uma importancia de 0

baseAirbnb_cod = baseAirbnb_cod.drop('is_business_travel_ready', axis=1)

y = baseAirbnb_cod['price']
x = baseAirbnb_cod.drop(['price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)
display(baseAirbnb_cod.columns)

inicio = time.time()
modelo_ET.fit(x_train, y_train)
fim = time.time()
tempo = fim - inicio
print(f'Tempo de treino: {tempo:,.3f}')

inicio = time.time()
previsao = modelo_ET.predict(x_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))
fim = time.time()
tempo = fim - inicio
print(f'Tempo de teste: {tempo:,.3f}')

baseTeste = baseAirbnb_cod.copy()
for coluna in baseTeste:
    if 'bed_type' in coluna:
        baseTeste = baseTeste.drop(coluna, axis=1)

y = baseTeste['price']
x = baseTeste.drop(['price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

inicio = time.time()
modelo_ET.fit(x_train, y_train)
fim = time.time()
tempo = fim - inicio
print(f'Tempo de treino: {tempo:,.3f}')
print('\n')

inicio = time.time()
previsao = modelo_ET.predict(x_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))
fim = time.time()
tempo = fim - inicio
print(f'Tempo de teste: {tempo:,.3f}')
print('\n')

baseTeste.to_csv('Dados_Refinados.csv', sep=';', index=False) # Salvar a tabela codificada final

import joblib

joblib.dump(modelo_ET, 'PrevisaoAirBnbRio.joblib') # salvar o modelo em uma arquivo externo (preservação de IA)

