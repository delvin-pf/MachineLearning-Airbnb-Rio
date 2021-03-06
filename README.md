# MachineLearning-Airbnb-Rio
 Using machine learning to predict Airbnb's prices 
 
### Contexto 
No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.

### Objetivo / Utilidade
Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel. Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.

### Arquivos
- As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro. As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
Os preços são dados em reais (R$).
Dados de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados. **Obs:** Da base de dados disponivel no link, não foi usado o arquivo total_data.csv

- AirbnbRio.py : Arquivo do projeto em formato .py
- Airbnb.Rio.ipynb: Arquivo do projeto em formato do Notebook Jupyter
- Deploy.py: Arquivo de deploy da IA para futuras previsões



### Expectativas Iniciais
A sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ

A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)

Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro

