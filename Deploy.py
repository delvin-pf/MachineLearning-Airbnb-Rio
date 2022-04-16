import pandas as pd
import streamlit as st
import joblib


x_numericos = {'latitude':0, 'longitude':0, 'accommodates':0, 'bathrooms':0, 'bedrooms':0, 'beds':0, 'extra_people':0,
               'minimum_nights':0, 'Mes':0, 'Ano':0, 'n_amenities':0, 'host_listings_count':0}

x_tf = {'host_is_superhost':'f', 'instant_bookable':'f',}

x_listas = {'property_type':['Apartment', 'Bed and breakfast', 'Condominium',
                             'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
            'room_type':['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'cancellation_policy':['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']}

valores = {}

for item in x_listas:
    for tipo in x_listas[item]:
        valores[f'{item}_{tipo}'] = 0

for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format='%.5f')
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:      
        valor = st.number_input(f'{item}', step=1, value=0)
    x_numericos[item] = valor

for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    if valor == 'Sim':
        x_tf[item] = 1
    else:
        x_tf[item] = 0
        
for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    valores[f'{item}_{valor}'] = 1

botao = st.button('Calcular valor')

if botao:
    valores.update(x_numericos)
    valores.update(x_tf)
    data = pd.DataFrame(valores, index=[0])
    modelo = joblib.load('PrevisaoAirBnbRio.joblib')
    preco = modelo.predict(data)
    st.write(preco[0])

