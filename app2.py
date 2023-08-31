from itertools import groupby
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use({
    'axes.facecolor': '#0e1118',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#0e1118',
    'figure.facecolor': '#0e1118',
    'figure.edgecolor': '#0e1118',
    'savefig.facecolor': '#0e1118',
    'savefig.edgecolor': '#1a1a1a',
})

data_sw = pd.read_csv('dataset_2_Week_later_salesmorethan0.csv')
data_sw['cost_campaign_dist'] = (data_sw['cost_campaign']/data_sw["cantidad_de_stores_por_storeGroup"])/ data_sw["cant_de_prod_por_storeGroup"]
data_sw['cpc_campaign_dist'] = (data_sw['cpc_campaign']/data_sw["cantidad_de_stores_por_storeGroup"])/ data_sw["cant_de_prod_por_storeGroup"]
data_sw['cpm_campaign_dist'] = (data_sw['cpm_campaign']/data_sw["cantidad_de_stores_por_storeGroup"])/ data_sw["cant_de_prod_por_storeGroup"]
data_sw['ctr_campaign_dist'] = (data_sw['ctr_campaign']/data_sw["cantidad_de_stores_por_storeGroup"])/ data_sw["cant_de_prod_por_storeGroup"]
data_sw['impressions_campaign_dist'] = (data_sw['impressions_campaign']/data_sw["cantidad_de_stores_por_storeGroup"])/ data_sw["cant_de_prod_por_storeGroup"]

storeGroup_idStoreId = data_sw[['id_storeGroup','id_store_retailer']].drop_duplicates(subset=['id_storeGroup','id_store_retailer'])
setIdStoresGroupsIdStores = {}
listIdGroups = (storeGroup_idStoreId['id_storeGroup'].unique())
# Construimos un diccionario con key: id_storeGroup y value: stores_id
for idsGroups in listIdGroups:
    h = set(storeGroup_idStoreId.query(f"id_storeGroup == {idsGroups}")['id_store_retailer'])
    setIdStoresGroupsIdStores[idsGroups] = h

def main():
    st.title("Aplicación 2")
    # Lógica de la primera aplicación
    with st.sidebar:
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
        if uploaded_file is not None:
            st.write("Archivo cargado con éxito!")
            dataventasmodelo = pd.read_csv(uploaded_file)
    if uploaded_file is not None:
        skuNew_Stores = set(dataventasmodelo['store_id'].drop_duplicates())
        comp_withStores = {}
        for key, value in setIdStoresGroupsIdStores.items():
            comp_withStores[key] = len(skuNew_Stores.intersection(value))

        valor_maximo = max(comp_withStores.values())
        claves_maximas = [clave for clave, valor in comp_withStores.items() if valor == valor_maximo]
        st.markdown(f"<h2 style=color:#f7dc00> Comportamiento similar encontrado con los stores groups: </h2>",unsafe_allow_html=True )
        opcion_seleccionada = st.radio("", claves_maximas)
        
        
        dataVentasWeek = dataventasmodelo.groupby('ISOweek').sum().reset_index()
        
        
        filter_data_storeGroup = data_sw.query(f"id_storeGroup == {opcion_seleccionada}")
        
        filtered_data = filter_data_storeGroup.groupby("yearweek_campaign").sum().reset_index()
        d = np.polyfit(filtered_data['cost_campaign_dist'],filtered_data['sales'],2)
        f = np.poly1d(d)
        filtered_data.insert(1,'Treg',f(filtered_data['cost_campaign_dist']))

        h = []
        j = []
        for i in range(1000):
            h.append(i)
            j.append(f(i))

        plt.figure(figsize=(8, 6))
        # ax=dataVentasWeek.plot.scatter(x='ISOweek', y='sales', color='yellow')
        # dataVentasWeek.plot.scatter(x='ISOweek',y='sales',color='red',legend=False)

        # Crear el gráfico interactivo
        # plt.xlabel('ISOweek')
        # plt.ylabel('sales')
        # plt.title(f'Gráfico filtrado:')
        # st.pyplot(plt)

        plt.scatter(h, j, label="Mis Datos", color="blue")
        st.pyplot(plt)


    



if __name__ == "__main__":
    main()
