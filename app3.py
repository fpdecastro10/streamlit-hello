import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from app1 import dataframe_to_markdown

zip_file_path = "dataset_to_detect_performance_of_stores.csv.zip"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(".")

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

data_sw = pd.read_csv("dataset_to_detect_performance_of_stores.csv")

unique_combinations = data_sw[["id_storeGroup","name_storeGroup","campaign_storeGroup"]].drop_duplicates(["id_storeGroup","name_storeGroup","campaign_storeGroup"])
unique_combinations_campaign = unique_combinations["campaign_storeGroup"].drop_duplicates()
relations_storeGroup_products = data_sw[["id_storeGroup","sku_id"]].drop_duplicates(["id_storeGroup","sku_id"])

def main():
    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        opciones_seleccionadas = st.selectbox("Filtre por nombre de campaña:", unique_combinations_campaign)
        unique_combinationsStore = unique_combinations.query(f"campaign_storeGroup in @opciones_seleccionadas")

        index_storeGroup = {}
        for index, row in unique_combinationsStore.iterrows():
            nueva_key=str(row["id_storeGroup"])+' - ' + row['name_storeGroup']
            index_storeGroup[nueva_key] = row["id_storeGroup"]
        
        # temp_index_storeGroup = dict(index_storeGroup)
        # for key, value in temp_index_storeGroup.items():
        #     if data_sw.query(f"id_storeGroup == {value}").groupby("yearweek_campaign").sum().reset_index().shape[0] < 6:
        #         index_storeGroup.pop(key)

        if index_storeGroup == {}:
            botones = ['No tiene sufieciente datos de campaña']
            selected_filter = st.selectbox("Seleccione un storegroup:", botones)
        else:
            botones = [key for key in index_storeGroup]
            selected_filter = st.selectbox("Seleccione un storegroup:", botones)

        id_storeGroup_filter = index_storeGroup[selected_filter]
        allProducts = relations_storeGroup_products.query(f"id_storeGroup == {id_storeGroup_filter}")["sku_id"]
        sku_selected = st.selectbox("Seleccione el sku_id:", allProducts)


        dataset_after_filter = data_sw.query(f"campaign_storeGroup in @opciones_seleccionadas and id_storeGroup == {id_storeGroup_filter} and sku_id == {sku_selected}")
        
        
        dataset_after_filter_sorted = dataset_after_filter.sort_values(by="ISOweek")

        min_value_calculated=min(dataset_after_filter['ISOweek'])
        max_value_calculated=max(dataset_after_filter['ISOweek'])
        selected_time = st.slider("Seleccione la ventana temporal de referencia para el cálculo de crecimiento", min_value=min_value_calculated, max_value=max_value_calculated, value=(min_value_calculated, max_value_calculated))
        
        stores_id = list(dataset_after_filter["id_store_retailer"].unique())
        stores_list = []
        coefficients_list = []

        for store in stores_id:
            dataset_after_filter_sorted_by_store = dataset_after_filter_sorted.query(f"id_store_retailer == {store} and {selected_time[0]} < ISOweek and ISOweek < {selected_time[1]}")
            y = list(dataset_after_filter_sorted_by_store["ISOweek"])
            x = list(dataset_after_filter_sorted_by_store["sales"])
            if not all(elemento == 0 for elemento in x):
                coefficients = np.polyfit(x,y,1)
                stores_list.append(store)
                coefficients_list.append(coefficients[0])
        dataframe_coefficients = {
            "stores": stores_list,
            "coefficients" : coefficients_list
        }
        
        df_stores = pd.DataFrame(dataframe_coefficients).sort_values("coefficients").head(10)

        stores_filter = df_stores["stores"] 
        store_selected = st.selectbox("Stores con tendencia negativa:", stores_filter)

    negative_tendecy_of_stores = dataset_after_filter_sorted.query(f"id_store_retailer == {store_selected} and {selected_time[0]} < ISOweek and ISOweek < {selected_time[1]}")[["ISOweek","sales"]]
    d = np.polyfit(negative_tendecy_of_stores["ISOweek"],negative_tendecy_of_stores["sales"],1)
    f = np.poly1d(d)
    negative_tendecy_of_stores.insert(1,"Treg",f(negative_tendecy_of_stores["ISOweek"]))
    
    plt.figure(figsize=(8, 6))
    ax = negative_tendecy_of_stores.plot.scatter(x="ISOweek",y="sales",color="yellow")
    negative_tendecy_of_stores.plot.scatter(x="ISOweek",y="Treg",color="red",legend=False,ax=ax)
    # etiquetas_personalizadas = df_tabla_medio_StoreGroup_sorted_list[::len(df_tabla_medio_StoreGroup_sorted_list) // (num_ticks - 1)]

    ISOweek_negative_tendecy = list(negative_tendecy_of_stores["ISOweek"].unique())
    num_ticks = 6
    etiquetas_personalizadas = ISOweek_negative_tendecy[::len(ISOweek_negative_tendecy) // (num_ticks - 1)]
    plt.xticks(etiquetas_personalizadas)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('week')
    plt.ylabel('sales')
    plt.title(f'Store Id: {selected_filter}')
    # Plot the trendline
    st.pyplot(plt)
    st.markdown(dataframe_to_markdown(df_stores))
