
import base64
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
        stores_id_list = []
        retailer_id_list = []
        coefficients_list = []
        percentage_week_of_cero = []
        week_with_cero_sales = []

        for store in stores_id:
            dataset_after_filter_sorted_by_store = dataset_after_filter_sorted.query(f"id_store_retailer == {store} and {selected_time[0]} < ISOweek and ISOweek < {selected_time[1]}")
            y = list(dataset_after_filter_sorted_by_store["ISOweek"])
            x = list(dataset_after_filter_sorted_by_store["sales"])
            if not all(elemento == 0 for elemento in x):
                coefficients = np.polyfit(x,y,1)
                stores_list.append(store)
                coefficients_list.append(str(round(coefficients[0])))
                stores_id_list.append(dataset_after_filter_sorted_by_store["id_store"].unique()[0])
                retailer_id_list.append(dataset_after_filter_sorted_by_store["retailer_name"].unique()[0])
                
                list_with_cero_sales_per_week = list(map(str,list(dataset_after_filter_sorted_by_store[['ISOweek','sales']].query("sales <= 0")['ISOweek'])))
                
                total_number_of_week = list(dataset_after_filter_sorted_by_store['ISOweek'].unique())
                percentage_week_of_cero.append(str(round((len(list_with_cero_sales_per_week) / len(total_number_of_week))*100,2)))
                
                week_with_cero_sales.append(", ".join(list_with_cero_sales_per_week))

        dataframe_coefficients = {
            "stores": stores_list,
            "Store id" : stores_id_list,
            "Retailer" : retailer_id_list,
            "Tendencia de venta" : coefficients_list,
            "% Semanas con venta cero" : percentage_week_of_cero,
            "Semanas con ventas cero" : week_with_cero_sales
        }
        
        df_stores = pd.DataFrame(dataframe_coefficients).sort_values("Tendencia de venta").head(10)

        stores_filter = df_stores["stores"] 
        store_selected = st.selectbox("Stores con tendencia negativa:", stores_filter)

    dataset_after_filter_sorted_by_store_and_time_window = dataset_after_filter_sorted.query(f"id_store_retailer == {store_selected} and {selected_time[0]} < ISOweek and ISOweek < {selected_time[1]}")
    negative_tendecy_of_stores = dataset_after_filter_sorted_by_store_and_time_window[["ISOweek","sales"]]

    boundaries = dict(negative_tendecy_of_stores['sales'].describe())
    
    df_25 =  negative_tendecy_of_stores.query(f"sales <= {boundaries['25%']}")
    df_50 =  negative_tendecy_of_stores.query(f"{boundaries['25%']} < sales & sales <= {boundaries['75%']}")
    df_75 =  negative_tendecy_of_stores.query(f"{boundaries['75%']} < sales")

    retailer_name = dataset_after_filter_sorted_by_store_and_time_window['retailer_name'].unique()[0]

    negative_tendecy_of_stores['row_number'] = range(1, len(negative_tendecy_of_stores) + 1)

    negative_tendecy_of_stores['ISOweek'] = negative_tendecy_of_stores['ISOweek'].astype('str')
    df_25['ISOweek'] = df_25['ISOweek'].astype('str')
    df_50['ISOweek'] = df_50['ISOweek'].astype('str')
    df_75['ISOweek'] = df_75['ISOweek'].astype('str')
    
    d = np.polyfit(negative_tendecy_of_stores["row_number"],negative_tendecy_of_stores["sales"],1)
    f = np.poly1d(d)
    negative_tendecy_of_stores.insert(1,"Treg",f(negative_tendecy_of_stores["row_number"]))
    
    plt.figure(figsize=(8, 6))
    
    # ax = negative_tendecy_of_stores.plot.scatter(x="ISOweek",y="sales",color="yellow")
    # negative_tendecy_of_stores.plot.scatter(x="ISOweek",y="Treg",color="red",legend=False,ax=ax)
    
    plt.scatter(negative_tendecy_of_stores['ISOweek'],negative_tendecy_of_stores['Treg'], color='#F5FCCD')
    plt.scatter(df_25['ISOweek'],df_25['sales'],color='#F8DE22',label='sales<=25')
    plt.scatter(df_50['ISOweek'],df_50['sales'],color='#F94C10',label='25<sales<=75')
    plt.scatter(df_75['ISOweek'],df_75['sales'],color='#C70039',label='75<sales')

    ISOweek_negative_tendecy = list(negative_tendecy_of_stores["ISOweek"].unique())
    num_ticks = 6
    etiquetas_personalizadas = ISOweek_negative_tendecy[::len(ISOweek_negative_tendecy) // (num_ticks - 1)]
    plt.xticks(etiquetas_personalizadas)
    plt.xlabel('week')
    plt.ylabel('sales')
    st.markdown("<h3 style='text-align:center'>Tendecias de ventas</h3>",unsafe_allow_html=True)
    plt.legend()
    title_graph = f"""
    Store Id: {store_selected} - Retailer: {retailer_name}
    Store Group: {selected_filter}"""
    plt.title(title_graph)
    # Plot the trendline
    st.pyplot(plt)
    st.markdown("<h3 style='text-align:center'>Top 10 stores con mayor tendencia negativa dentro del Store Group seleccionado</h3>", unsafe_allow_html=True)
    df_stores_formated = df_stores.to_html(index=False).replace("<td>","<td style='text-align:center'>").replace("<table border='1' class='dataframe'>","<table border='1' class='dataframe' sytle='margin:auto'>").replace("<th>","<th style='text-align:center'>")
    st.markdown(df_stores_formated,unsafe_allow_html=True)

    csv_data = df_stores.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="exported_data.csv">Descargar CSV</a>'
    st.markdown("<div style='height:40px'>",unsafe_allow_html=True)
    st.markdown(href,unsafe_allow_html=True)