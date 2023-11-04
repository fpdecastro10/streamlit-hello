from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app1 import dataframe_to_markdown

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
    
def concatenar_strings(lista):
    # Inicializa una cadena vacía para la concatenación
    resultado = ""
    
    # Itera a través de la lista
    for elemento in lista:
        # Verifica si el elemento es una cadena
        if isinstance(elemento, str):
            # Concatena la cadena al resultado
            resultado +=(", " + elemento)
    
    return resultado[1:]


data_sw = pd.read_csv("datasetCampignSales.csv")
data_sales_stores = pd.read_csv("dataset_2_Week_later_salesmorethan0.csv")
unique_combination_store_group = data_sw["campaign"].unique()

unique_combinations = data_sw[["store_group_id","name_storeGroup","campaign"]].drop_duplicates(["store_group_id","name_storeGroup","campaign"])

def main():
    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        opciones_seleccionadas = st.selectbox("Filtre por nombre de campaña:", unique_combination_store_group)
        
        unique_combinationsStore = unique_combinations.query(f"campaign in @opciones_seleccionadas")

        index_storeGroup = {}
        filter_list_store_group = []
        for index, row in unique_combinationsStore.iterrows():
            nueva_key=str(row["store_group_id"])+' - ' + row['name_storeGroup']
            filter_list_store_group.append(nueva_key)
            index_storeGroup[nueva_key] = row["store_group_id"]
        temp_index_storeGroup = dict(index_storeGroup)

        botones = [key for key in index_storeGroup]
        selected_filter = st.selectbox("Seleccione un storegroup:", botones)

        min_value_calculated=min(data_sw['ISOweek'])
        max_value_calculated=max(data_sw['ISOweek'])
        selected_time = st.slider("Seleccione la ventana temporal de referencia para el cálculo de crecimiento", min_value=min_value_calculated, max_value=max_value_calculated, value=(min_value_calculated, max_value_calculated))

        numero_ingresado = st.number_input("Ingrese el monto de campaña a invertir", value=0.0, step=0.1)
        
        opcion = st.multiselect('Seleccione los productos que desea que participen de la regresión de stores',list(filter_list_store_group))


    store_group_n = data_sw.query(f"store_group_id == {index_storeGroup[selected_filter]}")
    campaign_value = store_group_n.groupby(["store_group_id","tabla_medio","ISOweek"]).agg({'cost': 'sum', 'sales': 'mean'}).reset_index()
    
    dict_to_calculate = {}
    amount_week = list(campaign_value["ISOweek"].unique())
    tabla_medio = list(campaign_value["tabla_medio"].unique())

    amount_sales = []

    for medio in tabla_medio:
        cost_list = []
        dataframe_filter = campaign_value.query(f"tabla_medio in {[medio]}")
        for row in amount_week:
            campaign_filter = int(dataframe_filter.query(f"ISOweek == {row}")["cost"])
            cost_list.append(campaign_filter)
        dict_to_calculate[medio] = cost_list

    for row in amount_week:
        sales_week = campaign_value.query(f"ISOweek == {row}").groupby("ISOweek").mean().reset_index()["sales"]
        amount_sales.append(int(sales_week))
        
    dict_to_calculate["ISOweek"]=amount_week
    dict_to_calculate["sales"]=amount_sales

    df_tabla_medio_StoreGroup = pd.DataFrame(dict_to_calculate)

    X = df_tabla_medio_StoreGroup[tabla_medio]
    y = df_tabla_medio_StoreGroup["sales"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    model_test = sm.OLS(y, X).fit()
    summary = model_test.summary()
    # print(summary)

    st.markdown(f"<p>Nuestro modelo explica el <b>{round(model_test.rsquared,2)}%</b> de las ventas con respecto a los inputs de {concatenar_strings(tabla_medio)}</p>",unsafe_allow_html=True)

    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_
    coefs = reg.coef_
    st.markdown(f"<h4>Venta de base promedio <b>{round(alpha,2)}</b></h4>",unsafe_allow_html=True)
    
    for medio in range(len(tabla_medio)):
        string_medio = re.sub(r'\bWeekly\b', '', tabla_medio[medio])
        st.markdown(f"<h4>Si gastamos $100 en {string_medio}, esperamos tener una venta adicional de {round(coefs[medio]*100)} unidades</h4>",unsafe_allow_html=True)

    # Graficamos las ventas
    plt.figure(figsize=(8, 6))
    # ax=df_tabla_medio_StoreGroup.plot.scatter(x='Google Weekly', y='sales', color='yellow')
    df_tabla_medio_StoreGroup_sorted = df_tabla_medio_StoreGroup.sort_values(by='ISOweek')
    df_tabla_medio_StoreGroup_sorted_list = list(df_tabla_medio_StoreGroup_sorted["ISOweek"])

    df_tabla_medio_StoreGroup_sorted.plot.scatter(x='ISOweek',y='sales',color='yellow',legend=False)
    plt.xlabel('week')
    plt.ylabel('sales')
    plt.title(f'Store Group Id: {selected_filter}')
    num_ticks = 6
    etiquetas_personalizadas = df_tabla_medio_StoreGroup_sorted_list[::len(df_tabla_medio_StoreGroup_sorted_list) // (num_ticks - 1)]
    plt.xticks(etiquetas_personalizadas)
    plt.ticklabel_format(useOffset=False, style='plain')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    color_list = ["#362FD9","#C70039","green"]
    
    medio = 0
    if len(tabla_medio) > 1:
        for i in range(len(tabla_medio)-1):
            ax=df_tabla_medio_StoreGroup_sorted.plot.scatter(x=tabla_medio[medio], y='sales', color=color_list[medio], label=tabla_medio[medio])
            medio +=1
        df_tabla_medio_StoreGroup_sorted.plot.scatter(x=tabla_medio[medio],y='sales',color=color_list[medio],label=tabla_medio[medio],ax=ax)
    else:
        df_tabla_medio_StoreGroup_sorted.plot.scatter(x=tabla_medio[medio],y='sales',color=color_list[medio],label=tabla_medio[medio])

    plt.xlabel('cost')
    plt.ylabel('sales')
    plt.title(f'Store Group Id: {selected_filter}')
    st.pyplot(plt)


    dataset_analytic_approach = data_sw.query(f"{selected_time[0]} < ISOweek and ISOweek < {selected_time[1]} ")
    list_store_to_filter = []
    for i in opcion:
        list_store_to_filter.append(index_storeGroup[i])
    dataset_analytic_approach_filter = dataset_analytic_approach.query("store_group_id in @list_store_to_filter")
    
    lsit_tabla_medio = dataset_analytic_approach_filter["tabla_medio"].unique()
    calculated = {}
    for media in lsit_tabla_medio:
        list_tabla_medio = dataset_analytic_approach_filter[["tabla_medio","cost_convertion"]].query(f"tabla_medio in {[media]}")["cost_convertion"]
        sum_medio = np.sum(list_tabla_medio)
        avg_medio = np.mean(list_tabla_medio)
        calculated[media] = {"sum":sum_medio,"avg":avg_medio}
    
    total = 0
    for key, value in calculated.items():
        total += value["sum"]
    for key, value in calculated.items():
        calculated[key]["per"] = calculated[key]["sum"]/total

    if numero_ingresado > 0 and opcion != []:
        for medio in tabla_medio:
            percentage = round(numero_ingresado * calculated[medio]["per"])
            medio_iter = medio.split(" ")[0]
            st.markdown(f"<h4>Lo invertido en {medio_iter} debe ser el {percentage}</h4>",unsafe_allow_html=True)




          
