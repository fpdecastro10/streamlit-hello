import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from app1 import dataframe_to_markdown
from scipy.stats import zscore

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

data_sw = pd.read_csv("datasetCampignSales.csv")
data_sales_stores = pd.read_csv("dataset_2_Week_later_salesmorethan0.csv")

list_campaing_store_group = data_sw["campaign"].unique().tolist()
unique_combinations = data_sw[[
                            "store_group_id",
                            "name_storeGroup",
                            "campaign"]].drop_duplicates(["store_group_id","name_storeGroup","campaign"])

unique_combination_id_name_storeGroup = data_sw[[
    "store_group_id",
    "name_storeGroup",
    "campaign"]].drop_duplicates(["store_group_id","name_storeGroup","campaign"])

def return_storeGroup_of_Campaign(campaign_variable):
    combinations_storeGroup_nameStoreGroup = unique_combinations.query(f"campaign in @campaign_variable")
    return combinations_storeGroup_nameStoreGroup

def search_key_based_on_Value(diccionario, valor_buscado):
    # Itera a través del diccionario
    for clave, valor in diccionario.items():
        if valor == valor_buscado:
            return clave
    # Si no se encuentra el valor, devuelve None o una cadena vacía u otro valor predeterminado
    return None


def concatenar_strings(lista):
    resultado = ""
    for elemento in lista:
        if isinstance(elemento, str):
            resultado +=(", " + elemento)
    return resultado[1:]

def dict_number_name_storeGroup(df_filter):
    index_storeGroup = {}
    filter_list_store_group = []
    for index, row in df_filter.iterrows():
        nueva_key=str(row["store_group_id"])+' - ' + row['name_storeGroup']
        filter_list_store_group.append(nueva_key)
        index_storeGroup[nueva_key] = row["store_group_id"]
    return index_storeGroup, filter_list_store_group    

def df_builder_tablaMedio(df_tablaMedio_ISOweek):
    pivot_table_tablaMdio_cost = pd.pivot_table(df_tablaMedio_ISOweek,values='cost',index='ISOweek',columns="tabla_medio")
    isoWeek_sales_origin = df_tablaMedio_ISOweek[["ISOweek","sales"]].groupby("ISOweek").mean().reset_index()
    union_sales_tablaMedio_cost = pd.merge(pivot_table_tablaMdio_cost, isoWeek_sales_origin, on='ISOweek', how='left')
    return union_sales_tablaMedio_cost


def graph_dataset_ISOweek_sales(df, columnsx,columnsy,storeGroupId):
    plt.figure(figsize=(8, 6))
    df_temp = df.sort_values(by=columnsx)
    df_temp[columnsx] = df_temp[columnsx].astype('str')
    list_columnx_sorted = df_temp[columnsx].tolist()
    num_ticks = 6
    customize_label = list_columnx_sorted[::len(list_columnx_sorted) // (num_ticks - 1)]
    df_temp.plot.scatter(x=columnsx,y=columnsy,color='yellow',legend=True)
    plt.xticks(customize_label)
    plt.xlabel('week')
    plt.ylabel('sales')
    plt.title(f'Store Group Id: {storeGroupId}')
    st.pyplot(plt)

def graph_tablaMedio_sales(df,tabla_medio_list,storeGroupId):
    plt.figure(figsize=(8, 6))
    color_list = ["#362FD9","#C70039","green"]    
    medio = 0
    if len(tabla_medio_list) > 1:
        for i in range(len(tabla_medio_list)-1):
            ax=df.plot.scatter(x=tabla_medio_list[medio], y='sales', color=color_list[medio], label=tabla_medio_list[medio])
            medio +=1
        df.plot.scatter(x=tabla_medio_list[medio],y='sales',color=color_list[medio],label=tabla_medio_list[medio],ax=ax)
    else:
        df.plot.scatter(x=tabla_medio_list[medio],y='sales',color=color_list[medio],label=tabla_medio_list[medio])

    plt.xlabel('cost')
    plt.ylabel('sales')
    plt.title(f'Store Group Id: {storeGroupId}')
    st.pyplot(plt)


def first_approach(storeGroup_id):
    df_storeGroup_id = data_sw.query(f"store_group_id == {storeGroup_id}")

    df_tablaMedio_ISOweek = df_storeGroup_id.groupby([
        "store_group_id",
        "tabla_medio",
        "ISOweek"
        ]).agg({
            'cost': 'sum',
            'sales': 'mean' }).reset_index()

    unique_tablaMedio = df_tablaMedio_ISOweek["tabla_medio"].unique().tolist()

    df_tablaMedio_listCost = df_builder_tablaMedio(df_tablaMedio_ISOweek)
    
    X = df_tablaMedio_listCost[unique_tablaMedio]
    y = df_tablaMedio_listCost["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    model_test = sm.OLS(y, X).fit()
    summary = model_test.summary()
    
    st.markdown(f"<p>Nuestro modelo explica el <b>{round(model_test.rsquared,2)}%</b> de las ventas con respecto a los inputs de {concatenar_strings(unique_tablaMedio)}</p>",unsafe_allow_html=True)

    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_
    coefs = reg.coef_
    st.markdown(f"<h4>Venta de base promedio <b>{round(alpha,2)}</b></h4>",unsafe_allow_html=True)

    for medio in range(len(unique_tablaMedio)):
        string_medio = re.sub(r'\bWeekly\b', '', unique_tablaMedio[medio])
        st.markdown(f"<h4>Si gastamos $100 en {string_medio}, esperamos tener una venta adicional de {round(coefs[medio]*100)} unidades</h4>",unsafe_allow_html=True)

    graph_dataset_ISOweek_sales(df_tablaMedio_listCost,'ISOweek','sales',storeGroup_id)
    graph_tablaMedio_sales(df_tablaMedio_listCost,unique_tablaMedio,storeGroup_id)


def second_approach(input_number,list_seleceted_stores, dict_key_number_storeGroup,selected_time,bool_execute):
    if bool_execute:
        df_filter_by_time = data_sw.query(f"{selected_time[0]} < yearweek and yearweek < {selected_time[1]}")
        list_store_to_filter = []
        for storeFroup in list_seleceted_stores:
            list_store_to_filter.append(dict_key_number_storeGroup[storeFroup])
        df_filter_storeGroups = df_filter_by_time.query("store_group_id in @list_store_to_filter")
        list_tabla_medio = df_filter_storeGroups["tabla_medio"].unique().tolist()
        dict_tablaMedio_sum_avg = {}
        for medio in list_tabla_medio:
            list_tabla_medio_cost_convertion = df_filter_storeGroups[["tabla_medio","cost_convertion"]].query(f"tabla_medio in {[medio]}")["cost_convertion"]
            sum_medio = np.sum(list_tabla_medio_cost_convertion)
            avg_medio = np.mean(list_tabla_medio_cost_convertion)
            dict_tablaMedio_sum_avg[medio] = {"sum":sum_medio,"avg":avg_medio}
        total = 0
        for key, value in dict_tablaMedio_sum_avg.items():
            total += value["sum"]
        for key, value in dict_tablaMedio_sum_avg.items():
            dict_tablaMedio_sum_avg[key]["per"] = dict_tablaMedio_sum_avg[key]["sum"]/total
        
        df_sales_filter_by_date_list_StoreGroup = data_sales_stores.query(f"id_storeGroup in @list_store_to_filter and {selected_time[0]} < ISOweek and ISOweek < {selected_time[1]}")
        for medio in list_tabla_medio:
            percentage = round(input_number * dict_tablaMedio_sum_avg[medio]["per"])
            medio_iter = medio.split(" ")[0]
            st.markdown(f"<h4>Lo invertido en {medio_iter} debe ser el {percentage}</h4>",unsafe_allow_html=True)

            data_filter = df_sales_filter_by_date_list_StoreGroup.groupby("id_storeGroup").agg({"id_store_retailer":"nunique","sales":"sum"}).reset_index()
            data_filter["per tiendas"] = data_filter['id_store_retailer'] / data_filter['id_store_retailer'].sum() * 100
            data_filter["per sales"] = data_filter['sales'] / data_filter['sales'].sum() * 100
            data_filter["KPI"] =  data_filter["per sales"] / data_filter["per tiendas"]
            data_filter["share tiendas budget"] = (data_filter["per tiendas"]/100) * percentage
            data_filter["share ventas budget"] = (data_filter["per sales"]/100) * percentage
            data_filter["investment"] = (data_filter["share tiendas budget"] + data_filter["share ventas budget"]) / 2
            
            store_group_investment = {"Store Group":[],"Inversión":[]}
            for indice, fila in data_filter.iterrows():
                store_group_id = fila['id_storeGroup']
                investment = round(fila['investment'])
                store_group_investment["Store Group"].append(search_key_based_on_Value(dict_key_number_storeGroup,store_group_id))
                store_group_investment["Inversión"].append(investment)
                
            st.markdown(dataframe_to_markdown(pd.DataFrame(store_group_investment)), unsafe_allow_html=True)

def coefficient_tabla_medio(storeGroup_id):
    df_storeGroup_id = data_sw.query(f"store_group_id == {storeGroup_id} & sales > 0")
    df_tablaMedio_ISOweek = df_storeGroup_id.groupby([
        "store_group_id",
        "tabla_medio",
        "ISOweek"
        ]).agg({
            'cost': 'sum',
            'sales': 'mean' }).reset_index()
    
    unique_tablaMedio = df_tablaMedio_ISOweek["tabla_medio"].unique().tolist()

    df_tablaMedio_listCost = df_builder_tablaMedio(df_tablaMedio_ISOweek)
    df_tablaMedio_listCost =df_tablaMedio_listCost.dropna()
    unique_tablaMedio_sales = unique_tablaMedio + ["sales"]
    z_scores = zscore(df_tablaMedio_listCost[unique_tablaMedio_sales])
    threshold = 2

    outliers = (abs(z_scores) > threshold).all(axis=1)
    df_tablaMedio_listCost = df_tablaMedio_listCost[~outliers]

    X = df_tablaMedio_listCost[unique_tablaMedio]
    y = df_tablaMedio_listCost["sales"]
    
    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_
    coefs = reg.coef_

    dict_result = {}
    dict_result[storeGroup_id] = {'alpha':alpha}
    for medio in range(len(unique_tablaMedio)):
        dict_result[storeGroup_id][f"coefs_{unique_tablaMedio[medio]}".replace(" ","_")] = coefs[medio]/alpha

    return dict_result

def third_percent(list_storeGroups_id):
    df_accumulator = pd.DataFrame()
    for storeGroup in list_storeGroups_id:
        dict_coeff_storesGroups = coefficient_tabla_medio(storeGroup)
        result_dict = pd.Series(dict_coeff_storesGroups[storeGroup],name=storeGroup)
        df_accumulator = pd.concat([df_accumulator, result_dict.to_frame().T], ignore_index=False)
    
    medio_avg = {}
    for column in df_accumulator.columns:
        serie_column = df_accumulator[column]
        serie_positive = serie_column > 0
        medio_avg[column] = np.mean(serie_column[serie_positive])
    st.write(medio_avg)


def main():
    st.title("Estamos creando una nueva aplicación")

    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        type_of_client = st.selectbox("Tipo de cliente:", ["Cliente Nuevo","Cliente con Historial"])

        selected_option = st.selectbox("Filtre por nombre de campaña:", list_campaing_store_group)
        number_name_StoreGroup = return_storeGroup_of_Campaign(selected_option)
        key_number_name_storeGroup,number_name_storeGroup = dict_number_name_storeGroup(number_name_StoreGroup)

        # botones = [key for key in key_number_name_storeGroup]
        selected_filter = st.selectbox("Seleccione un storegroup:", number_name_storeGroup)

        min_value_calculated=min(data_sw['ISOweek'])
        max_value_calculated=max(data_sw['ISOweek'])
        selected_time = st.slider(  
            "Seleccione la ventana temporal de referencia para el cálculo de crecimiento",
            min_value=min_value_calculated,
            max_value=max_value_calculated,
            value=(min_value_calculated, max_value_calculated)
        )

        input_number = st.number_input("Ingrese el monto de campaña a invertir", value=0.0, step=0.1)
        key_number_name_name_storeGroup2,list_id_name_storeGroup =  dict_number_name_storeGroup(unique_combination_id_name_storeGroup)
        list_of_storeGroup_selected = st.multiselect('Seleccione los productos que desea que participen de la regresión de stores',list(list_id_name_storeGroup))

    first_approach(key_number_name_storeGroup[selected_filter])

    bool_aproach2 = input_number > 0 and list_of_storeGroup_selected != []
    second_approach(input_number,list_of_storeGroup_selected,key_number_name_name_storeGroup2,selected_time,bool_aproach2)

    list_of_store_group = data_sw["store_group_id"].unique().tolist()
    third_percent(list_of_store_group)
