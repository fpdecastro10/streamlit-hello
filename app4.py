import streamlit as st
import plotly.express as px
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
import zipfile

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

data_sw = pd.read_csv("datasetCampignSales.csv")
data_sales_stores = pd.read_csv("dataset_2_Week_later_salesmorethan0.csv")
if 'df_sales_storeGroup' in  st.session_state:
    df_sales_storeGroup = st.session_state.df_sales_storeGroup
else:
    st.session_state.df_sales_storeGroup = pd.read_csv("dataset_to_detect_performance_of_stores.csv")
    df_sales_storeGroup = st.session_state.df_sales_storeGroup

list_campaing_store_group = data_sw["campaign"].unique().tolist()
list_campaing_store_group_new_client = df_sales_storeGroup.query("campaign_storeGroup not in @list_campaing_store_group")["campaign_storeGroup"].unique().tolist()
list_campaing_store_group_new_client.append("PCT_STAGEI")

unique_combinations = data_sw[[
                            "store_group_id",
                            "name_storeGroup",
                            "campaign"]].drop_duplicates(["store_group_id","name_storeGroup","campaign"])

unique_combinations_new_client = df_sales_storeGroup[[
                                    "id_storeGroup",
                                    "name_storeGroup",
                                    "campaign_storeGroup"]].drop_duplicates(["id_storeGroup","name_storeGroup","campaign_storeGroup"])

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

def dict_number_name_storeGroup(df_filter,col1, col2):
    index_storeGroup = {}
    filter_list_store_group = []
    for index, row in df_filter.iterrows():
        nueva_key=str(row[col1])+' - ' + row[col2]
        filter_list_store_group.append(nueva_key)
        index_storeGroup[nueva_key] = row[col1]
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


def second_approach(input_number,list_seleceted_stores, selected_time,bool_execute):
    if bool_execute:
        if selected_time[0] != selected_time[1]:
            
            df_filter_by_time = data_sw.query(f"{selected_time[0]} <= ISOweek and ISOweek <= {selected_time[1]}").copy()
        else:
            df_filter_by_time = data_sw.copy()
        
        df_filter_storeGroups = df_filter_by_time.query("campaign in @list_seleceted_stores")
        
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
        
        df_sales_filter_by_date_list_StoreGroup = data_sales_stores.query(f"campaign_storeGroup in @list_seleceted_stores and {selected_time[0]} <= ISOweek and ISOweek <= {selected_time[1]}")
        df_sales_filter_by_date_list_StoreGroup["Store Group"] = df_sales_filter_by_date_list_StoreGroup["id_storeGroup"].astype(str) + " - " + df_sales_filter_by_date_list_StoreGroup["name_storeGroup"]
        storeGroupList = df_sales_filter_by_date_list_StoreGroup["Store Group"].unique().tolist()
        storeGroupSelected = st.multiselect("Seleccione un store group que desea excluir en la distribución",storeGroupList)
        df_sales_filter_by_date_list_StoreGroup = df_sales_filter_by_date_list_StoreGroup.query("`Store Group` not in @storeGroupSelected")

        for medio in list_tabla_medio:
            percentage = round(input_number * dict_tablaMedio_sum_avg[medio]["per"])
            medio_iter = medio.split(" ")[0]
            st.markdown(f"<h4>Se recomienda invertir ${percentage} del total en el canal {medio_iter}</h4>",unsafe_allow_html=True)

            data_filter = df_sales_filter_by_date_list_StoreGroup.groupby("Store Group").agg({"id_store_retailer":"nunique","sales":"sum"}).reset_index()
            data_filter["per tiendas"] = data_filter['id_store_retailer'] / data_filter['id_store_retailer'].sum() * 100
            data_filter["per sales"] = data_filter['sales'] / data_filter['sales'].sum() * 100
            data_filter["KPI"] =  data_filter["per sales"] / data_filter["per tiendas"]
            data_filter["share tiendas budget"] = (data_filter["per tiendas"]/100) * percentage
            data_filter["share ventas budget"] = (data_filter["per sales"]/100) * percentage
            data_filter["investment"] = (data_filter["share tiendas budget"] + data_filter["share ventas budget"]) / 2
                
            st.markdown(dataframe_to_markdown(pd.DataFrame(data_filter[["Store Group","investment"]])), unsafe_allow_html=True)
        st.markdown("<div style='height:20px'></div>",unsafe_allow_html=True)
        # pie_graph(pd.DataFrame(dict_of_avg_alpha_coefs_shares.items(), columns=['tabla_medio', 'investment']),"tabla_medio","investment","Distribución de budget por tabla medio")

        dict_medio_per = {}
        for key, value in dict_tablaMedio_sum_avg.items():
            dict_medio_per[key] = value["per"]
        pie_graph(pd.DataFrame(dict_medio_per.items(), columns=['canal', 'investment']),"canal","investment","Distribución del budget total por canal")
        pie_graph(data_filter,"Store Group","investment","Distribución del budget total por Store Group")
        # st.write(dict_tablaMedio_sum_avg)


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

def third_percent():
    list_storeGroups_id = data_sw["store_group_id"].unique().tolist()
    df_accumulator = pd.DataFrame()
    for storeGroup in list_storeGroups_id:
        dict_coeff_storesGroups = coefficient_tabla_medio(storeGroup)
        # print(dict_coeff_storesGroups)
        result_dict = pd.Series(dict_coeff_storesGroups[storeGroup],name=storeGroup)
        df_accumulator = pd.concat([df_accumulator, result_dict.to_frame().T], ignore_index=False)
    
    medio_avg = {}
    for column in df_accumulator.columns:
        serie_column = df_accumulator[column]
        serie_positive = serie_column > 0
        medio_avg[column] = np.mean(serie_column[serie_positive])
    return medio_avg

def share_dict(dict_of_avg_alpha_coefs):
    total_value = sum(dict_of_avg_alpha_coefs.values())
    new_dict = {}
    for key, value in dict_of_avg_alpha_coefs.items():
        new_dict[key] = value / total_value
    return new_dict

def pie_graph(df,columnsx,columnsy,title):
    fig = px.pie(names=df[columnsx], values=df[columnsy])
    st.markdown(f"<h3 style='text-align:center'>{title}</h3>",unsafe_allow_html=True)
    st.plotly_chart(fig)


def new_client(camaping_new_client,investment, window_time):
    df_filter_by_camp = df_sales_storeGroup.query(f"campaign_storeGroup in @camaping_new_client")
    df_filter_by_camp_time = df_filter_by_camp.query(f"{window_time[0]} < ISOweek and ISOweek < {window_time[1]}")
    # Filter the original dataset by the campaign and time window
    
    if "dict_of_avg_alpha_coefs" in st.session_state:
        dict_of_avg_alpha_coefs = st.session_state.dict_of_avg_alpha_coefs
    else:
        st.session_state.dict_of_avg_alpha_coefs = third_percent()
        dict_of_avg_alpha_coefs = st.session_state.dict_of_avg_alpha_coefs
        del dict_of_avg_alpha_coefs['alpha']
    
    dict_of_avg_alpha_coefs_shares = share_dict(dict_of_avg_alpha_coefs)
    
    df_filter_by_camp_time["Store Group"] = df_filter_by_camp_time["id_storeGroup"].astype(str) + " - " + df_filter_by_camp_time["name_storeGroup"]
    df_filter_by_camp_time = df_filter_by_camp_time.groupby("Store Group").agg({"id_store_retailer":"nunique","sales":"sum"}).reset_index()
    df_filter_by_camp_time["per tiendas"] = df_filter_by_camp_time['id_store_retailer'] / df_filter_by_camp_time['id_store_retailer'].sum() * 100
    df_filter_by_camp_time["per sales"] = df_filter_by_camp_time['sales'] / df_filter_by_camp_time['sales'].sum() * 100
    df_filter_by_camp_time["KPI"] =  df_filter_by_camp_time["per sales"] / df_filter_by_camp_time["per tiendas"]
    for key, value in dict_of_avg_alpha_coefs_shares.items():
        key_str = key.replace("coefs_","").replace("_Weekly","")
        value_to_invest = round(value*investment)
        st.markdown(f"<h3>Se recomienda invertir en ${value_to_invest} del total en el canal {key_str}</h3>",unsafe_allow_html=True)
        df_filter_by_camp_time["share tiendas budget"] = (df_filter_by_camp_time["per tiendas"]/100) * value_to_invest
        df_filter_by_camp_time["share ventas budget"] = (df_filter_by_camp_time["per sales"]/100) * value_to_invest 
        df_filter_by_camp_time["investment"] = (df_filter_by_camp_time["share tiendas budget"] + df_filter_by_camp_time["share ventas budget"]) / 2
        st.markdown(dataframe_to_markdown(df_filter_by_camp_time[["Store Group","investment"]]), unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>",unsafe_allow_html=True)
    pie_graph(pd.DataFrame(dict_of_avg_alpha_coefs_shares.items(), columns=['canal', 'investment']),"canal","investment","Distribución del budget total por canal")
    pie_graph(df_filter_by_camp_time,"Store Group","investment","Distribución del budget total por Store Group")

    growing_dict = {"% Crecimiento":[], "Inversión": []}
    total_growing = sum(dict_of_avg_alpha_coefs.values())
    denominator_to_calculate = 0
    for key, value in dict_of_avg_alpha_coefs.items():
        denominator_to_calculate += (value/total_growing) * value
    
    suma_total = np.mean(df_filter_by_camp_time['sales'])
    
    user_input_number = st.number_input("Ingresa un número:", min_value=0.0,value=2.0)
    calculate_of_growing = round(((user_input_number/(100*52))*suma_total) /denominator_to_calculate)

    st.markdown("<h3>Simulación de crecimiento</h3>",unsafe_allow_html=True)
    st.markdown(f"Si el crecimiento deseado es del <b>{user_input_number}</b> % anual, la inversión semanal en medios sugerida debería ser: ${calculate_of_growing}",unsafe_allow_html=True)


def main():

    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        type_of_client = st.selectbox("Tipo de cliente:", ["Cliente Nuevo","Cliente con Historial"])

        if type_of_client == "Cliente Nuevo":
            camaping_new_client = st.selectbox("Filtre por nombre de campaña:", list_campaing_store_group_new_client)
            serie_ISOweek = df_sales_storeGroup.query("campaign_storeGroup in @camaping_new_client")["ISOweek"]
            min_date, max_date = min(serie_ISOweek), max(serie_ISOweek)
            investment_in_media = st.number_input("Inversión en medios:",min_value=0)
            window_time = st.slider(
                "Seleccione la ventana temporal de referencia para el cálculo de distribución",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )
        else:
            selected_option = st.selectbox("Filtre por nombre de campaña:", list_campaing_store_group)
            investment_in_media = st.number_input("Inversión en medios:",min_value=0)
            serie_ISOweek = data_sw.query("campaign in @selected_option")["ISOweek"]                
            min_date, max_date = min(serie_ISOweek), max(serie_ISOweek)
            selected_time = st.slider(
                "Seleccione la ventana temporal de referencia para el cálculo de distribución",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )


    if type_of_client == "Cliente Nuevo":
        if investment_in_media > 0:
            new_client(camaping_new_client,investment_in_media, window_time)
        else:
            st.write("Debe ingresar un monto a invertir")
    else:
        if investment_in_media > 0:
            
            second_approach(investment_in_media,selected_option,selected_time,True)
        else:
            st.write("Debe ingresar un monto a invertir")
