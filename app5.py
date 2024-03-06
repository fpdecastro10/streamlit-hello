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
data_sw["concat_store_group_name"] = data_sw["store_group_id"].astype(str) + " - " + data_sw["name_storeGroup"]

data_sw_1 = pd.read_csv("datasetCampignSalesNew.csv")
data_sw_1["concat_store_group_name"] = data_sw_1["store_group_id"].astype(str) + " - " + data_sw_1["name"]


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

def df_builder_tablaMedio(df_tablaMedio_ISOweek):
    pivot_table_tablaMdio_cost = pd.pivot_table(df_tablaMedio_ISOweek,values='cost',index='ISOweek',columns="tabla_medio")
    isoWeek_sales_origin = df_tablaMedio_ISOweek[["ISOweek","sales"]].groupby("ISOweek").mean().reset_index()
    union_sales_tablaMedio_cost = pd.merge(pivot_table_tablaMdio_cost, isoWeek_sales_origin, on='ISOweek', how='left')
    return union_sales_tablaMedio_cost

def df_builder_tablaMedio_dict(df_tablaMedio_ISOweek):
    pivot_table_tablaMdio_cost = pd.pivot_table(df_tablaMedio_ISOweek,values='cost_campaign',index='ISOweek',columns="tabla_medio")
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
            dict_tablaMedio_sum_avg[key]["per"] = 1 - dict_tablaMedio_sum_avg[key]["sum"]/total
        
        df_sales_filter_by_date_list_StoreGroup = data_sales_stores.query(f"campaign_storeGroup in @list_seleceted_stores and {selected_time[0]} <= ISOweek and ISOweek <= {selected_time[1]}")
        df_sales_filter_by_date_list_StoreGroup["Store Group"] = df_sales_filter_by_date_list_StoreGroup["id_storeGroup"].astype(str) + " - " + df_sales_filter_by_date_list_StoreGroup["name_storeGroup"]
        storeGroupList = df_sales_filter_by_date_list_StoreGroup["Store Group"].unique().tolist()
        storeGroupSelected = st.multiselect("Seleccione los store groups que desea incluir en la regresión que estima la distribución",storeGroupList)
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

def third_percent_dict(selectedGroup):
    list_storeGroups_id = data_sw_1.query("concat_store_group_name in @selectedGroup")["concat_store_group_name"].unique().tolist()
    dict_result = {}
    count = 0
    for storeGroup in list_storeGroups_id:
        filter_datasw = data_sw_1[data_sw_1["concat_store_group_name"] == storeGroup]
        filter_datasw['tabla_medio'] = filter_datasw['tabla_medio'].fillna('No Campaign')
        filter_datasw['cost_campaign'] = filter_datasw['cost_campaign'].fillna(0)
        filter_datasw["yearweek"] = filter_datasw["yearweek"].fillna("-")
        df_tablaMedio_ISOweek = filter_datasw.groupby([
                    "store_group_id",
                    "tabla_medio",
                    "ISOweek",
                    "yearweek"
            ]).agg({
                    'cost_campaign': 'sum',
                    'sales': 'mean' }).reset_index()
        
        table_pivoted = df_builder_tablaMedio_dict(df_tablaMedio_ISOweek)
        list_tabla_medio = df_tablaMedio_ISOweek["tabla_medio"].unique().tolist()
        list_tabla_medio.remove("No Campaign")
        table_pivoted = table_pivoted.drop("No Campaign",axis=1)
        table_pivoted[list_tabla_medio] = table_pivoted[list_tabla_medio].fillna(0)

        if list_tabla_medio == []:
            continue

        X = table_pivoted[list_tabla_medio]
        y = table_pivoted["sales"]

        if X.shape[0] > 20:
            reg = LinearRegression().fit(X, y)
            alpha = reg.intercept_
            coefs = reg.coef_

            dict_pvalues = dict(sm.OLS(y, X).fit().pvalues)
            df_copia = X.copy()
            for key, value in dict_pvalues.items():
                if value > 0.05:
                    df_copia = df_copia.drop(key, axis=1)
                        
            if not df_copia.empty:                

                reg = LinearRegression().fit(df_copia, y)
                alpha = reg.intercept_
                coefs = reg.coef_

                coeff_name = df_copia.columns.tolist()
                dict_coeff = {}
                for coeff in range(len(df_copia.columns)):
                    dict_coeff[f"coeff_{coeff_name[coeff]}".replace(" ","_")] = coefs[coeff]/alpha
                dict_result[storeGroup] = dict_coeff
                count += 1
    return dict_result


def third_percent(selectedGroup):
    list_storeGroups_id = data_sw.query("concat_store_group_name in @selectedGroup")["concat_store_group_name"].unique().tolist()
    df_accumulator = pd.DataFrame()
    dict_result = {}

    for store in list_storeGroups_id:
        # df_storeGroup_id = data_sw.query(f"concat_store_group_name in @list_storeGroup  & sales > 0")
        df_storeGroup_id = data_sw[data_sw["concat_store_group_name"] == store].query("sales > 0")

        z = np.abs(zscore(df_storeGroup_id['cpc'])) 
        
        series_outliers = pd.Series(z<2.5)
        df_storeGroup_id = df_storeGroup_id[series_outliers]
        # display(df_storeGroup_id)

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

        X = df_tablaMedio_listCost[unique_tablaMedio].copy()
        

        # Mostrar el gráfico

        if X.shape[0] > 20:
            y = df_tablaMedio_listCost["sales"]
            reg = LinearRegression().fit(X, y)
            alpha = reg.intercept_
            coefs = reg.coef_

            dict_pvalues = dict(sm.OLS(y, X).fit().pvalues)
            
            df_copia = X.copy()
            for key, value in dict_pvalues.items():
                if value > 0.05:
                    df_copia = df_copia.drop(key, axis=1)
            

            if df_copia.empty:
                continue

            reg = LinearRegression().fit(df_copia, y)
            alpha = reg.intercept_
            coefs = reg.coef_

            coeff_name = df_copia.columns.tolist()
            dict_coeff = {}
            for coeff in range(len(df_copia.columns)):
                dict_coeff[f"coeff_{coeff_name[coeff]}".replace(" ","_")] = coefs[coeff]/alpha
            dict_result[store] = dict_coeff
    
    return dict_result

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


def new_client(camaping_new_client,investment, window_time,selectedStoreGroups):
    df_filter_by_camp = df_sales_storeGroup.query(f"campaign_storeGroup in @camaping_new_client")
    df_filter_by_camp_time = df_filter_by_camp.query(f"{window_time[0]} < ISOweek and ISOweek < {window_time[1]}")
    # Filter the original dataset by the campaign and time window
    
    # dict_of_avg_alpha_coefs = third_percent(selectedStoreGroups)
    # dict_of_avg_alpha_coefs = third_percent_dict(selectedStoreGroups)
    list_storeGroups_id = data_sw.query("concat_store_group_name in @selectedStoreGroups")

    tabla_medio = list_storeGroups_id['tabla_medio'].unique().tolist()
    dict_of_avg_alpha_coefs_shares_bis = {}
    for medio in tabla_medio:
        dict_of_avg_alpha_coefs_shares_bis[medio] = np.sum(list_storeGroups_id.query(f"tabla_medio in @medio")['cost'])

    dict_of_avg_alpha_coefs_shares_bis = share_dict(dict_of_avg_alpha_coefs_shares_bis)



    # if dict_of_avg_alpha_coefs == {}:
    #     st.write("Los stores groups seleccionados, no tienen suficientes datos para realizar la predicción")
    # else:

        # dict_coeff_list = {}
        # for key, items in dict_of_avg_alpha_coefs.items():
        #     for key, items in items.items():
        #         if key in dict_coeff_list:
        #             dict_coeff_list[key].append(items)
        #         else:
        #             dict_coeff_list[key] = [items]
        
        # for key, items in dict_coeff_list.items():
        #     dict_coeff_list[key] = np.mean(items)

        # dict_of_avg_alpha_coefs_shares = share_dict(dict_coeff_list)

        # st.write(dict_of_avg_alpha_coefs_shares)
        
    df_filter_by_camp_time["Store Group"] = df_filter_by_camp_time["id_storeGroup"].astype(str) + " - " + df_filter_by_camp_time["name_storeGroup"]
    df_filter_by_camp_time = df_filter_by_camp_time.groupby("Store Group").agg({"id_store_retailer":"nunique","sales":"sum"}).reset_index()
    df_filter_by_camp_time["per tiendas"] = df_filter_by_camp_time['id_store_retailer'] / df_filter_by_camp_time['id_store_retailer'].sum() * 100
    df_filter_by_camp_time["per sales"] = df_filter_by_camp_time['sales'] / df_filter_by_camp_time['sales'].sum() * 100
    df_filter_by_camp_time["KPI"] =  df_filter_by_camp_time["per sales"] / df_filter_by_camp_time["per tiendas"]

    for key, value in dict_of_avg_alpha_coefs_shares_bis.items():
        key_str = key.replace("coefs_","").replace("_Weekly","")
        value_to_invest = round(value*investment)
        st.markdown(f"<h3>Se recomienda invertir en ${value_to_invest} del total en el canal {key_str}</h3>",unsafe_allow_html=True)
        df_filter_by_camp_time["share tiendas budget"] = (df_filter_by_camp_time["per tiendas"]/100) * value_to_invest
        df_filter_by_camp_time["share ventas budget"] = (df_filter_by_camp_time["per sales"]/100) * value_to_invest 
        df_filter_by_camp_time["investment"] = (df_filter_by_camp_time["share tiendas budget"] + df_filter_by_camp_time["share ventas budget"]) / 2
        st.markdown(dataframe_to_markdown(df_filter_by_camp_time[["Store Group","investment"]]), unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>",unsafe_allow_html=True)
    st.write(dict_of_avg_alpha_coefs_shares_bis)
    pie_graph(pd.DataFrame(dict_of_avg_alpha_coefs_shares_bis.items(), columns=['canal', 'investment']),"canal","investment","Distribución del budget total por canal")
    pie_graph(df_filter_by_camp_time,"Store Group","investment","Distribución del budget total por Store Group")

        # growing_dict = {"% Crecimiento":[], "Inversión": []}
        # total_growing = sum(dict_coeff_list.values())

        # denominator_to_calculate = 0
        # for key, value in dict_coeff_list.items():
        #     denominator_to_calculate += (value/total_growing) * value

        # user_input_number = st.number_input("Ingresa un número:", min_value=0.0,value=2.0)
        # calculate_of_growing = round(user_input_number/denominator_to_calculate)

        # st.markdown("<h3>Simulación de crecimiento</h3>",unsafe_allow_html=True)
        # st.markdown(f"Para un aumento del <b>{user_input_number}</b> % semanal con respecto al promedio de ventas historico, la inversión en medios sugerida debería ser: ${calculate_of_growing} semanal",unsafe_allow_html=True)


def main():

    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        # selection_radio = st.radio("Seleccion con cuantas semanas de delay quiere hacer la prediccion",["1 semana","2 semanas"])

        type_of_client = st.selectbox("Tipo de cliente:", ["Cliente Nuevo","Cliente con Historial"])

        if type_of_client == "Cliente Nuevo":
            camaping_new_client = st.selectbox("Filtre por nombre de campaña:", list_campaing_store_group_new_client)
            serie_ISOweek = df_sales_storeGroup.query("campaign_storeGroup in @camaping_new_client")["ISOweek"]
            min_date = min(serie_ISOweek)
            max_date =  max(serie_ISOweek)
            list_store_group_coeff = ["Seleccionar todos"]
            list_store_group_coeff += data_sw['concat_store_group_name'].unique().tolist() 
            selection_storeGroups = st.multiselect("Seleccione un store group que desea incluir en la distribución",list_store_group_coeff)
            if "Seleccionar todos" in selection_storeGroups:
                selection_storeGroups = data_sw['concat_store_group_name'].unique().tolist()
            investment_in_media = st.number_input("Inversión en medios:",value=1000)
            start_date, end_date = st.select_slider(
                "Seleccione la ventana temporal de referencia para el cálculo de crecimiento",
                options=serie_ISOweek.sort_values(),
                value=[min_date, max_date]
            )
            selected_time = (start_date, end_date)
            
        else:
            selected_option = st.selectbox("Filtre por nombre de campaña:", list_campaing_store_group)
            investment_in_media = st.number_input("Inversión en medios:",min_value=0)
            serie_ISOweek = data_sw.query("campaign in @selected_option")["ISOweek"]                
            min_date, max_date = min(serie_ISOweek), max(serie_ISOweek)
            start_date, end_date = st.select_slider(
                "Seleccione la ventana temporal de referencia para el cálculo de crecimiento",
                options=serie_ISOweek.sort_values(),
                value=[min_date, max_date]
            )
            selected_time = (start_date, end_date)

    if type_of_client == "Cliente Nuevo":
        if investment_in_media > 0 and selection_storeGroups != []:
            new_client(camaping_new_client,investment_in_media, selected_time, selection_storeGroups)
        else:
            st.write("Debe ingresar un monto a invertir")
    else:

        if investment_in_media > 0:
            second_approach(investment_in_media,selected_option,selected_time,True)
        else:
            st.write("Debe ingresar un monto a invertir")