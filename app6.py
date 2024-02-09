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

data_sw = pd.read_csv("datasetCampignSalesNew.csv")
data_sw["concat_store_group_name"] = data_sw["store_group_id"].astype(str) + " - " + data_sw["name"]

def df_builder_tablaMedio(df_tablaMedio_ISOweek):
    pivot_table_tablaMdio_cost = pd.pivot_table(df_tablaMedio_ISOweek,values='cost_campaign',index='ISOweek',columns="tabla_medio")
    isoWeek_sales_origin = df_tablaMedio_ISOweek[["ISOweek","sales"]].groupby("ISOweek").mean().reset_index()
    union_sales_tablaMedio_cost = pd.merge(pivot_table_tablaMdio_cost, isoWeek_sales_origin, on='ISOweek', how='left')
    return union_sales_tablaMedio_cost


def graph_timeserie(df_store_group,name_store):
    filter_datasw = df_store_group
    filter_datasw = filter_datasw.sort_values(by="ISOweek")

    filter_datasw['yearweek'] = filter_datasw['yearweek'].astype(str)
    filter_datasw['ISOweek'] = filter_datasw['ISOweek'].astype(str)

    filter_datasw['tabla_medio'] = filter_datasw['tabla_medio'].fillna('No Campaign')
    filter_datasw['cost_campaign'] = filter_datasw['cost_campaign'].fillna(0)

    filter_datasw= filter_datasw[filter_datasw["yearweek"].notna()]
    
    df_tablaMedio_ISOweek = filter_datasw.groupby([
            "store_group_id",
            "tabla_medio",
            "ISOweek",
            "yearweek"
    ]).agg({
            'cost_campaign': 'sum',
            'sales': 'mean' }).reset_index()
    

    unique_tablaMedio = df_tablaMedio_ISOweek["tabla_medio"].unique().tolist()
    table_pivoted = df_builder_tablaMedio(df_tablaMedio_ISOweek)

    table_pivoted = table_pivoted.sort_values(by="ISOweek")

    table_pivoted['ISOweek'] = table_pivoted['ISOweek'].astype(str)

    table_pivoted['costo total'] = table_pivoted[unique_tablaMedio].sum(axis=1)
    
    fig, ax = plt.subplots()
    for tabla_medio in unique_tablaMedio:
        ax.plot(table_pivoted['ISOweek'], table_pivoted[tabla_medio], label=tabla_medio)
    
    ax2 = ax.twinx()
    
    ax2.plot(table_pivoted['ISOweek'], table_pivoted['sales'], label='sales',color='skyblue')

    ISOweek_negative_tendecy = list(df_tablaMedio_ISOweek["ISOweek"].unique())
    num_ticks = 5
    etiquetas_personalizadas = ISOweek_negative_tendecy[::len(ISOweek_negative_tendecy) // (num_ticks - 1)]

    xticks_positions = []
    for position in etiquetas_personalizadas:
        xticks_positions.append(ISOweek_negative_tendecy.index(position))
    
    plt.xticks(xticks_positions, etiquetas_personalizadas)

    plt.xlabel('week')
    plt.ylabel('sales')
    st.markdown("<h3 style='text-align:center'>Tendecias de ventas</h3>",unsafe_allow_html=True)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(plt)

    table_pivoted = table_pivoted.drop("No Campaign",axis=1)
    unique_tablaMedio.remove("No Campaign")
    if unique_tablaMedio != []:
        table_pivoted = table_pivoted.fillna(0)
        
        unique_tablaMedio_sales = unique_tablaMedio + ["sales"]
        
        X = table_pivoted[unique_tablaMedio]
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
            
            dict_result = {}
            
            if not df_copia.empty:                

                reg = LinearRegression().fit(df_copia, y)
                alpha = reg.intercept_
                coefs = reg.coef_

                coeff_name = df_copia.columns.tolist()
                dict_coeff = {}
                for coeff in range(len(df_copia.columns)):
                    dict_coeff[f"coeff_{coeff_name[coeff]}".replace(" ","_")] = coefs[coeff]/alpha
                dict_result[name_store] = dict_coeff

    else:
        st.write("No hay datos de campaña para este store group")


    # We will try to build the code to generate the coefficients

    
def main():

    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        campaing_store_group = data_sw['campaign'].unique().tolist()

        camaping_new_client = st.selectbox("Filtre por nombre de campaña:", campaing_store_group)

        list_store_group_campaign = data_sw.query("campaign in @camaping_new_client")["concat_store_group_name"].unique().tolist()
        
        camaping_new_client = st.selectbox("Seleccione un store group que desea incluir en la distribución", list_store_group_campaign)
        
        dataset_after_filter_sorted = data_sw.query("concat_store_group_name in @camaping_new_client").sort_values(by="ISOweek")

        min_value_calculated=min(dataset_after_filter_sorted['ISOweek'])
        max_value_calculated=max(dataset_after_filter_sorted['ISOweek'])
        
        start_date, end_date = st.select_slider(
            "Seleccione la ventana temporal de referencia para el cálculo de crecimiento",
            options=dataset_after_filter_sorted["ISOweek"],
            value=(min_value_calculated, max_value_calculated)
        )

        dataset_to_graph = dataset_after_filter_sorted.query(f"{start_date} < ISOweek and ISOweek < {end_date}")
    
    graph_timeserie(dataset_to_graph,dataset_after_filter_sorted)