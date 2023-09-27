import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def dataframe_to_markdown(df):
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"

    for index, row in df.iterrows():
        for column in df.columns:
            if column == 'sku_id':
                markdown += '|' + str(int(row[column]))
            else:
                markdown += '|' + str(round(row[column],1))
        markdown += '|\n'
    return markdown

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
    # Lógica de la primera aplicación
    with st.sidebar:
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
        if uploaded_file is not None:
            st.write("Archivo cargado con éxito!")
            numero_ingresado = st.number_input("Ingrese el monto de campaña a invertir", value=0.0, step=0.1)
            dataventasmodelo = pd.read_csv(uploaded_file)
            dataventasmodelo = dataventasmodelo.query('sales>0')
            datasetStoreSku = data_sw[['id_sku','name_product','id_store_retailer','sales']].groupby(['id_sku','name_product','id_store_retailer']).mean().reset_index()
            datasetStoreSkuMean = datasetStoreSku.groupby(['id_sku','name_product']).mean().reset_index()
            datasetStoreSkuMean['concat'] = datasetStoreSkuMean['id_sku'].astype('str') + ' - ' + datasetStoreSkuMean['name_product'] + ' - ('+ round(datasetStoreSkuMean['sales'],1).astype('str') +' avg)'
            nameFilterValue = datasetStoreSkuMean.set_index('concat')['id_sku'].to_dict()
            
            min_value_calculated = min(dataventasmodelo['ISOweek'])
            max_value_calculated = max(dataventasmodelo['ISOweek'])
            selected_time = st.slider("Seleccione la ventana temporal de referencia para el cálculo de crecimiento", min_value=min_value_calculated, max_value=max_value_calculated, value=(min_value_calculated, max_value_calculated))
            
            opcion = st.multiselect('Seleccione los productos que desea que participen de la regresión de stores',list(datasetStoreSkuMean['concat']))
            listToFilterSku = []
            for element in opcion:
                listToFilterSku.append(nameFilterValue[element])
        
    if uploaded_file is not None:
        
        dataVentasWeek = dataventasmodelo.groupby('ISOweek').sum().reset_index()
        plt.figure(figsize=(8, 6))
        dataVentasWeek.plot.scatter(x='ISOweek', y='sales', color='yellow')
        plt.xlabel('ISOweek')
        plt.ylabel('sales')
        plt.title(f'Sales por semanas del dataset cargado')
        st.pyplot(plt)
        totalStoresFile = set(dataventasmodelo['id_store_id'])
        cantidadDeproductos = len(dataventasmodelo['sku_id'].unique())
        numberTotalStoresFile = len(totalStoresFile)
        asiganacionCostoDeCampana = (numero_ingresado / numberTotalStoresFile) / cantidadDeproductos
        totalStoresModelo = set(data_sw.query('id_sku in @listToFilterSku')['id_store_retailer'])
        totalDeStoresEnComun = totalStoresFile.intersection(totalStoresModelo)
        totalDeStoresEnComunlist = list(totalDeStoresEnComun)
        numberTotalDeStoresEnComun=len(totalDeStoresEnComun)
        
        storeSkuidInCommon = dataventasmodelo.query("id_store_id in @totalDeStoresEnComunlist")[['sku_id','store_id','sales']].groupby(['sku_id','store_id']).mean().reset_index()
        skuidMeanSalesInCommon = storeSkuidInCommon[['sku_id','sales']].groupby('sku_id').mean().reset_index()
        skuidMeanSalesInCommon.rename(columns={'sku_id': 'Sku id'}, inplace=True)
        skuidMeanSalesInCommon.rename(columns={'sales': 'Avg weekly sales per store (un)'}, inplace=True)

        storeSkuid = dataventasmodelo[['sku_id','store_id','sales']].groupby(['sku_id','store_id']).mean().reset_index()
        skuidMeanSales = storeSkuid[['sku_id','sales']].groupby('sku_id').mean().reset_index()
        
        skuidMeanSales.rename(columns={'sku_id': 'Sku id'}, inplace=True)
        skuidMeanSales.rename(columns={'sales': 'Avg weekly sales per store (un)'}, inplace=True)
        
        if skuidMeanSales.shape[0] == 1:
            st.markdown(f'''
            <h2 style=color:#f7dc00> Promedio de ventas semanales por store: </h2>''',unsafe_allow_html=True)
            st.markdown(dataframe_to_markdown(skuidMeanSales[['Sku id', 'Avg weekly sales per store (un)']]))
        else:
            st.markdown(f"<p style=color:#ffffff>Promedio de ventas de los productos nuevos por stores es: </p>",unsafe_allow_html=True )
            st.dataframe(skuidMeanSales[['Sku id', 'Avg weekly sales per store (un)']])
        
        if not skuidMeanSalesInCommon.empty:
            if skuidMeanSalesInCommon.shape[0] == 1:
                st.markdown(f'''
                <h2 style=color:#f7dc00> Promedio de ventas semanales por store en comun: </h2>''',unsafe_allow_html=True)
                st.markdown(dataframe_to_markdown(skuidMeanSalesInCommon[['Sku id', 'Avg weekly sales per store (un)']]))
            else:
                st.markdown(f"<p style=color:#ffffff>Promedio de ventas de los productos nuevos por stores es: </p>",unsafe_allow_html=True )
                st.dataframe(skuidMeanSalesInCommon[['Sku id', 'Avg weekly sales per store (un)']])


        amountOfStoresInCommon = round(numberTotalDeStoresEnComun/numberTotalStoresFile,2)
        st.markdown('<div style="height: 10px;"></div>',unsafe_allow_html=True)
        st.markdown(f"Se encontro información para el {round(amountOfStoresInCommon*100)} % de los stores",unsafe_allow_html=True)
        
        promedio_ventas = np.mean(dataVentasWeek.query(f"{selected_time[0]} < ISOweek and ISOweek < {selected_time[1]} ")['sales'])
        st.markdown(f'''
            <h2 style=color:#f7dc00>Promedio de ventas semanal:
                <p style="color:#ffffff;font-size:2rem;margin-top:10px"><b>{round(promedio_ventas,1)} un</b>
                </p>
            </h2>''',unsafe_allow_html=True )
        
         accumulatorSales = 0
        intercept = 0
        coeficient = 0
        for i in totalDeStoresEnComun:
            datasetFiltradoProductoNuevo = data_sw.query(f'id_store_retailer == {i} and id_sku in @listToFilterSku')[['cost_campaign_dist','sales']]
            if datasetFiltradoProductoNuevo.shape[0] > 10:
                d = np.polyfit(datasetFiltradoProductoNuevo['cost_campaign_dist'],datasetFiltradoProductoNuevo['sales'],1)
                f = np.poly1d(d)
                intercept += f[0]
                coeficient += f[1]
                # valuePrediction = round(regression.predict(np.array(asiganacionCostoDeCampana).reshape(-1,1))[0])
                valuePrediction = f(asiganacionCostoDeCampana)
                # print(i, valuePrediction, np.mean(dataventasmodelo.groupby('id_store_id').mean().reset_index().query(f"id_store_id=={i}")['sales']))
                accumulatorSales += valuePrediction
        
        if coeficient<0:
            coeficient= coeficient*-1
        accumulatorSales = coeficient * asiganacionCostoDeCampana + intercept
       
        
        st.markdown(f'''
        <h2 style=color:#f7dc00>Proyección de ventas semanales:
            <p style="color:#ffffff;font-size:2rem;margin-top:10px"><b>{accumulatorSales} un</b>
            </p>
        </h2>''',unsafe_allow_html=True )
        st.markdown(f'''
        <h2 style=color:#f7dc00>Crecimiento esperado:
            <p style="color:#ffffff;font-size:2rem;margin-top:10px"><b>{round(((accumulatorSales-promedio_ventas)/promedio_ventas)*100,1)}%</b> vs venta promedio Semanal de <b>{round(promedio_ventas,1)}</b> un (período del {selected_time[0]} al {selected_time[1]})
            </p>
        </h2>''',unsafe_allow_html=True )


if __name__ == "__main__":
    main()
