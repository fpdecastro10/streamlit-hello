import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

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
        numero_ingresado = st.number_input("Ingrese el monto de campaña a invertir", value=0.0, step=0.1)
        if uploaded_file is not None:
            st.write("Archivo cargado con éxito!")
            dataventasmodelo = pd.read_csv(uploaded_file)
            datasetStoreSku = data_sw[['id_sku','id_store_retailer','sales']].groupby(['id_sku','id_store_retailer']).mean().reset_index()
            datasetStoreSkuMean = datasetStoreSku.groupby('id_sku').mean().reset_index()
            datasetStoreSkuMean['concat'] = datasetStoreSkuMean['id_sku'].astype('str') +' - ('+ round(datasetStoreSkuMean['sales'],3).astype('str') +' avg)'
            nameFilterValue = datasetStoreSkuMean.set_index('concat')['id_sku'].to_dict()
            
            min_value_calculated = min(dataventasmodelo['ISOweek'])
            max_value_calculated = max(dataventasmodelo['ISOweek'])
            selected_time = st.slider("Seleccione la ventana de semanas con que quiere comparar", min_value=min_value_calculated, max_value=max_value_calculated, value=(min_value_calculated, max_value_calculated))
            
            opcion = st.multiselect('Seleccione los productos que quiere que participen de la regresion de stores',list(datasetStoreSkuMean['concat']))
            listToFilterSku = []
            for element in opcion:
                listToFilterSku.append(nameFilterValue[element])
        
    if uploaded_file is not None:

        dataVentasWeek = dataventasmodelo.groupby('ISOweek').sum().reset_index()
        storeSkuid = dataventasmodelo.groupby(['sku_id','store_id']).mean().reset_index()
        skuidMeanSales = storeSkuid.groupby('sku_id').mean().reset_index()
        if skuidMeanSales.shape[0] == 1:
            st.markdown(f"<p style=color:#ffffff>Promedio de ventas por stores del producto nuevo es: </p>",unsafe_allow_html=True )
            st.dataframe(skuidMeanSales[['sku_id', 'sales']])
        else:
            st.markdown(f"<p style=color:#ffffff>Promedio de ventas de los productos nuevos por stores es: </p>",unsafe_allow_html=True )
            st.dataframe(skuidMeanSales[['sku_id', 'sales']])
        
        if numero_ingresado != 0 and uploaded_file is not None:
            totalStoresFile = set(dataventasmodelo['id_store_id'])
            cantidadDeproductos = len(dataventasmodelo['sku_id'].unique())
            numberTotalStoresFile = len(totalStoresFile)
            asiganacionCostoDeCampana = (numero_ingresado / numberTotalStoresFile) / cantidadDeproductos
            totalStoresModelo = set(data_sw.query('id_sku in @listToFilterSku')['id_store_retailer'])
            totalDeStoresEnComun = totalStoresFile.intersection(totalStoresModelo)
            numberTotalDeStoresEnComun=len(totalDeStoresEnComun)

            amountOfStoresInCommon = round(numberTotalDeStoresEnComun/numberTotalStoresFile,2)
            st.markdown(f"Se encontro información para el {amountOfStoresInCommon} % de los stores",unsafe_allow_html=True)
            accumulatorSales = 0
            if amountOfStoresInCommon > 0.5:
                for i in totalDeStoresEnComun:
                    datasetFiltradoProductoNuevo = data_sw.query(f'id_store_retailer == {i} and id_sku in @listToFilterSku')[['cost_campaign_dist','sales']]
                    datasetFiltradoProductoNuevoX= np.array(datasetFiltradoProductoNuevo['cost_campaign_dist'])
                    datasetFiltradoProductoNuevoY= np.array(datasetFiltradoProductoNuevo['sales'])
                    regression = LinearRegression()                                      
                    X = datasetFiltradoProductoNuevoX.reshape(-1, 1)  # Asegúrate de que X sea una matriz 2D (una característica)
                    regression.fit(X, datasetFiltradoProductoNuevoY)

                    accumulatorSales += round(regression.predict(np.array(asiganacionCostoDeCampana).reshape(-1,1))[0])

            st.markdown(f"<p> La predicción de ventas es {accumulatorSales}<p>",unsafe_allow_html=True)
            
            print(asiganacionCostoDeCampana)

        
        plt.figure(figsize=(8, 6))
        dataVentasWeek.plot.scatter(x='ISOweek', y='sales', color='yellow')
        plt.xlabel('ISOweek')
        plt.ylabel('sales')
        plt.title(f'Gráfico filtrado:')
        st.pyplot(plt)
    



if __name__ == "__main__":
    main()
