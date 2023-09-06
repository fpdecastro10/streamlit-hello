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
unique_combinations = data_sw[["id_storeGroup","name_storeGroup"]].drop_duplicates(["id_storeGroup","name_storeGroup"])

def dataframe_to_markdown_str(df):
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"

    # for index, row in df.iterrows():
    #     markdown += "| " + " | ".join(str(int(row[column])) for column in df.columns) + " |\n"
    for index, row in df.iterrows():
        for column in df.columns:
            try:
                markdown += '|' + str(int(row[column]))
        # markdown += "| " + " | ".join(str(row[column]) for column in df.columns) + " |\n"
            except:
                markdown += '|' + str(row[column])
        markdown += '|\n'
    
    return markdown

def dataframe_to_markdown(df):
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"

    for index, row in df.iterrows():
        for column in df.columns:
            try:
                markdown += '|' + str(int(row[column]))
        # markdown += "| " + " | ".join(str(row[column]) for column in df.columns) + " |\n"
            except:
                markdown += '|' + str(row[column])
        markdown += '|\n'
    return markdown


def main():
    with st.sidebar:
        imagen_local='./logo2x.png'
        st.image(imagen_local, use_column_width=True)
        st.markdown('<h1 style="font-size: 34px;">Filtros </h1>', unsafe_allow_html=True)

        #  built a index dictionary to show as a filter
        index_storeGroup = {}
        for index, row in unique_combinations.iterrows():
            nueva_key=str(row["id_storeGroup"])+' - ' + row['name_storeGroup']
            index_storeGroup[nueva_key] = row["id_storeGroup"]
        temp_index_storeGroup = dict(index_storeGroup)
        
        for key, value in temp_index_storeGroup.items():
            if data_sw.query(f"id_storeGroup == {value}").groupby("yearweek_campaign").sum().reset_index().shape[0] <= 8:
                index_storeGroup.pop(key)

        botones = [key for key in index_storeGroup]
        
        
        selected_filter = st.selectbox("Seleccione un storegroup:", botones)

        opciones = ['Semanal', 'Mes']
        filtros_seleccionados = st.radio('Seleccione la granularidad de tiempo:', opciones)

        numero_ingresado = st.number_input("Ingrese el monto de campaña a invertir", value=0.0, step=0.1)
        
        filter_data_storeGroup = data_sw.query(f"id_storeGroup == {index_storeGroup[selected_filter]}")    
        min_value_calculated=min(filter_data_storeGroup['yearweek_campaign'])
        max_value_calculated=max(filter_data_storeGroup['yearweek_campaign'])
        selected_time = st.slider("Seleccione la ventana temporal de referencia para el cálculo de crecimiento", min_value=min_value_calculated, max_value=max_value_calculated, value=(min_value_calculated, max_value_calculated))

    # Configuración de la aplicación
    st.markdown('<h1 style="text-align: center;">Costo de campaña vs Sales por Store group</h1>', unsafe_allow_html=True)
    # Filtrar los datos según el botón seleccionado
    filter_data_storeGroup = data_sw.query(f"id_storeGroup == {index_storeGroup[selected_filter]}") 
        
    filtered_data = filter_data_storeGroup.groupby("yearweek_campaign").sum().reset_index()
    promedio_ventas = np.mean(filtered_data.query(f"{selected_time[0]} < yearweek_campaign and yearweek_campaign < {selected_time[1]} ")['sales'])

    # Constuimos el data frame con sku_id
    filtered_data_product = filter_data_storeGroup.groupby(["id_sku","name_product"]).sum().reset_index()[['id_sku', 'name_product','sales']]
    total_sales = filtered_data_product['sales'].sum()
    filtered_data_product['share'] = filtered_data_product['sales']/total_sales
    
    # 81
    filtered_data_product_store = filter_data_storeGroup.groupby(['id_sku','id_store_retailer','name_retailer']).sum().reset_index()[['id_sku', 'id_store_retailer','name_retailer', 'sales']]
    filtered_data_product_store['share'] = filtered_data_product_store['sales']/total_sales

    d = np.polyfit(filtered_data['cost_campaign_dist'],filtered_data['sales'],2)
    f = np.poly1d(d)
    filtered_data.insert(1,'Treg',f(filtered_data['cost_campaign_dist']))

    plt.figure(figsize=(8, 6))
    ax=filtered_data.plot.scatter(x='cost_campaign_dist', y='sales', color='yellow')
    filtered_data.plot.scatter(x='cost_campaign_dist',y='Treg',color='red',legend=False,ax=ax)

    correlation_matrix = np.corrcoef(filtered_data["Treg"], filtered_data["sales"])
    correlation = correlation_matrix[0, 1]
    st.write(f"Correlación: {round(correlation,4)}")

    # Crear el gráfico interactivo
    plt.xlabel('costo de campaña')
    plt.ylabel('sales')
    plt.title(f'Store Group Id: {selected_filter}')
    st.pyplot(plt)

    if numero_ingresado == 0:
        st.markdown(f"## Coloque un monto a calcular")
    else:
        if filtros_seleccionados == 'Mes':
            ventas_totales = f(numero_ingresado/4) * 4
        else:
            ventas_totales=f(numero_ingresado)

        st.markdown(f'''
            <h2 style=color:#f7dc00> Proyección de ventas:
                <p style="color:#ffffff;font-size:2rem;margin-top:10px"><b>{round(ventas_totales)}</b> un {filtros_seleccionados}
                </p>
            </h2>''',unsafe_allow_html=True )
        st.markdown(f'''
            <h2 style=color:#f7dc00> Crecimiento esperado:
                <p style="color:#ffffff;font-size:2rem;margin-top:10px"><b>{round(((ventas_totales-promedio_ventas)/promedio_ventas)*100,1)}%</b> vs venta promedio {filtros_seleccionados} de <b>{round(promedio_ventas)}</b> un (período del {selected_time[0]} al {selected_time[1]})
                </p>
            </h2>''',unsafe_allow_html=True)
        filtered_data_product['share_sales'] = round(filtered_data_product['share']*ventas_totales)
        filtered_data_product_store['share_sales'] = round(filtered_data_product_store['share']*ventas_totales)
        
        filtered_data_product.rename(columns={'share_sales': 'Qty sales (un)'}, inplace=True)
        filtered_data_product.rename(columns={'id_sku': 'Sku id'}, inplace=True)
        filtered_data_product.rename(columns={'name_product': 'Producto'}, inplace=True)
        st.markdown(f"<h2 style=color:#f7dc00> Detalle por producto: </h2>",unsafe_allow_html=True )

        st.markdown(dataframe_to_markdown_str(filtered_data_product[['Sku id','Producto','Qty sales (un)']]))
        filtered_data_product_store.rename(columns={'id_sku': 'Sku id'}, inplace=True)
        filtered_data_product_store.rename(columns={'id_store_retailer': 'Store id'}, inplace=True)
        filtered_data_product_store.rename(columns={'name_retailer': 'Retailer'}, inplace=True)
        filtered_data_product_store.rename(columns={'share_sales': 'Qty sales (un)'}, inplace=True)
        st.markdown('<div style="height: 10px;"></div>',unsafe_allow_html=True)
        st.markdown(f"<h2 style=color:#f7dc00> Detalle por producto y store: </h2>",unsafe_allow_html=True )
        st.markdown(dataframe_to_markdown(filtered_data_product_store[['Sku id','Store id','Retailer','Qty sales (un)']].sort_values(by='Qty sales (un)', ascending=False)))

if __name__ == "__main__":
    main()
