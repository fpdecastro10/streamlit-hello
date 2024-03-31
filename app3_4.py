
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from app1 import dataframe_to_markdown
from app3 import main as app3_main
from app4 import main as app4_main

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

        options = ["Stores con tendencia negativa","Stores con tendencia positiva"]

        app_selection = st.selectbox("Seleccione la tendencia que desea analizar:",options)

    if app_selection == "Stores con tendencia negativa":
        app3_main()
    else:
        app4_main()


