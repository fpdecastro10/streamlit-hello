import streamlit as st

def main():
    st.title("Seleccione el tipo de predicción que desea realizar:")
    # app_selection = st.selectbox("",["Producto con historial de asignación","Productos nuevos","Stores con tendencia negativa","Distribución por campaña"])
    app_selection = st.selectbox("",["1. Inversión inicial y distribución de budget","2. Predicción de sales + analytics seasonality, trend & media","3. Tendencia de Ventas en Stores","4. Predicción de sales - Regresión polinomial"])

    if app_selection == "4. Predicción de sales - Regresión polinomial":
        # Ejecutar la primera aplicación
        from app1 import main as app1_main
        app1_main()
    elif app_selection == "Productos nuevos":
        # Ejecutar la segunda aplicación
        from app2 import main as app2_main
        app2_main()
    elif app_selection == "3. Tendencia de Ventas en Stores":
        from app3_4 import main as app3_4_main
        app3_4_main()
    elif app_selection == "1. Inversión inicial y distribución de budget":
        # from app5_copy import main as app5_main
        from app5 import main as app5_main
        app5_main()
    elif app_selection == "2. Predicción de sales + analytics seasonality, trend & media":
        from app6 import main as app6_main
        app6_main()

if __name__ == "__main__":
    main()