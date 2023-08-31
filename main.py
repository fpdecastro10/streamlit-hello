import streamlit as st

def main():

    st.markdown("<h2>Seleccione el tipo de predicción que desea realizar<h2>",unsafe_allow_html=True)
    app_selection = st.selectbox("",["Asignación de costos","Predicción ventas"])

    if app_selection == "Asignación de costos":
        # Ejecutar la primera aplicación
        from app1 import main as app1_main
        app1_main()
    else:
        # Ejecutar la segunda aplicación
        from app2 import main as app2_main
        app2_main()

if __name__ == "__main__":
    main()
