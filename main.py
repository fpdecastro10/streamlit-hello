import streamlit as st

def main():
    st.title("Seleccione el tipo de predicción que desea realizar:")
    app_selection = st.selectbox("",["Producto con historial de asignación","Productos nuevos","Tendencia stores"])

    if app_selection == "Producto con historial de asignación":
        # Ejecutar la primera aplicación
        from app1 import main as app1_main
        app1_main()
    elif app_selection == "Productos nuevos":
        # Ejecutar la segunda aplicación
        from app2 import main as app2_main
        app2_main()
    else:
        from app3 import main as app3_main
        app3_main()

if __name__ == "__main__":
    main()
