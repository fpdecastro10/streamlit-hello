import streamlit as st

def main():
    st.title("Que aplicación desea ejecutar")
    app_selection = st.selectbox("",["Productos nuevos","Producto con historial de asignación"])

    if app_selection == "Producto con historial de asignación":
        # Ejecutar la primera aplicación
        from app1 import main as app1_main
        app1_main()
    else:
        # Ejecutar la segunda aplicación
        from app2 import main as app2_main
        app2_main()

if __name__ == "__main__":
    main()
