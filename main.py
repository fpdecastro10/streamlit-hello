import streamlit as st

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color:#0e1118;
}
<\style>
"""
def main():
    st.markdown(page_bg_img,unsafe_allow_html=True)
    st.title("Que aplicación desea ejecutar")
    app_selection = st.selectbox("",["Predicción ventas","Asignación de costos"])

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
