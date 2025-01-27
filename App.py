import streamlit as st # type: ignore

st.set_page_config(page_title="Tierra Sphere Carbon Capture Prediction Application", page_icon=":earth_africa:")

with st.sidebar:
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        st.image("assets/images/tierra-foods-logo.png", width=100)
    with col2:
        st.image("assets/images/uoh-logo.svg", width=100)

st.title("Tierra Sphere Carbon Capture Prediction Application")
