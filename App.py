import streamlit as st # type: ignore
from utils.sidebar_menu import sidebar

sidebar(layout_style="wide")

st.title("Tierrasphere Carbon Capture Prediction Application")

st.subheader("Contributors")

st.write("This application is developed by the [University of Huddersfield](https://www.hud.ac.uk/) in collaboration with [Tierrasphere](https://www.tierrasphere.com/).")

st.subheader("Contact")

st.write("For more information, please contact [Tierrasphere](https://www.tierrasphere.com/#contact).")

