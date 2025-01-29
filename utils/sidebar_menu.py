import streamlit as st # type: ignore
import datetime

version = "1.0.0"
tierra_foods_logo = "assets/images/tierra-foods-logo.png"
uoh_logo = "assets/images/uoh-logo.svg"

def sidebar(title="Tierra Sphere Carbon Capture Prediction Application", layout_style="centered"):
    st.set_page_config(
        page_title=title,
        page_icon="assets/images/tierra-foods-logo.png",
        layout=layout_style
    )

    with st.sidebar:

        st.logo(tierra_foods_logo, size="small")
        
        # st.page_link("app.py", label="Home")
        # st.page_link("pages/soil_data_selector.py", label="Soil Data Selector")
        # st.page_link("pages/visualise_soil_data.py", label="Visualise Soil Data")

        st.html("""
                <div style="text-align: center; font-size: 12px; color: #808080;"">
                    <h3>Developed in collaboration with
                        <a target="_blank" href="https://www.tierrafoods.com/">Tierra Foods</a> and 
                        The <a target="_blank" href="https://www.hud.ac.uk/">University of Huddersfield</a>
                    </h3>
                </div>
                """)
        
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            st.image(tierra_foods_logo, width=100)
        with col2:
            st.image(uoh_logo, width=100)

        st.html(f"""
                <div style="text-align: center;">
                    <p style="font-size: 12px; color: #808080;">
                        Version {version} &copy; {datetime.datetime.now().year}
                    </p>
                </div>
                """)