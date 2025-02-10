import streamlit as st # type: ignore
import datetime

version = "1.0.0"
tierra_logo = "assets/images/tierra-sphere-logo.png"
uoh_logo = "assets/images/uoh-logo.svg"

def sidebar(title="Tierra Sphere Carbon Capture Prediction Application", layout_style="centered"):
    st.set_page_config(
        page_title=title,
        page_icon=tierra_logo,
        layout=layout_style
    )

    with st.sidebar:

        st.logo(tierra_logo, size="large")

        # st.page_link("app.py", label="Home")
        # st.page_link("pages/soil_data_selector.py", label="Soil Data Selector")
        # st.page_link("pages/visualise_soil_data.py", label="Visualise Soil Data")

        col1, col2 = st.columns(2, vertical_alignment="center", gap="small")
        with col1:
            st.image(tierra_logo, use_container_width=True)
        with col2:
            st.image(uoh_logo, use_container_width=True)
        
        st.markdown("""
                <div style="text-align: center;">
                    <h3 style="font-size: 14px; font-weight: bold; color: #306f56;">Developed in collaboration with
                        <a href="https://www.tierrasphere.com/" target="_blank" rel="noopener noreferrer">Tierra Sphere</a> and 
                        The <a href="https://www.hud.ac.uk/" target="_blank" rel="noopener noreferrer">University of Huddersfield</a>
                    </h3>
                </div>
                """, unsafe_allow_html=True)

        st.html(f"""
                <div style="text-align: center;">
                    <p style="font-size: 12px; color: #306f56;">
                        Version {version} &copy; {datetime.datetime.now().year} All rights reserved.
                    </p>
                </div>
                """)