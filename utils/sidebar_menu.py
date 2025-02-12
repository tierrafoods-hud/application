import streamlit as st # type: ignore
import datetime

version = "1.0.0"
tierra_logo = "assets/images/tierra-sphere-logo.png"
uoh_logo = "assets/images/uoh-logo.png"
fav_icon = "favicon.ico"

def sidebar(title="Tierrasphere Carbon Capture Prediction Application", about="", layout_style="centered"):
    st.set_page_config(
        page_title=title + " | Tierrasphere",
        page_icon=fav_icon,
        layout=layout_style,
        menu_items={
            'Get Help': 'https://www.tierrasphere.com/#contact',
            'Report a bug': "https://www.tierrasphere.com/#contact",
            'About': about if about else title
        }
    )

    with st.sidebar:

        st.logo(tierra_logo, size="large")
        st.header("Menu")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/soil_data_selector.py", label="Soil Data Selector", icon="📁")
        st.page_link("pages/visualise_soil_data.py", label="Visualise Soil Data", icon="📊")
        st.page_link("pages/make_predictions.py", label="Make Predictions", icon="📈")

        st.subheader("For developer use only")
        st.page_link("pages/build_regression_model.py", label="Train Regression Model", icon="🔍")

        st.divider()
        
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