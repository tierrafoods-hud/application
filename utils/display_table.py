import streamlit as st # type: ignore

def display_table_data(master_dataset):
    # Add download button for full dataset
        st.download_button(
            "Download Full Dataset",
            master_dataset.to_csv(index=False).encode('utf-8'),
            "soil_data.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Allow switching between paginated and full view
        view_type = st.radio(
            "Select View Type",
            ["Paginated View", "Full Table"],
            horizontal=True
        )
        
        if view_type == "Paginated View":
            # Calculate total pages
            rows_per_page = 10
            total_pages = len(master_dataset) // rows_per_page + (1 if len(master_dataset) % rows_per_page > 0 else 0)
            
            # Page selector
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            # Display paginated data
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            st.dataframe(master_dataset.iloc[start_idx:end_idx])
            
            st.caption(f"Showing page {page} of {total_pages}")
        else:
            # Display full table
            st.dataframe(master_dataset)