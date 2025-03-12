import streamlit as st # type: ignore
import random

def display_table_data(master_dataset, filename='dataset.csv'):
    """
    Display the data from the master dataset in a table format with options for pagination or full table view. Also, provide a download button to download the dataset as a CSV file.
    @param master_dataset - The dataset to be displayed
    @param filename - The name of the CSV file to be downloaded (default is 'dataset.csv')
    @return None
    """
    # Add download button for full dataset
    st.download_button(
        "Download Full Dataset",
        master_dataset.to_csv(index=False).encode('utf-8'),
        filename, 
        "text/csv",
        key=random.randint(1, 1000000)
    )
    
    # Allow switching between paginated and full view using a container
    view_container = st.container()
    with view_container:
        view_type = st.radio(
            "Select View Type",
            ["Paginated View", "Full Table"],
            horizontal=True,
            key=random.randint(1, 1000000)
        )
        
        if view_type == "Paginated View":
            # Calculate total pages once
            rows_per_page = 10
            total_rows = len(master_dataset)
            total_pages = (total_rows + rows_per_page - 1) // rows_per_page
            
            # Use columns for page navigation
            col1, col2 = st.columns([3,1])
            with col1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=random.randint(1, 1000000))
            
            # Calculate indices efficiently
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            # Display data using a cached dataframe
            @st.cache_data
            def get_page_data(df, start, end):
                return df.iloc[start:end]
            
            st.dataframe(get_page_data(master_dataset, start_idx, end_idx))
            st.caption(f"Showing page {page} of {total_pages} ({total_rows} total records)")
        else:
            # Cache the full table display
            @st.cache_data
            def get_full_data(df):
                return df
                
            st.dataframe(get_full_data(master_dataset))