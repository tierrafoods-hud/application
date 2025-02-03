import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_grid_cell(lat, lon, cell_size=500):
    """
    Get the grid cell ID for a given coordinate.
    
    Parameters:
    lat (float): Latitude in WGS84
    lon (float): Longitude in WGS84
    cell_size (float): Size of grid cells in meters (default 500)
    
    Returns:
    tuple: (row_id, col_id) identifying the grid cell
    """
    # Convert WGS84 coordinate to UTM
    point = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"  # WGS84
    ).to_crs("EPSG:6372")  # Mexico UTM
    
    # Get UTM coordinates
    x = point.geometry.x[0]
    y = point.geometry.y[0]
    
    # Calculate grid cell indices
    row_id = int(np.floor(y / cell_size))
    col_id = int(np.floor(x / cell_size))
    
    return (row_id, col_id)


def get_cell_bounds(grid_cell_id, cell_size=500):
    """
    Get the bounding box coordinates for a grid cell.
    
    Parameters:
    grid_cell_id (tuple): (row_id, col_id) of the cell
    cell_size (float): Size of grid cells in meters (default 500)
    
    Returns:
    tuple: (minx, miny, maxx, maxy) in UTM coordinates
    """
    row_id, col_id = grid_cell_id
    
    minx = col_id * cell_size
    miny = row_id * cell_size
    maxx = minx + cell_size
    maxy = miny + cell_size
    
    return (minx, miny, maxx, maxy)


def get_grid_cell_dataset(df, latitude='latitude', longitude='longitude', cell_size=500):
    """
    helper function to take as input a dataset and return a new dataset with the grid cell id

    Parameters:
    df (pandas.DataFrame): Input dataset
    latitude (str): Name of the column with latitude values
    longitude (str): Name of the column with longitude values
    cell_size (float): Size of grid cells in meters (default 500)

    Returns:
    pandas.DataFrame: Input dataset with the grid cell id
        cell_id are tuples (row_id, col_id) identifying the grid cell
    """
    # Get grid cell ID for each point
    df['cell_id'] = df.apply(lambda x: get_grid_cell(x[latitude], x[longitude], cell_size), axis=1)
    
    return df


def create_visuals(shapefile, cell_colours, espg=6372, cell_size=500):
    """
    cell_colours: dictionary with cell_id (tuple) as keys and values as colours (str)
        for example: {(0, 0): 'red', (1, 1): 'blue'}
    """

    colours_dict = {}
    for cell_id in cell_colours:
        bounds = get_cell_bounds(cell_id, cell_size=cell_size)
        colours_dict[(bounds[0], bounds[1])] = cell_colours[cell_id]

    # Convert Mexico to UTM coordinates
    shapefile_crs = shapefile.to_crs(epsg=espg)

    # Get the bounds of Mexico in UTM coordinates
    minx, miny, maxx, maxy = shapefile_crs.total_bounds

    # Create grid cells of 10km
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    # Create grid cells with color information
    grid_cells = []
    cell_colors = []
    for x in tqdm(x_coords):
        for y in y_coords:
            cell = box(x, y, x + cell_size, y + cell_size)
            grid_cells.append(cell)
            
            grid_found = False
            for key in colours_dict:
                if key[0] <= x < key[0] + cell_size and key[1] <= y < key[1] + cell_size:
                    cell_colors.append(colours_dict[key])
                    grid_found = True
                    break
            
            if not grid_found:
                cell_colors.append('none')
            

    # Create GeoDataFrame from grid with color information
    grid = gpd.GeoDataFrame({
        'geometry': grid_cells,
        'color': cell_colors
    }, crs=shapefile_crs.crs)

    print("Starting clipping")
    # Clip grid to Mexico's boundary
    grid_clipped = gpd.clip(grid, shapefile_crs)
    print("Done clipping")

    # Create the plot
    fig, ax = plt.subplots(figsize=(30, 20))

    # Plot Mexico
    print("Plotting shapefile region")
    shapefile_crs.plot(ax=ax, color='white', edgecolor='black', linewidth=2)

    # Plot grid with colors
    print("Plotting grid")
    # Plot cells with no color (transparent)
    transparent_cells = grid_clipped[grid_clipped['color'] == 'none']
    transparent_cells.plot(ax=ax, facecolor='none', edgecolor='grey', alpha=0.5, linewidth=0.5)

    # Plot colored cells
    colored_cells = grid_clipped[grid_clipped['color'] != 'none']
    if not colored_cells.empty:
        colored_cells.plot(ax=ax, facecolor=colored_cells['color'], edgecolor='grey', alpha=0.5, linewidth=0.5)

    # Customize the plot
    plt.title(f'Map of Region with {cell_size}m Grid', fontsize=16)
    ax.axis('off')

    plt.tight_layout()
    # plt.show()

    return fig, ax


# Example usage

################
# Add grid id to an existing dataset 
################
# import pandas as pd
# import grids

# df = pd.read_csv('Mexico_wosis_merged(in)_temp_precip_landcover(in)_ecoregions.csv')
# df_with_grids = grids.get_grid_cell_dataset(df)
# df_with_grids


################
# Create visuals
################
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import grids

# # Read the shapefile
# world = gpd.read_file('../assets/shapefiles/World_Countries_(Generalized)/World_Countries_Generalized.shp')

# # Ensure the shapefile is in WGS 84
# if world.crs != 'EPSG:6372':
#     world = world.to_crs(epsg=6372)


# mexico = world[world['COUNTRY'] == 'Mexico']

# # list of cell_id and what colour they should be
# cell_colours = {(82, 280):'red'}

# fig, ax = grids.create_visuals(mexico, cell_colours=cell_colours, cell_size=10000)

# plt.show()