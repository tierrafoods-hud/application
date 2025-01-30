import numpy as np
import geopandas as gpd
from shapely.geometry import Point

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
