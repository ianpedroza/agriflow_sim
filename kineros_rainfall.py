import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# ----------------------------
# (1) Parsing the KINEROS2 Output File for Runoff Data
# ----------------------------
def parse_runoff_file(runoff_filename):
    """
    Parse a KINEROS2 output text file to extract, for each plane,
    the outflow and the cumulated area.
    
    Returns a dictionary mapping plane ID (int) to runoff efficiency (outflow/area).
    Only rows for planes (not channels) are used.
    """
    runoff_dict = {}
    with open(runoff_filename, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the "Tabular Summary" section
    start_idx = None
    for idx, line in enumerate(lines):
        if "Tabular Summary" in line:
            start_idx = idx
            break
    if start_idx is None:
        print("Could not find Tabular Summary section in runoff file.")
        return runoff_dict

    # Skip header lines (assume the next 3 lines are header)
    table_lines = lines[start_idx+3:]
    for line in table_lines:
        # Skip empty lines
        if not line.strip():
            continue
        # Check if line starts with a digit (the ID)
        tokens = line.strip().split()
        if len(tokens) < 10:
            continue  # skip lines that do not have enough tokens
        if not tokens[0].isdigit():
            continue
        # tokens[0]: ID, tokens[1]: Element Type
        if tokens[1].lower() != "plane":
            continue  # ignore channels
        
        try:
            plane_id = int(tokens[0])
            # tokens[2] is Element Cumulated Area (ft^2) and tokens[6] is Outflow (cu ft)
            area = float(tokens[2])
            outflow = float(tokens[6])
            # Compute runoff efficiency as outflow per unit area
            efficiency = outflow / area if area != 0 else 0
            runoff_dict[plane_id] = efficiency
        except Exception as e:
            print(f"Error parsing line: {line}\n{e}")
    return runoff_dict

# ----------------------------
# (2) Parsing the Parameter File for Plane Coordinates
# ----------------------------
def parse_param_file(param_filename):
    """
    Parse a parameter file (test_rainfall.par) to extract the X and Y coordinates for each plane.
    Returns a dictionary mapping plane ID (int) to (X, Y) tuple.
    """
    plane_coords = {}
    with open(param_filename, 'r') as f:
        lines = f.readlines()
    
    current_id = None
    for line in lines:
        line = line.strip()
        # Look for a line that starts a plane block
        if line.startswith("BEGIN PLANE"):
            current_id = None
        elif line.startswith("ID ="):
            # Expect something like: "ID = 10"
            try:
                parts = line.split("=")
                current_id = int(parts[1].strip())
            except:
                current_id = None
        elif "X =" in line and "Y =" in line and current_id is not None:
            # Expect something like: "X = 561917.8, Y = 275357.2"
            try:
                # Remove any trailing comments.
                tokens = line.split(',')
                x_token = tokens[0]
                y_token = tokens[1]
                x_val = float(x_token.split('=')[1].strip())
                y_val = float(y_token.split('=')[1].strip())
                plane_coords[current_id] = (x_val, y_val)
            except Exception as e:
                print(f"Error parsing coordinates in line: {line}\n{e}")
    return plane_coords

# ----------------------------
# (3) Load the DEM from CSV (terrain_mesh.csv)
# ----------------------------
def load_csv_as_dem(csv_filename):
    """
    Load a DEM from a CSV file.
    Assumes the CSV has columns: "x", "y", "z".
    Returns: dem (2D array), x_grid, y_grid (2D coordinate arrays).
    """
    df = pd.read_csv(csv_filename)
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    # Create grid arrays (assume regular grid)
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    # Create DEM by pivoting; assume z values are arranged in order.
    dem = df.pivot(index="y", columns="x", values="z").values
    # Note: Depending on your CSV, you might need to flip the y-axis.
    return dem, x_grid, y_grid

# ----------------------------
# (4) Interpolate DEM z at a given (x,y) point using nearest neighbor
# ----------------------------
def get_elevation(x, y, x_grid, y_grid, dem):
    """
    Given x,y (scalar), return the corresponding z value from the DEM.
    Uses a nearest-neighbor approach.
    """
    # Flatten grid arrays for easier nearest-neighbor search.
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    dem_flat = dem.flatten()
    # Compute distances to each grid point.
    dists = np.sqrt((x_flat - x)**2 + (y_flat - y)**2)
    idx = np.argmin(dists)
    return dem_flat[idx]

# ----------------------------
# (5) Main Execution: Parse files, compute runoff efficiency and overlay plot.
# ----------------------------
def main():
    # File paths (adjust as needed)
    runoff_file = "test_rainfall.out"    # KINEROS2 output file
    param_file = "test_rainfall.par"       # Parameter file with plane definitions
    dem_csv = "terrain_mesh.csv"           # DEM CSV file
    
    # Parse runoff efficiency from the output file
    runoff_dict = parse_runoff_file(runoff_file)
    if not runoff_dict:
        print("No runoff data found!")
        return
    print("Runoff efficiency by plane (ID: efficiency):")
    for pid, eff in runoff_dict.items():
        print(f"{pid}: {eff:.4f}")
    
    # Parse plane coordinates from the parameter file
    plane_coords = parse_param_file(param_file)
    if not plane_coords:
        print("No plane coordinates found!")
        return

    # Load DEM from CSV
    dem, x_grid, y_grid = load_csv_as_dem(dem_csv)
    
    # For each plane (by ID) that exists in both dictionaries, pair its (X, Y) with its runoff efficiency.
    # Also get an elevation from the DEM at (X,Y) using nearest-neighbor.
    plane_runoff_data = []
    for pid, (x_val, y_val) in plane_coords.items():
        if pid in runoff_dict:
            eff = runoff_dict[pid]
            z_val = get_elevation(x_val, y_val, x_grid, y_grid, dem)
            plane_runoff_data.append((x_val, y_val, z_val, eff))
    
    # Prepare arrays for plotting
    plane_runoff_data = np.array(plane_runoff_data)  # shape (N,4)
    if plane_runoff_data.size == 0:
        print("No matching plane runoff data found.")
        return
    Xp = plane_runoff_data[:, 0]
    Yp = plane_runoff_data[:, 1]
    Zp = plane_runoff_data[:, 2]
    Eff = plane_runoff_data[:, 3]
    
    # Create a 3D plot: Plot DEM surface, and overlay scatter markers colored by runoff efficiency.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot DEM surface (using meshgrid from the CSV)
    surf = ax.plot_surface(x_grid, y_grid, dem, cmap=cm.terrain, alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")
    
    # Scatter plot the plane locations with color representing runoff efficiency.
    sc = ax.scatter(Xp, Yp, Zp, c=Eff, cmap='viridis', s=100, edgecolor='k', depthshade=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label="Runoff Efficiency (Outflow/Area)")
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Elevation (meters)")
    ax.set_title("Runoff Efficiency at Plane Locations over DEM")
    
    plt.show()

if __name__ == "__main__":
    main()
