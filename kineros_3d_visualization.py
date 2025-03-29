import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_kineros_tabular_summary(content):
    """Extracts runoff data and coordinates from the tabular summary."""
    lines = content.split('\n')
    planes = []
    
    parsing = False
    for line in lines:
        if "Tabular Summary of Element Hydrologic Components" in line:
            parsing = True
            continue
        
        if parsing and "Plane" in line:
            parts = line.split()
            if len(parts) < 10:
                continue
            
            plane_id = int(parts[0])  # Plane ID
            area = float(parts[2])    # Area (ft^2)
            rainfall = float(parts[5])  # Total Rainfall (cu ft)
            runoff = float(parts[6])    # Total Runoff (cu ft)
            infiltration = float(parts[8])  # Infiltration (cu ft)
            
            # Compute runoff coefficient
            runoff_coeff = runoff / (rainfall + 1e-6)  # Avoid division by zero

            # Store plane data
            planes.append({
                "id": plane_id,
                "area": area,
                "rainfall": rainfall,
                "runoff": runoff,
                "infiltration": infiltration,
                "runoff_coeff": runoff_coeff
            })
    
    return pd.DataFrame(planes)

def parse_plane_coordinates(param_file_content):
    """Extracts X, Y coordinates for each plane from the parameter file."""
    lines = param_file_content.split("\n")
    planes = {}

    parsing = False
    current_plane = None
    
    for line in lines:
        line = line.strip()

        if line.startswith("BEGIN PLANE"):
            parsing = True
            current_plane = None  # Reset for the new plane
            continue
        
        if parsing:
            if "ID =" in line:  # Look specifically for ID
                parts = line.split("=")
                current_plane = int(parts[1].split(",")[0].strip())  # Extract Plane ID
                continue  # Move to next line

            if "X =" in line and "Y =" in line and current_plane is not None:
                parts = line.replace("X =", "").replace("Y =", "").split(",")
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                planes[current_plane] = (x, y)
                parsing = False  # End of plane section
    
    return planes

def plot_3d_runoff(planes_df, plane_coords):
    """Creates a 3D plot showing runoff across all planes."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinate and runoff data
    x_vals = []
    y_vals = []
    z_vals = []
    colors = []

    for index, row in planes_df.iterrows():
        plane_id = row["id"]
        if plane_id in plane_coords:
            x, y = plane_coords[plane_id]
            runoff = row["runoff"]  # Color by runoff coefficient
            
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(runoff)
            colors.append(runoff)

    # Normalize colors for visualization
    norm = plt.Normalize(min(colors), max(colors))
    cmap = plt.get_cmap("jet")  # Color scale

    # Scatter plot with color-coded runoff intensity
    sc = ax.scatter(x_vals, y_vals, z_vals, c=colors, cmap=cmap, s=100)

    # Labels and formatting
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Runoff Coefficient")
    ax.set_title("3D Visualization of Runoff Across Planes")
    fig.colorbar(sc, label="Runoff Coefficient")

    plt.show()

# --- Load and Analyze ---
param_file_path = "wg11.par"  # Replace with actual parameter file path
output_file_path = "4aug80.out"  # Replace with actual KINEROS2 output file

with open(param_file_path, 'r') as f:
    param_content = f.read()

with open(output_file_path, 'r') as f:
    output_content = f.read()

# Extract data
planes_df = parse_kineros_tabular_summary(output_content)
plane_coords = parse_plane_coordinates(param_content)

# Create the 3D runoff visualization
plot_3d_runoff(planes_df, plane_coords)
