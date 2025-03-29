import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_kineros_tabular_summary(content):
    """Extracts runoff data from the tabular summary of the KINEROS2 output file."""
    lines = content.split('\n')
    planes = {}

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
            runoff = float(parts[6])  # Total Runoff (cu ft)

            # Store runoff value for the plane
            planes[plane_id] = runoff
    
    return planes

def parse_plane_coordinates(param_file_content):
    """Extracts plane details (coordinates, length, width, slope) from the parameter file."""
    lines = param_file_content.split("\n")
    planes = {}

    parsing = False
    current_plane = {}

    for line in lines:
        line = line.strip()

        if line.startswith("BEGIN PLANE"):
            parsing = True
            current_plane = {}  # Reset for new plane
            continue

        if parsing:
            if "ID =" in line:
                current_plane["ID"] = int(line.split("=")[1].split(",")[0].strip())
            elif "LENGTH =" in line and "WIDTH =" in line and "SLOPE =" in line:
                parts = line.split(",")
                for part in parts:
                    key, value = part.split("=")
                    current_plane[key.strip()] = float(value.strip())
            elif "X =" in line and "Y =" in line:
                parts = line.replace("X =", "").replace("Y =", "").split(",")
                current_plane["X"] = float(parts[0].strip())
                current_plane["Y"] = float(parts[1].strip())

            if "END PLANE" in line:
                parsing = False
                if all(k in current_plane for k in ["ID", "LENGTH", "WIDTH", "SLOPE", "X", "Y"]):
                    planes[current_plane["ID"]] = current_plane

    return planes

def create_3d_terrain(planes, runoff_data):
    """Creates a 3D terrain plot with color coding based on runoff and corrected scaling."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    all_x, all_y, all_z = [], [], []
    min_runoff, max_runoff = min(runoff_data.values()), max(runoff_data.values())

    for plane_id, plane in planes.items():
        length, width, slope = plane["LENGTH"], plane["WIDTH"], plane["SLOPE"]
        x_center, y_center = plane["X"], plane["Y"]

        # Compute bounding box
        x_min, x_max = x_center - length / 2, x_center + length / 2
        y_min, y_max = y_center - width / 2, y_center + width / 2

        x = np.linspace(x_min, x_max, 10)
        y = np.linspace(y_min, y_max, 10)
        X, Y = np.meshgrid(x, y)
        Z = slope * (X - x_min)  # Elevation based on slope

        # Store values for axis scaling
        all_x.extend(X.flatten())
        all_y.extend(Y.flatten())
        all_z.extend(Z.flatten())

        # Get the runoff value for this plane
        runoff = runoff_data.get(plane_id, 0)
        norm_runoff = (runoff - min_runoff) / (max_runoff - min_runoff + 1e-5)  # Normalize 0 to 1
        color = (1 - norm_runoff, norm_runoff, 0)  # Green to Red scale

        # Plot the plane as a surface
        ax.plot_surface(X, Y, Z, color=color, alpha=0.8, edgecolor='k')

    # Set equal scaling for all axes
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)

    max_range = max(x_range, y_range, z_range)  # Use the largest range for scaling

    # Calculate midpoints for centering
    x_mid, y_mid, z_mid = np.mean(all_x), np.mean(all_y), np.mean(all_z)

    # Set axis limits so that scaling is equal
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Elevation")
    ax.set_title("3D Terrain Visualization with Runoff (Equal Scaling)")

    # Show color legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("jet"), norm=plt.Normalize(vmin=min_runoff, vmax=max_runoff))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Runoff (cu ft)")

    plt.show()

# --- Load and Analyze ---
param_file_path = "wg11.par"  # Update with actual file path
output_file_path = "4aug80.out"  # Update with actual KINEROS2 output file

with open(param_file_path, 'r') as f:
    param_content = f.read()

with open(output_file_path, 'r') as f:
    output_content = f.read()

# Extract data
runoff_data = parse_kineros_tabular_summary(output_content)
planes = parse_plane_coordinates(param_content)

# Create 3D terrain plot
create_3d_terrain(planes, runoff_data)
