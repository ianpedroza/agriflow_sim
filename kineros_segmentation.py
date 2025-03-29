import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

##############################################
# 1. DEM Loading & Quad Partitioning Functions
##############################################

def load_csv_as_dem(file_path):
    """
    Load terrain mesh data from CSV into a structured DEM grid.
    Assumes the CSV has columns: "x", "y", and "z".
    """
    df = pd.read_csv(file_path)
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())[::-1]  # reverse Y for natural orientation

    dem = np.zeros((len(y_unique), len(x_unique)))
    for _, row in df.iterrows():
        x_idx = np.where(x_unique == row["x"])[0][0]
        y_idx = np.where(y_unique == row["y"])[0][0]
        dem[y_idx, x_idx] = row["z"]

    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    return dem, x_grid, y_grid

def partition_grid_quads(dem, threshold=5):
    """
    Partition the DEM grid into non-overlapping quadrilateral regions (merged grid cells)
    based on a simple slope measure.
    
    Returns a list of bounding boxes (in grid-cell coordinates) as tuples:
         (i_min, j_min, i_max, j_max)
    """
    rows, cols = dem.shape
    nrows = rows - 1
    ncols = cols - 1

    quad_slopes = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            elevs = [dem[i, j], dem[i, j+1], dem[i+1, j+1], dem[i+1, j]]
            quad_slopes[i, j] = max(elevs) - min(elevs)
    bins = (quad_slopes / threshold).astype(int)

    covered = np.zeros((nrows, ncols), dtype=bool)
    rectangles = []
    for i in range(nrows):
        for j in range(ncols):
            if not covered[i, j]:
                bin_val = bins[i, j]
                width = 1
                while j + width < ncols and not covered[i, j+width] and bins[i, j+width] == bin_val:
                    width += 1
                height = 1
                done = False
                while i + height < nrows and not done:
                    for k in range(width):
                        if covered[i+height, j+k] or bins[i+height, j+k] != bin_val:
                            done = True
                            break
                    if not done:
                        height += 1
                for di in range(height):
                    for dj in range(width):
                        covered[i+di, j+dj] = True
                rectangles.append((i, j, i + height - 1, j + width - 1))
    return rectangles

def convert_box_to_quad(box, x_grid, y_grid, dem):
    """
    Convert a bounding box (in grid-cell indices) into a quad defined by four real-world vertices.
    The box (i_min, j_min, i_max, j_max) spans vertices:
      top-left:    (i_min, j_min)
      top-right:   (i_min, j_max+1)
      bottom-right:(i_max+1, j_max+1)
      bottom-left: (i_max+1, j_min)
    """
    i_min, j_min, i_max, j_max = box
    x_min, y_min = x_grid[i_min, j_min], y_grid[i_min, j_min]
    x_max, y_max = x_grid[i_max + 1, j_max + 1], y_grid[i_max + 1, j_max + 1]
    z_vals = [dem[i_min, j_min], dem[i_min, j_max+1], dem[i_max+1, j_max+1], dem[i_max+1, j_min]]
    # We form the quad with the DEM values at the corners.
    quad = np.array([
        [x_min, y_min, dem[i_min, j_min]],       # top-left
        [x_max, y_min, dem[i_min, j_max+1]],       # top-right
        [x_max, y_max, dem[i_max+1, j_max+1]],     # bottom-right
        [x_min, y_max, dem[i_max+1, j_min]]        # bottom-left
    ])
    return quad

def plot_quad_mesh(dem, x_grid, y_grid, quads):
    """Plot the DEM and overlay the quad mesh using real-world coordinates."""
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(x_grid, y_grid, dem, cmap="terrain", edgecolor="none", alpha=0.8)
    ax1.set_title("Original DEM")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Elevation (m)")
    
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(x_grid, y_grid, dem, cmap="terrain", edgecolor="none", alpha=0.4)
    polys = [quad for quad in quads]
    pc = Poly3DCollection(polys, alpha=0.7, edgecolor="black", facecolor="royalblue")
    ax2.add_collection3d(pc)
    ax2.set_title("Quad Mesh")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Elevation (m)")
    
    ax1.set_xlim(x_grid.min(), x_grid.max())
    ax1.set_ylim(y_grid.min(), y_grid.max())
    ax1.set_zlim(dem.min(), dem.max())
    ax2.set_xlim(x_grid.min(), x_grid.max())
    ax2.set_ylim(y_grid.min(), y_grid.max())
    ax2.set_zlim(dem.min(), dem.max())
    
    plt.tight_layout()
    plt.show()

##############################################
# 2. Compute Plane & Channel Parameters
##############################################

def compute_plane_parameters(quads):
    """
    For each quad (plane) compute:
      - X and Y: we use the lower-left corner (first vertex)
      - LENGTH and WIDTH: based on horizontal extents
      - SLOPE: vertical difference divided by diagonal distance
      - z_avg: average elevation from the quad vertices
      - Also compute a centroid = (x + length/2, y + width/2)
    
    Returns a list of plane dictionaries.
    """
    planes = []
    for idx, quad in enumerate(quads):
        x_coords = quad[:,0]
        y_coords = quad[:,1]
        z_coords = quad[:,2]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        dx, dy = x_max - x_min, y_max - y_min
        length = max(dx, dy)
        width = min(dx, dy)
        z_min, z_max = z_coords.min(), z_coords.max()
        horiz_dist = np.sqrt(dx**2 + dy**2)
        slope = (z_max - z_min) / horiz_dist if horiz_dist != 0 else 0
        z_avg = np.mean(z_coords)
        centroid = (x_min + dx/2.0, y_min + dy/2.0)
        planes.append({
            "id": idx + 1,
            "label": chr(65 + idx) if idx < 26 else f"#{idx+1}",
            "x": x_min,
            "y": y_min,
            "length": length,
            "width": width,
            "slope": slope + 0.001,
            "manning": 0.061,
            "ksat": 0.521,
            "g": 6.65,
            "porosity": 0.453,
            "rock": 0.406,
            "dist": 0.6,
            "cv": 0.8,
            "inter": 0.014,
            "canopy": 1.0,
            "z_avg": z_avg,
            "centroid": centroid
        })
    return planes

def detect_channels(dem, x_grid, y_grid, gradient_threshold=5):
    """
    Detect channels from the DEM by computing the gradient magnitude using Sobel filters,
    thresholding, and labeling connected regions.
    Returns a list of channel dictionaries with x, y, length, width, and slope.
    """
    gx = ndimage.sobel(dem, axis=1)
    gy = ndimage.sobel(dem, axis=0)
    grad_mag = np.hypot(gx, gy)
    channel_mask = grad_mag > gradient_threshold
    labeled, num_features = ndimage.label(channel_mask)
    channels = []
    for i in range(1, num_features+1):
        indices = np.where(labeled == i)
        if len(indices[0]) == 0:
            continue
        i_min, i_max = indices[0].min(), indices[0].max()
        j_min, j_max = indices[1].min(), indices[1].max()
        centroid_i = np.mean(indices[0])
        centroid_j = np.mean(indices[1])
        x_centroid = x_grid[int(round(centroid_i)), int(round(centroid_j))]
        y_centroid = y_grid[int(round(centroid_i)), int(round(centroid_j))]
        x_min, x_max = x_grid[i_min, j_min], x_grid[i_max, j_max]
        y_min, y_max = y_grid[i_min, j_min], y_grid[i_max, j_max]
        dx, dy = abs(x_max - x_min), abs(y_max - y_min)
        length = max(dx, dy)
        width = min(dx, dy)
        if length == 0 or width == 0:
            continue  # invalid channel, skip it
        avg_slope = np.mean(grad_mag[indices])
        channels.append({
            "x": x_centroid,
            "y": y_centroid,
            "length": length,
            "width": width,
            "slope": avg_slope + 0.001
        })
    return channels

def assign_channel_connectivity(channel_params, plane_params, distance_threshold=500):
    """
    For each channel, find all planes whose centroids are within a given distance of the channel centroid.
    Then assign the plane with the highest average elevation as the UPSTREAM element and all others as LATERAL.
    """
    for ch in channel_params:
        adjacent = []
        for plane in plane_params:
            px, py = plane["centroid"]
            dist = np.sqrt((ch["x"] - px)**2 + (ch["y"] - py)**2)
            if dist < distance_threshold:
                adjacent.append((dist, plane))
        adjacent.sort(key=lambda tup: tup[0])
        if adjacent:
            # Choose upstream as the one with maximum average elevation
            upstream_plane = max([p for d, p in adjacent], key=lambda p: p.get("z_avg", 0))
            ch["upstream"] = str(upstream_plane["id"])
            laterals = [p for d, p in adjacent if p["id"] != upstream_plane["id"]]
            ch["lateral"] = ",".join(str(p["id"]) for p in laterals) if laterals else ""
        else:
            ch["upstream"] = ""
            ch["lateral"] = ""
    return channel_params

##############################################
# 3. Parameter File Generation Functions
##############################################

def generate_parameter_file(plane_params, channel_params, out_file):
    """
    Write out a parameter file that includes a GLOBAL block, one block per plane,
    and one block per channel (with connectivity filled in).
    """
    with open(out_file, "w") as f:
        f.write("! Walnut Gulch subwatershed 11, 17 elements\n\n")
        f.write("BEGIN GLOBAL\n  CLEN = 3530.0, UNITS = ENGLISH\nEND GLOBAL\n\n")
        # Write plane blocks.
        for plane in plane_params:
            f.write(f"BEGIN PLANE ! Overland flow area '{plane['label']}'\n")
            f.write(f"  ID = {plane['id']}\n")
            f.write(f"  LENGTH = {plane['length']:.1f}, WIDTH = {plane['width']:.1f}, SLOPE = {plane['slope']:.3f}, MANNING = {plane['manning']:.3f}\n")
            f.write(f"  KSAT = {plane['ksat']:.3f}, G = {plane['g']:.2f}, POROSITY = {plane['porosity']:.3f}, ROCK = {plane['rock']:.3f}, DIST = {plane['dist']:.1f}, CV = {plane['cv']:.1f}\n")
            f.write(f"  INTER = {plane['inter']:.3f}, CANOPY = {plane['canopy']:.1f}\n")
            f.write(f"  X = {plane['x']:.1f}, Y = {plane['y']:.1f}\n")
            f.write("END PLANE\n\n")
        # Write channel blocks.
        for ch in channel_params:
            # Skip channels with invalid dimensions (already filtered in detect_channels)
            if ch["length"] == 0 or ch["width"] == 0:
                continue
            f.write("BEGIN CHANNEL ! Automatically detected channel\n")
            # Here we assume channel IDs are assigned in order.
            f.write(f"  ID = {channel_params.index(ch)+1}\n")
            f.write(f"  UPSTREAM = {ch['upstream']}, LATERAL = {ch['lateral']}\n")
            f.write(f"  LENGTH = {ch['length']:.1f}, WIDTH = {ch['width']:.1f}, SLOPE = {ch['slope']:.3f}, MANNING = 0.042\n")
            f.write("  SS1 = 0.653, SS2 = 0.524\n")
            f.write("  KSAT = 10.998, G = 2.21, POROSITY = 0.437, DIST = 0.8\n")
            f.write("  SAT = 0.1\n")
            f.write("  WOOL = YES\n")
            f.write("END CHANNEL\n\n")

##############################################
# 4. Main Execution
##############################################

# Replace with your actual CSV file path.
file_path = "terrain_mesh.csv"  
dem, x_grid, y_grid = load_csv_as_dem(file_path)

# Partition DEM grid into quads and convert them.
rectangles = partition_grid_quads(dem, threshold=0.5)
quads = [convert_box_to_quad(box, x_grid, y_grid, dem) for box in rectangles]

# (Optional) Plot the DEM with the quad mesh overlay.
plot_quad_mesh(dem, x_grid, y_grid, quads)

# Compute plane parameters (including centroids and z_avg).
plane_params = compute_plane_parameters(quads)

# Detect channels from the DEM.
channel_params = detect_channels(dem, x_grid, y_grid, gradient_threshold=5)

# Assign connectivity: for each channel, determine adjacent planes.
channel_params = assign_channel_connectivity(channel_params, plane_params, distance_threshold=500)

# Generate parameter file.
output_file = "parameter.txt"
generate_parameter_file(plane_params, channel_params, output_file)
print(f"Parameter file saved to {output_file}")
