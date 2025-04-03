from flask import Flask, jsonify, request
import subprocess
import os
import re
from dataclasses import dataclass

app = Flask(__name__)

# Paths
KINEROS_EXECUTABLE = "k2"  # Make sure it's in the same directory
KINEROS_INPUT_FILE = "kin.fil"  # Used in batch mode
PAR_FILE = "agriflow.par"
PRE_FILE = "agriflow.pre"
SIMULATION_DIR = os.path.abspath(".")
OUTPUT_FILE = "agriflow.out"  # Updated output file name

# --- Data structure for sensor planes ---
@dataclass
class Plane:
    id: int
    x: float
    y: float
    width: float
    length: float
    slope: float
    saturation: float

# --- Constants for PAR file generation ---
PLANE_CONSTANTS = {
    "MANNING": 0.05,
    "KSAT": 0.4,
    "G": 6.0,
    "POROSITY": 0.45,
    "ROCK": 0.0,
    "DIST": 0.6,
    "CV": 0.7,
    "INTER": 0.002,
    "CANOPY": 0.7
}

CHANNEL_CONSTANTS = {
    "MANNING": 0.03,
    "SS1": 0.5,
    "SS2": 0.5,
    "KSAT": 7.5,
    "G": 2.5,
    "POROSITY": 0.40,
    "ROCK": 0.1,
    "DIST": 0.8,
    "SAT": 0.1,
    "WOOL": "YES",
    "SLOPE": 0.1,
    "WIDTH": 0
}

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400

        data = request.get_json()

        rainfall_series = [entry['rainfall'] for entry in sorted(
            data.get('rainfallData', []),
            key=lambda r: r['timestamp']
        )]

        # --- Determine simulation duration based on rainfall data ---
        simulation_duration_min = len(rainfall_series) * 60  # each point is 60 minutes apart

        # --- Write updated kin.fil ---
        kin_filenames = f"{PAR_FILE},{PRE_FILE},{OUTPUT_FILE},\"agriflow_simulation\",{simulation_duration_min},5,Y,N,N,Y"
        with open(os.path.join(SIMULATION_DIR, KINEROS_INPUT_FILE), "w") as f:
            f.write(kin_filenames + "\n")

        sensor_planes = []
        for i, entry in enumerate(data.get('sensorData', [])):
            plane = Plane(
                id=entry.get('id', i),
                x=entry['x'],
                y=entry['y'],
                width=entry['width'],
                length=entry['length'],
                slope=entry['slope'],
                saturation=entry['averageMoisture'] / 100.0
            )
            sensor_planes.append(plane)

        par_lines = []
        clen = sensor_planes[0].length if sensor_planes else 100.0
        nele = len(sensor_planes) * 2
        par_lines.append("BEGIN GLOBAL")
        par_lines.append(f"  CLEN = {clen},  UNITS = METRIC")
        par_lines.append(f"  Nele = {nele}")
        par_lines.append("END GLOBAL\n")

        channel_id_start = max(plane.id for plane in sensor_planes) + 1
        for i, plane in enumerate(sensor_planes):
            gage_id = plane.id
            par_lines.append("BEGIN PLANE")
            par_lines.append(f"  ID = {plane.id}")
            par_lines.append(
                f"  LENGTH = {plane.length:.2f}, WIDTH = {plane.width:.2f}, SLOPE = {plane.slope:.4f}, MANNING = {PLANE_CONSTANTS['MANNING']:.3f}")
            par_lines.append(
                f"  KSAT = {PLANE_CONSTANTS['KSAT']}, G = {PLANE_CONSTANTS['G']}, POROSITY = {PLANE_CONSTANTS['POROSITY']}, ROCK = {PLANE_CONSTANTS['ROCK']}, DIST = {PLANE_CONSTANTS['DIST']}, CV = {PLANE_CONSTANTS['CV']}")
            par_lines.append(
                f"  INTER = {PLANE_CONSTANTS['INTER']}, CANOPY = {PLANE_CONSTANTS['CANOPY']}, SATURATION = {plane.saturation:.4f}")
            par_lines.append(f"  X = {plane.x:.4f}, Y = {plane.y:.4f}")
            par_lines.append(f"  RAINGAGE = {gage_id}")
            par_lines.append(f"  PRINT = 2")
            par_lines.append("END PLANE\n")

            channel_id = channel_id_start + i
            par_lines.append("BEGIN CHANNEL")
            par_lines.append(f"  ID = {channel_id}")
            par_lines.append(f"  LATERAL = {plane.id}")
            par_lines.append(
                f"  LENGTH = {plane.length:.2f}, WIDTH = {CHANNEL_CONSTANTS['WIDTH']}, SLOPE = {CHANNEL_CONSTANTS['SLOPE']}, MANNING = {CHANNEL_CONSTANTS['MANNING']}")
            par_lines.append(
                f"  SS1 = {CHANNEL_CONSTANTS['SS1']}, SS2 = {CHANNEL_CONSTANTS['SS2']}")
            par_lines.append(
                f"  KSAT = {CHANNEL_CONSTANTS['KSAT']}, G = {CHANNEL_CONSTANTS['G']}, POROSITY = {CHANNEL_CONSTANTS['POROSITY']}, DIST = {CHANNEL_CONSTANTS['DIST']}, ROCK = {CHANNEL_CONSTANTS['ROCK']}")
            par_lines.append(f"  SAT = {CHANNEL_CONSTANTS['SAT']}")
            par_lines.append(f"  WOOL = {CHANNEL_CONSTANTS['WOOL']}")
            par_lines.append("END CHANNEL\n")

        with open(os.path.join(SIMULATION_DIR, PAR_FILE), "w") as f:
            f.write("\n".join(par_lines))

        pre_lines = []
        for plane in sensor_planes:
            pre_lines.append(f"BEGIN GAGE {plane.id}")
            pre_lines.append(f"  X = {plane.x:.4f}, Y = {plane.y:.4f}")
            pre_lines.append(f"  N = {len(rainfall_series)}")
            pre_lines.append("  TIME        INTENSITY !(mm/hr)")
            for i, val in enumerate(rainfall_series):
                time_min = i * 60
                pre_lines.append(f"  {time_min:<10} {val:.2f}")
            pre_lines.append("END\n")

        with open(os.path.join(SIMULATION_DIR, PRE_FILE), "w") as f:
            f.write("\n".join(pre_lines))

        result = subprocess.run(
            [KINEROS_EXECUTABLE, "-b", KINEROS_INPUT_FILE],
            cwd=SIMULATION_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "KINEROS execution failed",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        output_path = os.path.join(SIMULATION_DIR, OUTPUT_FILE)
        if not os.path.exists(output_path):
            return jsonify({"status": "error", "message": f"Output file '{OUTPUT_FILE}' not found"}), 500

        results_dict = {}
        current_plane_id = None
        inside_table = False
        table_data = []

        with open(output_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.strip().startswith("Plane Element"):
                current_plane_id = int(line.strip().split()[-1])
                inside_table = False
                table_data = []

            elif current_plane_id and line.strip().startswith("Elapsed Time"):
                inside_table = True
                lines_to_skip = 2
                continue
            
            elif inside_table and lines_to_skip > 0:
                lines_to_skip -= 1
                continue

            elif inside_table:
                if line.strip() == "" or line.strip().startswith("Channel Elem") or line.strip().startswith("Plane Element"):
                    inside_table = False
                    if current_plane_id:
                        results_dict[current_plane_id] = {
                            "id": current_plane_id,
                            "table": table_data
                        }
                        current_plane_id = None
                    continue

                parts = line.split()
                if len(parts) == 4:
                    try:
                        time, rainfall, outflow_mmhr, outflow_cumecs = map(float, parts)
                        table_data.append({
                            "time_min": time,
                            "rainfall_mm_hr": rainfall,
                            "outflow_mm_hr": outflow_mmhr,
                            "outflow_cu_m_s": outflow_cumecs
                        })
                    except ValueError:
                        continue
        for i, line in enumerate(lines):
            if line.strip().startswith("Tabular Summary of Element Hydrologic Components"):
                for summary_line in lines[i+4:]:
                    if not summary_line.strip():
                        break
                    parts = summary_line.split()
                    if len(parts) < 9:
                        continue
                    try:
                        element_id = int(parts[0])
                        element_type = parts[1]
                        if element_type != "Plane":
                            continue
                        rainfall = float(parts[5])
                        outflow = float(parts[6])
                        outflow_percentage = outflow / rainfall if rainfall != 0 else 0
                        if element_id in results_dict:
                            results_dict[element_id].update({
                                "rainfall": rainfall,
                                "outflow": outflow,
                                "outflow_percentage": outflow_percentage
                            })
                        else:
                            results_dict[element_id] = {
                                "id": element_id,
                                "rainfall": rainfall,
                                "outflow": outflow,
                                "outflow_percentage": outflow_percentage,
                                "table": []
                            }
                    except ValueError:
                        continue
                break

        return jsonify({"returnData": list(results_dict.values())})

    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "KINEROS timed out (possible infinite wait for input?)"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
