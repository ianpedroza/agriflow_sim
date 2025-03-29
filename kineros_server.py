from flask import Flask, request, jsonify
import subprocess
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
KINEROS_PATH = "k2.exe"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'par', 'pre', 'out'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_until_prompt(process):
    """Read output until we detect a prompt (ends with ':')"""
    output = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output.append(line.strip())
        if line.strip().endswith(':'):
            break
    return output

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        required_params = [
            'parameter_file',
            'rainfall_file',
            'output_file',
            'description',
            'duration',
            'time_step',
            'adjust',
            'sediment',
            'multipliers'
        ]
        
        for param in required_params:
            if param not in data:
                return jsonify({"error": f"Missing required parameter: {param}"}), 400

        # Start KINEROS process
        process = subprocess.Popen(
            KINEROS_PATH,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Wait for and capture initial header
        header_output = []
        while len(header_output) < 5:  # Wait for the 5-line header
            line = process.stdout.readline()
            if not line:
                break
            header_output.append(line.strip())

        # Prepare responses and their corresponding prompts
        responses = [
            data['parameter_file'],
            data['rainfall_file'],
            data['output_file'],
            data['description'],
            str(data['duration']),
            str(data['time_step']),
            data['adjust'],
            data['sediment'],
            data['multipliers'],
            'y'  # Tabular Summary
        ]

        all_output = header_output
        
        # Process each response
        for response in responses:
            # Wait for and read the prompt
            prompt_output = read_until_prompt(process)
            all_output.extend(prompt_output)
            
            # Send the response
            process.stdin.write(f"{response}\n")
            process.stdin.flush()
            
            # Small delay to ensure proper timing
            time.sleep(0.1)

        # Get any remaining output
        stdout, stderr = process.communicate()
        all_output.extend([line.strip() for line in stdout.split('\n') if line.strip()])

        if process.returncode != 0:
            return jsonify({
                "error": "KINEROS execution failed",
                "stderr": stderr,
                "stdout": '\n'.join(all_output)
            }), 500

        # Check if output file was created
        output_file = data['output_file']
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                simulation_results = f.read()
        else:
            simulation_results = "Output file not found"

        return jsonify({
            "status": "success",
            "console_output": '\n'.join(all_output),
            "simulation_results": simulation_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)