import requests

url = 'http://localhost:5000/run_simulation'
data = {
    'parameter_file': 'EX1.PAR',
    'rainfall_file': 'EX1.PRE',
    'output_file': 'EX1.OUT',
    'description': 'Example Data',
    'duration': 200,
    'time_step': 1,
    'adjust': 'n',
    'sediment': 'y',
    'multipliers': 'n'
}

# If you need to upload files with the request:

response = requests.post(url, json=data)
print(response.json())