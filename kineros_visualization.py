import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_kineros_output(content):
    """Parse KINEROS2 output file content and extract time series data."""
    # Split content into lines
    lines = content.split('\n')
    
    data = []
    started = False
    
    # Parse the time series data
    for line in lines:
        if 'Elapsed Time' in line:
            started = True
            continue
        if started and line.strip():
            # Check if line contains numerical data
            try:
                values = line.split()
                if len(values) >= 5:
                    data.append({
                        'time': float(values[0]),
                        'rainfall': float(values[1]),
                        'outflow': float(values[2]),
                        'outflow_cms': float(values[3]),
                        'sediment': float(values[4])
                    })
            except (ValueError, IndexError):
                if 'Channel' in line:
                    break
    
    return pd.DataFrame(data)

def create_visualizations(df):
    """Create visualizations of key runoff metrics."""
    # Set up the plot style
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Rainfall-Runoff Comparison
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['time'], df['rainfall'], 'b-', label='Rainfall', alpha=0.6)
    ax1.plot(df['time'], df['outflow'], 'r-', label='Runoff', linewidth=2)
    ax1.set_title('Rainfall-Runoff Relationship')
    ax1.set_ylabel('Rate (mm/hr)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Cumulative Volume
    ax2 = plt.subplot(3, 1, 2)
    # Calculate cumulative volumes (converting mm/hr to volume)
    time_step = df['time'].diff().fillna(1/60)  # Convert to hours
    rainfall_vol = (df['rainfall'] * time_step).cumsum()
    runoff_vol = (df['outflow'] * time_step).cumsum()
    
    ax2.plot(df['time'], rainfall_vol, 'b-', label='Cumulative Rainfall')
    ax2.plot(df['time'], runoff_vol, 'r-', label='Cumulative Runoff')
    ax2.set_title('Cumulative Volume')
    ax2.set_ylabel('Volume (mm)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Sediment Transport
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df['time'], df['sediment'], 'g-', label='Sediment Transport')
    ax3.fill_between(df['time'], df['sediment'], alpha=0.3, color='g')
    ax3.set_title('Sediment Transport Rate')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Sediment Transport Rate (kg/s)')
    ax3.legend()
    ax3.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate and display key metrics
    peak_rainfall = df['rainfall'].max()
    peak_runoff = df['outflow'].max()
    peak_sediment = df['sediment'].max()
    total_rainfall = rainfall_vol.iloc[-1]
    total_runoff = runoff_vol.iloc[-1]
    
    metrics = f"""
    Key Metrics:
    Peak Rainfall: {peak_rainfall:.2f} mm/hr
    Peak Runoff: {peak_runoff:.2f} mm/hr
    Runoff Coefficient: {(total_runoff/total_rainfall*100):.1f}%
    Peak Sediment Transport: {peak_sediment:.2f} kg/s
    Total Rainfall Volume: {total_rainfall:.1f} mm
    Total Runoff Volume: {total_runoff:.1f} mm
    """
    
    return fig, metrics

# Example usage
def analyze_kineros_output(content):
    """Main function to analyze KINEROS2 output."""
    # Parse the data
    df = parse_kineros_output(content)
    
    # Create visualizations and get metrics
    fig, metrics = create_visualizations(df)
    
    return fig, metrics, df

filepath = "EX1.OUT"  # Replace with your actual file path
with open(filepath, 'r') as f:
    content = f.read()

# Analyze the output
fig, metrics, df = analyze_kineros_output(content)

# Display the metrics
print(metrics)

# Show the plots
plt.show()