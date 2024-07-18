def ntu(arr, angle):  # Normalize arrays to unity.
    for i in range(len(arr)):
            arr[i] = float(arr[i])
    arr_new = copy(arr)
    for i in range(len(arr_new)):
        arr_new[i] = max(0., arr_new[i])
    coeff = 1. / simps(arr_new, angle)
    result = copy(arr) * coeff
    
    return(result, coeff)
    
def get_shift(data):  # Get the value to shift the experimental data by using the off-peak cross sections.
    count = 0
    off_peak = []
    angle = data[:, 0]
    val = data[:, 1]
    
    for i in range(len(data)):
        if (angle[i] <= pi - 1.1) or (angle[i] >= pi + 1.1):
            if val[i] >= 0:
                off_peak.append(val[i])
                count += 1
            
    if len(off_peak) >= 4:
       off_peak[off_peak.index(min(off_peak))] = 0
       count -= 1
       off_peak[off_peak.index(max(off_peak))] = 0
       count -= 1
    
    if count <= 2:
        shift = 0
    else:
        shift = sum(off_peak) / count
        
    return(shift)
    
#!/usr/bin/env python3

import pandas as pd
from io import StringIO
import numpy as np
# Replace 'path/to/your/file.csv' with the actual path to your CSV file
file_path = 'Figure3g.csv'

# Read the CSV file into a list of lines
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize variables
data_blocks = []
current_data = ""
current_header = ""

# Process lines to split data blocks
for line in lines:
    if line.startswith("#:"):
        if current_data:
            data_blocks.append((current_header, current_data))
            current_data = ""
        current_header = line
    else:
        current_data += line

if current_data:  # Add the last data block
    data_blocks.append((current_header, current_data))

# Function to process and convert each data block to a DataFrame
def process_data_block(header, data):
    csv_io = StringIO(data)
    df = pd.read_csv(csv_io)
    return header, df

# Process all data blocks
data_frames = [process_data_block(header, data) for header, data in data_blocks]

# Display the DataFrames
ii = 0
for header, df in data_frames:
    print(header)
    print(df)
    np.savetxt("blue_point_{}".format(int(ii)), df)
    ii = ii +1
    
    
