import serial
import numpy as np
import pandas as pd
import re
from math import sqrt
from scipy.signal import savgol_filter
from hampel import hampel
import tensorflow as tf
import joblib
import traceback
import datetime 
import os 


SERIAL_PORT = '/dev/ttyACM0'  
# SERIAL_PORT = 'COM10'     
BAUD_RATE = 115200
SEGMENT_LENGTH = 20
COLUMNS_TO_DROP = [2, 3, 4, 5, 32, 59, 60, 61, 62, 63]

MODEL_FILE = 'bilstm_model_3ak1.keras'
SCALER_FILE = 'bilstm_scaler_3ak1.pkl' 
MY_MAC_ADDRESS = '24:0A:C4:00:56:18' 


label_names = ['no activity', 'walking', 'lying', 'sitting', 'fidgeting'] 


if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
print(f"ERROR: Could not find the file {MODEL_FILE} or {SCALER_FILE}")
print("Make sure the .keras and .pkl files are in the same folder as the script.")
    exit()

try:
print("Loading the model and scaler...")
model = tf.keras.models.load_model(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"ERROR: Unable to load the files. {e}")
    exit()




def parse_csi_string(csi_string):
    """Parses a raw CSI string into a list of integers."""
    try:
        
        csi_raw_str = re.findall(r'\[(.*)\]', csi_string)[0]
        csi_raw = [int(x) for x in csi_raw_str.split(' ') if x]
        return csi_raw
    except Exception as e:
        return None

def calculate_amplitude(csi_raw):
    """Computes the amplitude from raw CSI data."""
    imaginary, real, amplitudes = [], [], []
    for i, val in enumerate(csi_raw):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    for i in range(len(real)):
        try:
            # sqrt(i^2 + q^2)
            amplitudes.append(sqrt(imaginary[i]**2 + real[i]**2))
        except (ValueError, IndexError):
            amplitudes.append(0) 
    return amplitudes

def apply_noise_filters(df):
    """Applies Hampel and Savitzky-Golay filters to a data frame."""
    filtered_data = []
    df = df.astype(float) 
    for col in df.columns:
        col_series = df[col]
        hampel_filtered = hampel(col_series, window_size=10).filtered_data
        sg_filtered = savgol_filter(hampel_filtered, window_length=10, polyorder=3)
        filtered_data.append(sg_filtered)
    
    filtered_df = pd.DataFrame(np.array(filtered_data).T, columns=df.columns)
    return filtered_df



print("\nStarting to listen on the serial port...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Port {SERIAL_PORT} opened successfully.")
except serial.SerialException as e:
    print(f"ERROR: Unable to open port {SERIAL_PORT}.")
    exit()

all_amplitudes = []
print("Main loop started. Waiting for data...")
print(f"Active filter: ONLY packets with MAC = {MY_MAC_ADDRESS}")

while True:
    try:
        
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if line:
          
            if line.count('[') == 1 and line.count(']') == 1:
            
                line_parts = line.split(',')
               
                if (len(line_parts) == 26 and          
                    line_parts[0] == "CSI_DATA" and 
                    line_parts[1] == "AP" and 
                    line_parts[2] == MY_MAC_ADDRESS): 
               
                    csi_raw = parse_csi_string(line)
              
                    if csi_raw and len(csi_raw) == 128:
                        amplitude_data = calculate_amplitude(csi_raw)

                        if len(amplitude_data) == 64:
                            all_amplitudes.append(amplitude_data)
                            print(".", end="", flush=True) 

                            if len(all_amplitudes) % 10 == 0:
                                print(f"  [{len(all_amplitudes)}/{SEGMENT_LENGTH}]")

                            if len(all_amplitudes) == SEGMENT_LENGTH:
                                
                                print(f"\nCollected a segment of {SEGMENT_LENGTH} samples. Processing...")

                                segment_df = pd.DataFrame(all_amplitudes, columns=range(64))  # (20, 64)

                                print("Step 1: Applying filters (Hampel, Savitzky–Golay)...")
                                denoised_df = apply_noise_filters(segment_df)

                                print("Step 2: Dropping columns...")
                                trimmed_df = denoised_df.drop(columns=COLUMNS_TO_DROP, axis=1)  # (20, 54)

                                print("Step 3: Scaling and reshaping...")
                                n_samples, n_timesteps, n_features = 1, SEGMENT_LENGTH, trimmed_df.shape[1]
                                data_reshaped = trimmed_df.values.reshape((n_samples, n_timesteps * n_features))
                                scaled_data = scaler.transform(data_reshaped)

                                final_input = scaled_data.reshape((n_samples, n_timesteps, n_features))  # (1, 20, 54)

                                print("Step 4: BiLSTM model prediction...")

                            
                                
                                prediction_probs = model.predict(final_input)[0] 
                                
                                predicted_class_index = np.argmax(prediction_probs)
                                
                                confidence = prediction_probs[predicted_class_index] * 100
                                
                                predicted_class_name = label_names[predicted_class_index]
                                    
                                print("==========================================")
                                print(f"   ACTIVITY: {predicted_class_name.upper()}   (Confidence: {confidence:.2f}%)")
                                print("==========================================")                            
                                print("\nWaiting for data for a new segment...")
                                
                                all_amplitudes = [] 
                                

            
    except KeyboardInterrupt:
        break
    except Exception as e:
        traceback.print_exc()
        all_amplitudes = [] 

print("Closing the serial port.")
ser.close()