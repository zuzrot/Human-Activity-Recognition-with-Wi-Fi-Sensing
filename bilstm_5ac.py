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
import datetime # Dodaj ten import do logowania (jeśli chcesz go później)
import os # Dodaj ten import do sprawdzania plików

# Konfiguracja 
SERIAL_PORT = '/dev/ttyACM0'  
# SERIAL_PORT = 'COM10'      # Dla Windows
BAUD_RATE = 115200
SEGMENT_LENGTH = 20
COLUMNS_TO_DROP = [2, 3, 4, 5, 32, 59, 60, 61, 62, 63]

MODEL_FILE = 'bilstm_model_3ak1.keras'
SCALER_FILE = 'bilstm_scaler_3ak1.pkl' 
MY_MAC_ADDRESS = '24:0A:C4:00:56:18' 

# Nazwy klas
label_names = ['chodzenie' , 'lezenie' , 'siedzenie'] 

# Sprawdzenie, czy pliki istnieją
# Poprawka: Użyj os.path.exists i obsłuż ogólny wyjątek
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    print(f"BŁĄD: Nie znaleziono pliku {MODEL_FILE} lub {SCALER_FILE}")
    print("Upewnij się, że pliki .keras i .pkl są w tym samym folderze co skrypt.")
    exit()

try:
    print("Ładowanie modelu i scalera...")
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Model i scaler załadowane pomyślnie.")
except Exception as e:
    print(f"BŁĄD: Nie można załadować plików. {e}")
    exit()


# Funkcje do przetwarzania danych (te same co w notatniku) 

def parse_csi_string(csi_string):
    """Przetwarza surowy string CSI na listę liczb całkowitych."""
    try:
        # Znajduje zawartość w nawiasach kwadratowych
        csi_raw_str = re.findall(r'\[(.*)\]', csi_string)[0]
        # Dzieli po spacjach i konwertuje na int, ignorując puste wpisy
        csi_raw = [int(x) for x in csi_raw_str.split(' ') if x]
        return csi_raw
    except Exception as e:
        # Zwraca None, jeśli parsowanie się nie powiedzie
        return None

def calculate_amplitude(csi_raw):
    """Oblicza amplitudę z surowych danych CSI."""
    imaginary, real, amplitudes = [], [], []
    for i, val in enumerate(csi_raw):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)
    
    # Upewnienie się, że obie listy mają tę samą długość
    for i in range(len(real)):
        try:
            # sqrt(i^2 + q^2)
            amplitudes.append(sqrt(imaginary[i]**2 + real[i]**2))
        except (ValueError, IndexError):
            amplitudes.append(0) 
    return amplitudes

def apply_noise_filters(df):
    """Stosuje filtry Hampel i Savitzky-Golay na ramce danych."""
    filtered_data = []
    df = df.astype(float) 
    for col in df.columns:
        col_series = df[col]
        hampel_filtered = hampel(col_series, window_size=10).filtered_data
        sg_filtered = savgol_filter(hampel_filtered, window_length=10, polyorder=3)
        filtered_data.append(sg_filtered)
    
    filtered_df = pd.DataFrame(np.array(filtered_data).T, columns=df.columns)
    return filtered_df



print("\nRozpoczynanie nasłuchu na porcie szeregowym...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Port {SERIAL_PORT} otwarty pomyślnie.")
except serial.SerialException as e:
    print(f"BŁĄD: Nie można otworzyć portu {SERIAL_PORT}.")
    exit()

all_amplitudes = []
print("Pętla główna rozpoczęta. Oczekiwanie na dane...")
print(f"Filtr aktywny: TYLKO pakiety z MAC = {MY_MAC_ADDRESS}")

while True:
    try:
        # Dodajemy 'errors='ignore'' dla bezpieczeństwa, tak jak w skrypcie zbierającym
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if line:
            
            #  LOGIKA FILTRUJĄCA 
            # 1. Sprawdzenie czy linia ma 1 nawias otwierający i 1 zamykający
            if line.count('[') == 1 and line.count(']') == 1:
                
                # 2. Podzielenie linii po przecinkach, aby sprawdzić metadane
                line_parts = line.split(',')
                
                # 3. Sprawdzenie czy to linia CSI, od AP, z odpowiednim MAC i ma poprawną długość
                if (len(line_parts) == 26 and          # Sprawdza długość danych
                    line_parts[0] == "CSI_DATA" and 
                    line_parts[1] == "AP" and 
                    line_parts[2] == MY_MAC_ADDRESS): 
                    
                    
                    # Jeśli linia przeszła filtr, przetwarzamy ją (stary kod)
                    csi_raw = parse_csi_string(line)
                    
                    # Sprawdzenie, czy parsowanie [..] dało 128 liczb
                    if csi_raw and len(csi_raw) == 128:
                        amplitude_data = calculate_amplitude(csi_raw)

                        # Sprawdzenie, czy obliczono 64 amplitudy
                        if len(amplitude_data) == 64:
                            all_amplitudes.append(amplitude_data)
                            print(".", end="", flush=True) #kropka, aby pokazać postęp

                            # Ten print jest teraz mniej przydatny przy SEGMENT_LENGTH = 20
                            # Możesz go usunąć lub zmienić warunek
                            if len(all_amplitudes) % 10 == 0:
                                print(f"  [{len(all_amplitudes)}/{SEGMENT_LENGTH}]")
                            
                            # Zebraliśmy pełny segment 20 próbek
                            # --- TO JEST KLUCZOWY IF ---
                            if len(all_amplitudes) == SEGMENT_LENGTH:
                                
                                # --- POCZĄTEK BLOKU PRZETWARZANIA (POPRAWNE WCIĘCIE) ---
                                print(f"\nZebrano segment {SEGMENT_LENGTH} próbek. Przetwarzanie...")

                                segment_df = pd.DataFrame(all_amplitudes, columns=range(64)) # (20, 64)

                                print("Krok 1: Stosowanie filtrów (Hampel, Savitzky-Golay)...")
                                denoised_df = apply_noise_filters(segment_df)

                                print("Krok 2: Usuwanie kolumn...")
                                trimmed_df = denoised_df.drop(columns=COLUMNS_TO_DROP, axis=1) # (20, 54)

                                print("Krok 3: Skalowanie i zmiana kształtu...")
                                n_samples, n_timesteps, n_features = 1, SEGMENT_LENGTH, trimmed_df.shape[1]
                                data_reshaped = trimmed_df.values.reshape((n_samples, n_timesteps * n_features))
                                scaled_data = scaler.transform(data_reshaped)
                                
                                final_input = scaled_data.reshape((n_samples, n_timesteps, n_features)) # (1, 20, 54)
                                
                                print("Krok 4: Predykcja modelem BiLSTM...")
                            
                                # --- ZMIANA: Logika predykcji dla 5 klas ---
                                
                                prediction_probs = model.predict(final_input)[0] 
                                
                                predicted_class_index = np.argmax(prediction_probs)
                                
                                confidence = prediction_probs[predicted_class_index] * 100
                                
                                predicted_class_name = label_names[predicted_class_index]
                                    
                                print(f"==========================================")
                                print(f"   AKTYWNOŚĆ: {predicted_class_name.upper()}   (Pewność: {confidence:.2f}%)")
                                print(f"==========================================")
                                
                                # --- TUTAJ MOŻESZ DODAĆ LOGOWANIE DO PLIKU ---
                                # try:
                                #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                                #     with open("log_testu_na_zywo.csv", "a") as log_file:
                                #         log_file.write(f"{timestamp},{predicted_class_name},{confidence:.2f}\n")
                                # except Exception as e:
                                #     print(f"Błąd podczas zapisywania logu: {e}")
                                # --- KONIEC LOGOWANIA ---
                                
                                print("\nOczekiwanie na dane do nowego segmentu...")
                                
                                all_amplitudes = [] # Wyczyść bufor
                                # --- KONIEC BLOKU PRZETWARZANIA ---

            
    except KeyboardInterrupt:
        print("\nZamykanie programu.")
        break
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd w pętli głównej: {e}")
        traceback.print_exc()
        print("Resetowanie bufora...")
        all_amplitudes = [] # Wyczyść bufor w razie błędu

print("Zamykanie portu szeregowego.")
ser.close()