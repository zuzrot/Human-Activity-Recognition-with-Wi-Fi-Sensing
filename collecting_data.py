import serial  
import csv    
import time    


COM_PORT = '/dev/ttyACM0'  
BAUD_RATE = 115200
CSV_FILENAME = 'lezenie2.csv' 


print(f"Łączenie z portem {COM_PORT} przy {BAUD_RATE} bps...")

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        print(f"Zapisywanie czystych danych CSI do pliku: {CSV_FILENAME}")
        print("Rozpoczęto nasłuchiwanie... (Naciśnij CTRL+C, aby zakończyć)")
        
        print_line = False

        while True:
            try:
             
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if line:
                    if not print_line:
                        print(f"Odebrano pierwszą linię: {line}")
                        print_line = True
                    if line.count('[') == 1 and line.count(']') == 1:                   
                        line_parts = line.split(',')                
                        CSI_data_str = line_parts[-1].replace('[', '').replace(']', '').split(' ')
                        
                        if (line_parts[0] == "CSI_DATA" and 
                            line_parts[1] == "AP" and 
                            line_parts[2] == "24:0A:C4:00:56:18" and  
                            len(line_parts) == 26 and 
                            len(CSI_data_str) == 129):
                            
                            writer.writerow([field.strip() for field in line_parts])


            except serial.SerialException as e:
                print(f"Błąd portu szeregowego: {e}")
                break
            except UnicodeDecodeError:
                pass

except serial.SerialException as e:
    print(f"BŁĄD: Nie można otworzyć portu {COM_PORT}. Sprawdź, czy jest podłączony.")

except KeyboardInterrupt:
    print("\nZatrzymano zbieranie danych.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print(f"Zamknięto port {COM_PORT}.")
    print(f"Dane zostały zapisane w {CSV_FILENAME}")