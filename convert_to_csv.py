import json
import csv
import os

# --- Configuration ---
input_json = 'medications.json'
output_csv = 'medications.csv'
# ----------------------

if not os.path.exists(input_json):
    print(f"Error: {input_json} not found.")
else:
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare to write to CSV
    # We use semicolon (;) because it opens better in German Excel versions
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        
        # Write Header
        writer.writerow(['filename', 'medication'])
        
        count = 0
        for entry in data:
            full_path = entry.get('document', '')
            medication = entry.get('medication_text', '')
            
            # Logic: Strip the path, keep only the filename
            filename = os.path.basename(full_path)
            
            writer.writerow([filename, medication])
            count += 1

    print(f"Successfully converted {count} entries.")
    print(f"File saved as: {output_csv}")
    