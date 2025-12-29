import json
import csv
import os

# --- Configuration ----
input_json = 'medications.json'
output_csv = 'medications.csv'
# ----------------------

input_path = os.path.join('Averbis_annotationen', input_json)
output_path = os.path.join('Averbis_annotationen', output_csv)

if not os.path.exists(input_path):
    print(f"Error: {input_json} not found.")
else:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare to write to CSV
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as csvfile:
        # Keeping delimiter as ';' based on your previous file
        writer = csv.writer(csvfile, delimiter=';')
        
        # Write Header (Added 'confidence')
        writer.writerow(['filename', 'medication', 'confidence'])
        
        count = 0
        for entry in data:
            full_path = entry.get('document', '')
            medication = entry.get('medication_text', '')
            # Extract the confidence value
            confidence = entry.get('confidence', '')
            
            # Logic: Strip the path, keep only the filename
            filename = os.path.basename(full_path)
            
            # Write row with the new column
            writer.writerow([filename, medication, confidence])
            count += 1

    print(f"Successfully converted {count} entries.")
    print(f"File saved as: {output_csv}")