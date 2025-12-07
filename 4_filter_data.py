import json
import os
import re

# --- Configuration ---
input_file = 'analysis_results.json'
output_file = 'medications.json'
# ----------------------

def natural_sort_key(s):
    """
    Key function for natural sorting. 
    Splits string into text and numbers so '2.txt' comes before '10.txt'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please run the export script first.")
else:
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted_medications = []
    medication_count = 0

    # Iterate through each document in the results
    # The JSON structure shows 'textAnalysisResultDtos' contains the list of documents
    for document in data.get('textAnalysisResultDtos', []):
        doc_name = document.get('documentName', 'Unknown Document')
        
        # Iterate through all annotations in the document
        for annotation in document.get('annotationDtos', []):
            annotation_type = annotation.get('type', '')
            
            # Filter logic: Check if the type contains "Medication"
            # Standard type is usually 'de.averbis.types.health.Medication'
            if 'Medication' in annotation_type:
                
                # Get and clean the covered text
                raw_text = annotation.get('coveredText', '')
                if raw_text:
                    # Replace newline/tabs with space, then strip leading/trailing whitespace
                    clean_text = raw_text.replace('\n', ' ').replace('\r', '').replace('\t', ' ').strip()
                    # Remove any double spaces created by the replacement
                    clean_text = " ".join(clean_text.split())
                else:
                    clean_text = ""

                # Create a simplified object for the medication
                med_entry = {
                    'document': doc_name,
                    'medication_text': clean_text,
                    'standardized_name': annotation.get('dictCanon'), # The dictionary canonical name
                    'concept_id': annotation.get('conceptID'),        # ATC or ID
                    'begin': annotation.get('begin'),
                    'end': annotation.get('end'),
                    'confidence': annotation.get('confidence') # Some types might not have this, but good to check
                }
                
                extracted_medications.append(med_entry)
                medication_count += 1

    # Sort the results using natural sorting on the 'document' field
    # This fixes the 1, 10, 2 issue
    extracted_medications.sort(key=lambda x: natural_sort_key(x['document']))

    # Save the filtered list to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_medications, f, ensure_ascii=False, indent=4)

    print(f"Done! Found {medication_count} medications.")
    print(f"Results sorted and saved to: {output_file}")
    