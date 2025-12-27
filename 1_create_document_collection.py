import os
from averbis import Client

# --- Configuration ----
client = 'https://ahd.test.imi-frankfurt.de/health-discovery/'
api_token = '12257b50f157bb0e2ebc4736e14d2263cc9a3ac7b9ce5579df7d48881523f39c'
project_name = 'Evaluation Medikation'
document_collection_name = 'Datensatz'
data_origin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datensatz')
# ----------------------

client = Client(client, api_token=api_token)
project = client.get_project(project_name)
document_collection = project.create_document_collection(document_collection_name)

# Check if the folder exists to avoid errors
if os.path.exists(data_origin):
    # Get all files in the directory
    files = [f for f in os.listdir(data_origin) if f.endswith('.txt')]
    
    print(f"Found {len(files)} text files. Starting import...")

    for filename in files:
        file_path = os.path.join(data_origin, filename)
        
        with open(file_path, "r", encoding="UTF-8") as input_io:
            document_collection.import_documents(input_io)
            print(f"Imported (UTF-8): {filename}")
    
else:
    print(f"Error: The folder '{data_origin}' does not exist.")

print(f"Total documents in collection: {document_collection.get_number_of_documents()}")
