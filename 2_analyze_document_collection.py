import time
from averbis import Client

# --- Configuration ---
client_url = 'https://ahd.test.imi-frankfurt.de/health-discovery/'
api_token = '12257b50f157bb0e2ebc4736e14d2263cc9a3ac7b9ce5579df7d48881523f39c'
project_name = 'Evaluation Medikation'
document_collection_name = 'Datensatz'
pipeline_name = 'Medication_1' 
process_name = 'Analysis_Run_01'
# ----------------------

# Initialize Client and Project
client = Client(client_url, api_token=api_token)
project = client.get_project(project_name)

# Get existing resources
document_collection = project.get_document_collection(document_collection_name)
pipeline = project.get_pipeline(pipeline_name)

# Ensure pipeline is active before running
print(f"Ensuring pipeline '{pipeline_name}' is started...")
pipeline.ensure_started()

print(f"Starting process '{process_name}' on collection '{document_collection_name}'...")

# Create and run the analysis process
process = document_collection.create_and_run_process(process_name=process_name, pipeline=pipeline)

# Monitor the process state
while process.get_process_state().state == "PROCESSING":
    print(f"Process state: {process.get_process_state().state}...", end='\r')
    time.sleep(1)

final_state = process.get_process_state().state
print(f"\nProcess finished with state: {final_state}")
    