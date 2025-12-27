import json
from averbis import Client

# --- Configuration ----
client_url = 'https://ahd.test.imi-frankfurt.de/health-discovery/'
api_token = '12257b50f157bb0e2ebc4736e14d2263cc9a3ac7b9ce5579df7d48881523f39c'
project_name = 'Evaluation Medikation'
process_name = 'Analysis_Run_01' # Must match the name used in the analysis script
output_file = 'analysis_results.json'
# ----------------------

client = Client(client_url, api_token=api_token)
project = client.get_project(project_name)

processes = project.list_processes()

results = processes[0].export_text_analysis()

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
