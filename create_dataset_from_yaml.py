import mlflow
from mlflow.genai.datasets import create_dataset
import yaml
from pathlib import Path
import os
import json
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', ''))
MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT', '')
mlflow.set_experiment(MLFLOW_EXPERIMENT)
API_URL = os.environ.get('SEMBIIQ_URL', '')
RAG_API_URL = os.environ.get('RAG_SEMBIIQ_URL', '')
API_KEY = os.environ.get('SEMBIIQ_API_KEY', '')

# =========================================================
# RAG Upload Helper
# =========================================================
def upload_to_rag(file_entry):
    """
    Uploads one file to the RAG system and returns its document ID.
    Expected file_entry fields:
        - path (local file path)
        - mime_type (e.g., 'application/pdf')
        - delete_after_days (int)
    """
    if not RAG_API_URL or not API_KEY:
        raise RuntimeError("RAG API URL or key not set in environment variables.")

    headers = {
        'x-api-key': os.environ.get('SEMBIIQ_API_KEY', ''),
        'x-brand': os.environ.get('SEMBIIQ_BRAND', 'XRAY'),
        'x-product': os.environ.get('SEMBIIQ_PRODUCT', 'XRAY'),
        'x-customer': os.environ.get('SEMBIIQ_CUSTOMER', 'CC_XRAY'),
        'x-provider': os.environ.get('SEMBIIQ_PROVIDER', None),
        'x-llm-key': os.environ.get('SEMBIIQ_LLM_KEY', None),
        'x-model-id': os.environ.get('SEMBIIQ_MODEL_ID', None),
        'x-advanced-tracing': str(os.environ.get('SEMBIIQ_ADVANCED_TRACING', 'false')).lower(),
    }

    payload = {
        "content_type": file_entry.get("mime_type", ""),
        "delete_after_days": file_entry.get("delete_after_days", 90),
    }
    
    # Step 1: Request presigned upload URL
    resp = requests.post(RAG_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    presigned_url = data.get("presigned_url")
    
    if not presigned_url:
        raise ValueError("RAG response did not include a presigned_url.")

    # Step 2: Upload file bytes to S3
    path = file_entry.get("path")
    mime_type = file_entry.get("mime_type", "")
    fields = presigned_url.get("fields")
    fields["Content-Type"] = mime_type

    #print(f"PRESIGNEDURL: {presigned_url}")
    #print(f"FILE_PATH: {path}")

    with open(path, "rb") as f:
        ffiles = {"file": (os.path.basename(path), f, mime_type)}
        put_resp = requests.post(presigned_url.get("url"), data=fields, files=ffiles)
        print(f"POST_RESPONSE: {put_resp}")
        put_resp.raise_for_status()

    # Step 3: Extract doc_id from RAG response (or fallback to file name)
    doc_id = data.get("doc_id") or Path(path).stem
    print(f"✅ Uploaded {path} → doc_id={doc_id}")
    return doc_id

# =========================================================
# YAML Loader with Multi-File Upload
# =========================================================
def upload_files_for_case(uploaded_files):
    """
    Uploads multiple files for a single test case.
    Returns list of doc_ids.
    """
    doc_ids = []
    if not uploaded_files:
        return doc_ids

    # Optional parallel upload for speed
    with ThreadPoolExecutor(max_workers=min(4, len(uploaded_files))) as executor:
        futures = {}
        for file_entry in uploaded_files:
            path = file_entry.get("path")
            if not path or not os.path.exists(path):
                print(f"⚠️ File not found, skipping: {path}")
                continue
            futures[executor.submit(upload_to_rag, file_entry)] = path

        for future in as_completed(futures):
            try:
                doc_id = future.result()
                doc_ids.append(doc_id)
            except Exception as e:
                print(f"❌ Upload failed for {futures[future]}: {e}")

    return doc_ids

def upload_files_for_case_sync(uploaded_files):
    """
    Uploads multiple files for a single test case sequentially.
    Returns list of doc_ids.
    """
    doc_ids = []
    if not uploaded_files:
        return doc_ids

    for file_entry in uploaded_files:
        path = file_entry.get("path")
        if not path or not os.path.exists(path):
            print(f"⚠️ File not found, skipping: {path}")
            continue

        try:
            doc_id = upload_to_rag(file_entry)
            doc_ids.append(doc_id)
        except Exception as e:
            print(f"❌ Upload failed for {path}: {e}")

    return doc_ids

def load_yaml_files(dataset_folder='new_datasets'):
    dataset_folder = Path(dataset_folder)
    cases = []
    for f in sorted(dataset_folder.glob('*.yaml')):
        with open(f, 'r') as stream:
            data = yaml.safe_load(stream)
            #print('Data', data)
            if 'inputs' in data:
                test_data = data['inputs']
            else:
                test_data = data
        
        context = test_data.get('context', {})
        uploaded_files = context.get("uploaded_files", [])
        doc_ids = context.get("doc_ids", [])

        # Upload multiple files (if provided)
        if uploaded_files:
            print(f"📂 Uploading {len(uploaded_files)} file(s) for {f.name}...")
            new_doc_ids = upload_files_for_case_sync(uploaded_files)
            doc_ids.extend(new_doc_ids)
            context.pop("uploaded_files", None)
            context["doc_ids"] = sorted(set(doc_ids))

        record = {
            'inputs': {
                'chat_id': test_data.get('chat_id'),
                'language': test_data.get('language'),
                'testing_framework': test_data.get('testing_framework'),
                'user_query': test_data.get('user_query'),
                'context': context,
                'current_code': test_data.get('current_code', ''),
                'current_imports': test_data.get('current_imports', ''),
                'current_config_file': test_data.get('current_config_file', ''),
                'test_case': test_data.get('test_case', {}),
                'thinking_budget': test_data.get('thinking_budget', 0),
            },
            'expectations': data.get('expectations', {})
        }
        cases.append(record)

    return cases

def create_dataset_from_yaml(dataset_name='ai_script_test_suite', folder='new_datasets'):
    records = load_yaml_files(folder)
    if not records:
        print('No YAML files found in', folder)
        return
    dataset = create_dataset(
        name=dataset_name,
        tags={'type': 'regression_suite', 'source': 'yaml'},
    )
    dataset.merge_records(records)
    print(f"Created dataset '{dataset.name}' with {len(records)} records.")
    print(f"Dataset created with ID: {dataset.dataset_id}")
    with open("dataset_info.json", "w") as f:
        json.dump({"dataset_id": dataset.dataset_id}, f)

if __name__ == '__main__':
    create_dataset_from_yaml()
