import os
import json
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langfuse import get_client

# =========================================================
# Setup
# =========================================================
load_dotenv()
langfuse = get_client()

API_KEY = os.environ.get('SEMBIIQ_API_KEY', '')
RAG_API_URL = os.environ.get('RAG_SEMBIIQ_URL', '')

# =========================================================
# RAG Upload Helper
# =========================================================
def upload_to_rag(file_entry):
    """
    Uploads a file to the RAG API and returns its document ID.
    Expected fields:
      - path (local file path)
      - mime_type (e.g., 'application/pdf')
      - delete_after_days (int)
    """
    if not RAG_API_URL or not API_KEY:
        raise RuntimeError("RAG API URL or API key not set in environment variables.")

    headers = {
        'x-api-key': API_KEY,
        'x-brand': os.environ.get('SEMBIIQ_BRAND', 'XRAY'),
        'x-product': os.environ.get('SEMBIIQ_PRODUCT', 'XRAY'),
        'x-customer': os.environ.get('SEMBIIQ_CUSTOMER', 'XRAY'),
    }

    payload = {
        "content_type": file_entry.get("mime_type", "application/pdf"),
        "delete_after_days": file_entry.get("delete_after_days", 90),
    }

    # Step 1: request presigned upload URL
    resp = requests.post(RAG_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    print(f"✅ PRESIGNED RESPONSE: {data}")

    presigned_url = data.get("presigned_url")
    if not presigned_url:
        raise ValueError("RAG API response missing presigned_url.")

    # Step 2: upload file bytes
    path = file_entry["path"]
    fields = data.get("fields")
    mime_type = file_entry.get("mime_type", "application/pdf")
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, mime_type)}
        put_resp = requests.post(presigned_url, data=fields, files=files)
        put_resp.raise_for_status()

    # Step 3: extract doc_id
    doc_id = data.get("doc_id") or Path(path).stem
    print(f"✅ Uploaded {path} → doc_id={doc_id}")
    return doc_id

# =========================================================
# Multi-file uploader
# =========================================================
def upload_files_for_case(uploaded_files):
    """
    Uploads multiple files concurrently for a test case.
    Returns a list of document IDs.
    """
    if not uploaded_files:
        return []

    doc_ids = []
    with ThreadPoolExecutor(max_workers=min(4, len(uploaded_files))) as executor:
        futures = {}
        for f in uploaded_files:
            path = f.get("path")
            if not path or not os.path.exists(path):
                print(f"⚠️ File not found, skipping: {path}")
                continue
            futures[executor.submit(upload_to_rag, f)] = path

        for future in as_completed(futures):
            try:
                doc_ids.append(future.result())
            except Exception as e:
                print(f"❌ Upload failed for {futures[future]}: {e}")

    return doc_ids

# =========================================================
# YAML Loader
# =========================================================
def load_yaml_files(dataset_folder='new_datasets'):
    dataset_folder = Path(dataset_folder)
    cases = []

    for f in sorted(dataset_folder.glob('*.yaml')):
        with open(f, 'r') as stream:
            data = yaml.safe_load(stream)

        # Extract test input block
        test_data = data.get('inputs', data)
        context = test_data.get('context', {})
        uploaded_files = context.get("uploaded_files", [])
        doc_ids = context.get("doc_ids", [])

        # Upload files (if any)
        if uploaded_files:
            print(f"📂 Uploading {len(uploaded_files)} file(s) for {f.name}...")
            new_doc_ids = upload_files_for_case(uploaded_files)
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
            'expectations': data.get('expectations', {}),
            'file_name': f.name
        }

        cases.append(record)

    return cases

# =========================================================
# Create Langfuse Dataset
# =========================================================
def create_dataset_from_yaml(dataset_name='ai_script_test_suite', folder='new_datasets'):
    records = load_yaml_files(folder)
    if not records:
        print(f'No YAML files found in {folder}')
        return

    print(f"🚀 Creating Langfuse dataset: {dataset_name} ...")
    langfuse.create_dataset(
        name=dataset_name,
        description="Regression suite imported from YAML test cases",
        metadata={"source": "yaml", "num_records": len(records)}
    )

    # Add items to dataset
    for idx, record in enumerate(records):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=record['inputs'],
            expected_output=record.get('expectations', {}),
            metadata={
                "file": record.get('file_name'),
                "index": idx,
                "has_docs": bool(record['inputs'].get('context', {}).get('doc_ids'))
            }
        )
        print(f"   ✅ Added record {idx+1}/{len(records)}: {record.get('file_name')}")

    print(f"🎉 Created Langfuse dataset '{dataset_name}' with {len(records)} items.")

    # Save dataset info locally
    with open("dataset_info.json", "w") as f:
        json.dump({"dataset_name": dataset_name, "num_items": len(records)}, f)

if __name__ == '__main__':
    create_dataset_from_yaml()
