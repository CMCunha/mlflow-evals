import mlflow
import json
import requests
from pathlib import Path
from datetime import datetime
import tempfile
import os
from validator import evaluate
from mlflow.entities import SpanType
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', ''))
MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT', '')
API_URL = os.environ.get('SEMBIIQ_URL', '')

API_HEADERS = {
    'x-api-key': os.environ.get('SEMBIIQ_API_KEY', ''),
    'x-brand': os.environ.get('SEMBIIQ_BRAND', 'XRAY'),
    'x-product': os.environ.get('SEMBIIQ_PRODUCT', 'XRAY'),
    'x-customer': os.environ.get('SEMBIIQ_CUSTOMER', 'XRAY'),
    'x-provider': os.environ.get('SEMBIIQ_PROVIDER', None),
    'x-llm-key': os.environ.get('SEMBIIQ_LLM_KEY', None),
    'x-model-id': os.environ.get('SEMBIIQ_MODEL_ID', None),
    'x-advanced-tracing': str(os.environ.get('SEMBIIQ_ADVANCED_TRACING', 'false')).lower(),
}
@mlflow.trace(span_type=SpanType.TOOL)
def call_ai_api(record):
    payload = {
        'chat_id': record.inputs.get('chat_id'),
        'language': record.inputs.get('language'),
        'testing_framework': record.inputs.get('testing_framework'),
        'user_query': record.inputs.get('user_query'),
        'context': record.inputs.get('context', {}),
        'current_code': record.inputs.get('current_code', ''),
        'current_imports': record.inputs.get('current_imports', ''),
        'current_config_file': record.inputs.get('current_config_file', ''),
        'test_case': record.inputs.get('test_case', {}),
        'thinking_budget': record.inputs.get('thinking_budget', 0),
    }
    headers = {k: v for k, v in API_HEADERS.items() if v}
    mlflow.log_text(json.dumps(payload), artifact_file='ai_api_payload.py')
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    #print(f'RESPONSE: {resp.json()}')
    resp.raise_for_status()
    response_json = resp.json()
    return response_json #.get('code') or response_json.get('script') or response_json.get('data') or ''

def run_generation(dataset_name='ai_script_test_suite'):
    with open("dataset_info.json") as f:
        info = json.load(f)

    dataset_id = info["dataset_id"]
    ds = mlflow.genai.datasets.get_dataset(dataset_id=dataset_id)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for record in ds.records:
        run_name = record.inputs.get('test_case', {}).get('title') or record.inputs.get('chat_id') or 'unnamed'
        with mlflow.start_run(run_name=run_name):
            try:
                api_response = call_ai_api(record)
                generated_script = api_response.get('code','') 
                generated_imports = api_response.get('imports', '')
                generated_config = api_response.get('config_file', '')
                
                gen_file = Path(tempfile.mktemp(suffix='.py'))
                gen_file.write_text(generated_script)

                gen_imports = Path(tempfile.mktemp(suffix='.py'))
                gen_imports.write_text(generated_imports)

                gen_config = Path(tempfile.mktemp(suffix='.py'))
                gen_config.write_text(generated_config)

                expected_script = record.expectations.get('code', '')
                exp_file = Path(tempfile.mktemp(suffix='.py'))
                exp_file.write_text(expected_script)

                if expected_script.strip():
                    result = evaluate(record.inputs, gen_file, gen_imports, gen_config, baseline_code=exp_file)
                    #print(f'Evaluate ', result)
                    mlflow.log_metric('overall_weighted_score', result['overall_weighted_score'])
                    mlflow.log_dict(result, artifact_file='validation_result.json')
                    #print('SEMANTIC ', result.get('code','').get['semantic_diff_text'])
                    mlflow.log_text(result['code']['semantic_diff_text'], artifact_file='semantic_diff.txt')

                mlflow.log_params({
                    'language': record.inputs.get('language'),
                    'framework': record.inputs.get('testing_framework'),
                })
                mlflow.log_text(generated_script, artifact_file='generated_script.py')
                mlflow.log_text(generated_imports, artifact_file='generated_imports.py')
                mlflow.log_text(generated_config, artifact_file='generated_config.py')
                mlflow.log_param('status', 'success')
                print(f'Completed {run_name}')

            except Exception as e:
                mlflow.log_param('status', 'failed')
                mlflow.log_text(e.response.text, artifact_file='api_error_response.txt')
                #mlflow.log_text(str(e), artifact_file='api_error.txt')
                print(f'Error on {run_name}: {e}')

if __name__ == '__main__':
    run_generation()
