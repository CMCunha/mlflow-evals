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
from mlflow_utils import log_evaluation_to_mlflow

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
#@mlflow.trace(span_type=SpanType.TOOL)
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
        # Start a trace/span in the current run
    with mlflow.start_span("AI_API_Call") as span:
        mlflow.log_text(json.dumps(payload, indent=2), artifact_file="api_payload.json")
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        mlflow.log_text(f"HTTP status: {resp.status_code}", artifact_file="api_status.txt")
        mlflow.log_text(resp.text, artifact_file="api_response.json")
        resp.raise_for_status()
        return resp.json()
    #    mlflow.log_text(json.dumps(payload), artifact_file='ai_api_payload.py')
    #resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    #resp.raise_for_status()
    #return resp.json() 

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

                # Load expected outputs
                expected_script = record.expectations.get('code', '')
                expected_imports = record.expectations.get('imports', '')
                expected_config = record.expectations.get('config', '')

                # Write temp files for baseline
                exp_script_file = Path(tempfile.mktemp(suffix='.py'))
                exp_script_file.write_text(expected_script)
                exp_imports_file = Path(tempfile.mktemp(suffix='.py'))
                exp_imports_file.write_text(expected_imports)
                exp_config_file = Path(tempfile.mktemp(suffix='.py'))
                exp_config_file.write_text(expected_config)

                if expected_script.strip():
                    result = evaluate(record.inputs, gen_file, gen_imports, gen_config, baseline_code=exp_script_file,baseline_imports=exp_imports_file,baseline_config=exp_config_file)

                    # Build structured records for MLflow
                    evaluation_records = [
                        {
                            "chat_id": record.inputs.get('chat_id'),
                            "section": "code",
                            "expected": expected_script,
                            "generated": generated_script,
                            "score": result["code"]["functional"]["score"],
                            "semantic_score": result["code"]["functional"]["score"],
                            "language": record.inputs.get("language"),
                            "framework": record.inputs.get("testing_framework"),
                        },
                        {
                            "chat_id": record.inputs.get('chat_id'),
                            "section": "imports",
                            "expected": expected_imports,
                            "generated": generated_imports,
                            "score": result["imports"]["score"],
                            "semantic_score": result["imports"]["score"],
                            "language": record.inputs.get("language"),
                            "framework": record.inputs.get("testing_framework"),
                        },
                        {
                            "chat_id": record.inputs.get('chat_id'),
                            "section": "config",
                            "expected": expected_config,
                            "generated": generated_config,
                            "score": result["config"]["score"],
                            "semantic_score": result["config"]["score"],
                            "language": record.inputs.get("language"),
                            "framework": record.inputs.get("testing_framework"),
                        },
                    ]

                    metrics = {
                        "overall_weighted_score": result["overall_weighted_score"],
                        "functional_score": result["code"]["functional"]["score"],
                        "assertions_score": result["code"]["assertions"]["score"],
                        "syntax_and_execution": result["code"]["syntax_and_execution"]["score"],
                        "quality": result["code"]["quality"]["score"],
                    }

                    # Log evaluation and artifacts in one run
                    log_evaluation_to_mlflow(
                        run_name=run_name,
                        results=evaluation_records,
                        metrics=metrics,
                        log_diffs=True
                    )

                    # Also log full evaluation JSON
                    mlflow.log_dict(result, artifact_file="validation_result.json")
                    mlflow.log_text(result['code']['semantic_diff_text'], artifact_file='semantic_diff.txt')
                # Log generated files for traceability
                mlflow.log_text(generated_script, artifact_file='generated_script.py')
                mlflow.log_text(generated_imports, artifact_file='generated_imports.py')
                mlflow.log_text(generated_config, artifact_file='generated_config.py')

                # Log basic parameters for traceability
                mlflow.log_params({
                    'language': record.inputs.get('language'),
                    'framework': record.inputs.get('testing_framework'),
                    'chat_id': record.inputs.get('chat_id'),
                    'status': 'success'
                })

                #mlflow.log_text(generated_script, artifact_file='generated_script.py')
                #mlflow.log_text(generated_imports, artifact_file='generated_imports.py')
                #mlflow.log_text(generated_config, artifact_file='generated_config.py')

                #mlflow.log_metric('overall_weighted_score', result['overall_weighted_score'])
                #mlflow.log_dict(result, artifact_file='validation_result.json')
                #mlflow.log_text(result['code']['semantic_diff_text'], artifact_file='semantic_diff.txt')

                #mlflow.log_param('status', 'success')
                print(f'Completed {run_name}')

            except Exception as e:
                mlflow.log_param('status', 'failed')
                try:
                    mlflow.log_text(e.response.text, artifact_file='api_error_response.txt')
                except Exception:
                    mlflow.log_text(str(e), artifact_file='api_error.txt')
                print(f'Error on {run_name}: {e}')

if __name__ == '__main__':
    run_generation()
