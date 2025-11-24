import mlflow
import json
import requests
from pathlib import Path
import tempfile
import os
from validator import evaluate
from dotenv import load_dotenv
from mlflow_utils import log_evaluation_to_mlflow
from mlflow.entities import SpanType

# =========================================================
# ENVIRONMENT SETUP
# =========================================================
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

# =========================================================
# HELPER: CALL AI API
# =========================================================
def call_ai_api(payload, iteration_idx=None):
    headers = {k: v for k, v in API_HEADERS.items() if v}
    with mlflow.start_span(f"AI_API_Call_Iteration_{iteration_idx or 1}"):
        mlflow.log_text(json.dumps(payload, indent=2),
                        artifact_file=f"api_payload_iteration_{iteration_idx or 1}.json")
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        mlflow.log_text(f"HTTP status: {resp.status_code}",
                        artifact_file=f"api_status_iteration_{iteration_idx or 1}.txt")
        mlflow.log_text(resp.text,
                        artifact_file=f"api_response_iteration_{iteration_idx or 1}.json")
        resp.raise_for_status()
        return resp.json()

# =========================================================
# MAIN EXECUTION LOOP
# =========================================================
def run_generation(dataset_name='ai_script_test_suite'):
    with open("dataset_info.json") as f:
        info = json.load(f)

    dataset_id = info["dataset_id"]
    ds = mlflow.genai.datasets.get_dataset(dataset_id=dataset_id)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for record in ds.records:
        # ✅ Use attribute-style access (DatasetRecord object)
        inputs = record.inputs
        expectations = record.expectations or {}

        run_name = (
            inputs.get("test_case", {}).get("title")
            or inputs.get("chat_id")
            or "unnamed"
        )

        # Initialize iteration state
        current_code = inputs.get("current_code", "")
        current_imports = inputs.get("current_imports", "")
        current_config = inputs.get("current_config_file", "")

        iterations = inputs.get("iterations", [])
        if not iterations:
            iterations = [{"user_query": inputs.get("user_query", "")}]

        print(f"\n🚀 Running test '{run_name}' with {len(iterations)} iteration(s)")

        with mlflow.start_run(run_name=run_name):
            all_iteration_records = []

            for i, iteration in enumerate(iterations):
                try:
                    print(f"🔁 Iteration {i + 1}: {iteration.get('user_query')}")

                    payload = {
                        "chat_id": inputs.get("chat_id"),
                        "language": inputs.get("language"),
                        "testing_framework": inputs.get("testing_framework"),
                        "user_query": iteration.get("user_query"),
                        "context": inputs.get("context", {}),
                        "current_code": current_code,
                        "current_imports": current_imports,
                        "current_config_file": current_config,
                        "test_case": inputs.get("test_case", {}),
                        "thinking_budget": inputs.get("thinking_budget", 0),
                    }

                    api_response = call_ai_api(payload, i + 1)
                    generated_code = api_response.get("code", "")
                    generated_imports = api_response.get("imports", "")
                    generated_config = api_response.get("config_file", "")

                    # Write temp files for evaluation
                    gen_code_file = Path(tempfile.mktemp(suffix=".py"))
                    gen_code_file.write_text(generated_code)
                    gen_imports_file = Path(tempfile.mktemp(suffix=".py"))
                    gen_imports_file.write_text(generated_imports)
                    gen_config_file = Path(tempfile.mktemp(suffix=".py"))
                    gen_config_file.write_text(generated_config)

                    expected_code = expectations.get("code", "")
                    expected_imports = expectations.get("imports", "")
                    expected_config = expectations.get("config", "")

                    exp_code_file = Path(tempfile.mktemp(suffix=".py"))
                    exp_code_file.write_text(expected_code)
                    exp_imports_file = Path(tempfile.mktemp(suffix=".py"))
                    exp_imports_file.write_text(expected_imports)
                    exp_config_file = Path(tempfile.mktemp(suffix=".py"))
                    exp_config_file.write_text(expected_config)

                    if expected_code.strip():
                        result = evaluate(
                            inputs,
                            gen_code_file,
                            gen_imports_file,
                            gen_config_file,
                            baseline_code=exp_code_file,
                            baseline_imports=exp_imports_file,
                            baseline_config=exp_config_file,
                        )

                        evaluation_records = [
                            {
                                "chat_id": payload["chat_id"],
                                "iteration": i + 1,
                                "section": "code",
                                "expected": expected_code,
                                "generated": generated_code,
                                "score": result["code"]["functional"]["score"],
                                "semantic_score": result["code"]["functional"]["score"],
                                "language": payload["language"],
                                "framework": payload["testing_framework"],
                            },
                            {
                                "chat_id": payload["chat_id"],
                                "iteration": i + 1,
                                "section": "imports",
                                "expected": expected_imports,
                                "generated": generated_imports,
                                "score": result["imports"]["score"],
                                "semantic_score": result["imports"]["score"],
                                "language": payload["language"],
                                "framework": payload["testing_framework"],
                            },
                            {
                                "chat_id": payload["chat_id"],
                                "iteration": i + 1,
                                "section": "config",
                                "expected": expected_config,
                                "generated": generated_config,
                                "score": result["config"]["score"],
                                "semantic_score": result["config"]["score"],
                                "language": payload["language"],
                                "framework": payload["testing_framework"],
                            },
                        ]

                        all_iteration_records.extend(evaluation_records)

                        metrics = {
                            "overall_weighted_score": result["overall_weighted_score"],
                            "functional_score": result["code"]["functional"]["score"],
                            "assertions_score": result["code"]["assertions"]["score"],
                            "syntax_and_execution": result["code"]["syntax_and_execution"]["score"],
                            "quality": result["code"]["quality"]["score"],
                        }

                        # ✅ Log iteration-specific diffs immediately
                        log_evaluation_to_mlflow(
                            run_name=run_name,
                            results=evaluation_records,
                            metrics=metrics,
                            log_diffs=True,
                            iteration=i + 1,
                            add_trace=True,  # ✅ now logged in trace
                        )

                        mlflow.log_metrics(
                            {f"{k}_iter_{i+1}": v for k, v in metrics.items()}
                        )
                        mlflow.log_dict(
                            result, artifact_file=f"validation_result_iteration_{i+1}.json"
                        )
                        mlflow.log_text(
                            result["code"]["semantic_diff_text"],
                            artifact_file=f"semantic_diff_iteration_{i+1}.txt",
                        )

                    # Log generated outputs
                    mlflow.log_text(
                        generated_code, artifact_file=f"generated_code_iteration_{i+1}.py"
                    )
                    mlflow.log_text(
                        generated_imports,
                        artifact_file=f"generated_imports_iteration_{i+1}.py",
                    )
                    mlflow.log_text(
                        generated_config,
                        artifact_file=f"generated_config_iteration_{i+1}.py",
                    )

                    # Update context for next iteration
                    current_code = generated_code
                    current_imports = generated_imports
                    current_config = generated_config

                    print(f"✅ Completed iteration {i+1} for {run_name}")

                except Exception as e:
                    mlflow.log_param("status", "failed")
                    try:
                        mlflow.log_text(
                            e.response.text,
                            artifact_file=f"api_error_response_iter_{i+1}.txt",
                        )
                    except Exception:
                        mlflow.log_text(str(e), artifact_file=f"api_error_iter_{i+1}.txt")
                    print(f"❌ Error on iteration {i+1} for {run_name}: {e}")

            # Log combined evaluation table once (after all iterations)
            if all_iteration_records:
                log_evaluation_to_mlflow(
                    run_name=run_name,
                    results=all_iteration_records,
                    metrics={},
                    log_diffs=True,
                    iteration=None,  # this creates 'all_iterations' summary
                    add_trace=True
                )

            mlflow.log_params(
                {
                    "language": inputs.get("language"),
                    "framework": inputs.get("testing_framework"),
                    "chat_id": inputs.get("chat_id"),
                    "total_iterations": len(iterations),
                    "status": "success",
                }
            )

            print(f"🏁 Completed all iterations for {run_name}")


if __name__ == "__main__":
    run_generation()
