#!/usr/bin/env python3
import os, asyncio, json, tempfile, requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from validator import evaluate

# Disable OpenTelemetry
os.environ["LANGFUSE_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())



# =========================================================
#  Setup & ENV
# =========================================================
load_dotenv()
API_URL = os.getenv("SEMBIIQ_URL", "")
API_HEADERS = {
    "x-api-key": os.getenv("SEMBIIQ_API_KEY", ""),
    "x-brand": os.getenv("SEMBIIQ_BRAND", "XRAY"),
    "x-product": os.getenv("SEMBIIQ_PRODUCT", "XRAY"),
    "x-customer": os.getenv("SEMBIIQ_CUSTOMER", "XRAY"),
    "x-provider": os.getenv("SEMBIIQ_PROVIDER"),
    "x-llm-key": os.getenv("SEMBIIQ_LLM_KEY"),
    "x-model-id": os.getenv("SEMBIIQ_MODEL_ID"),
}

from langfuse import get_client
langfuse = get_client()

# =========================================================
#  API call
# =========================================================
def call_ai_api(record_inputs):
    payload = {
        "chat_id": record_inputs.get("chat_id"),
        "language": record_inputs.get("language"),
        "testing_framework": record_inputs.get("testing_framework"),
        "user_query": record_inputs.get("user_query"),
        "context": record_inputs.get("context", {}),
        "current_code": record_inputs.get("current_code", ""),
        "current_imports": record_inputs.get("current_imports", ""),
        "current_config_file": record_inputs.get("current_config_file", ""),
        "test_case": record_inputs.get("test_case", {}),
        "thinking_budget": record_inputs.get("thinking_budget", 0),
    }

    headers = {k: v for k, v in API_HEADERS.items() if v}
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()

# =========================================================
#  Evaluator logic
# =========================================================
def validator_evaluator(item, output):
    record_inputs = item.input
    expectations = item.expected_output or {}

    generated_script = output.get("code", "")
    generated_imports = output.get("imports", "")
    generated_config = output.get("config_file", "")
    expected_script = expectations.get("code", "")

    # temp files
    gen_file = Path(tempfile.mktemp(suffix=".py")); gen_file.write_text(generated_script or "")
    gen_imports = Path(tempfile.mktemp(suffix=".py")); gen_imports.write_text(generated_imports or "")
    gen_config = Path(tempfile.mktemp(suffix=".xml")); gen_config.write_text(generated_config or "")
    exp_file = Path(tempfile.mktemp(suffix=".py")); exp_file.write_text(expected_script or "")

    try:
        result = evaluate(record_inputs, gen_file, gen_imports, gen_config, baseline_code=exp_file)
        overall = result.get("overall_weighted_score", 0.0)
        sub_scores = {
            "syntax_exec": result["code"]["syntax_and_execution"]["score"],
            "functional": result["code"]["functional"]["score"],
            "assertions": result["code"]["assertions"]["score"],
            "quality": result["code"]["quality"]["score"],
            "imports": result["imports"]["score"],
            "config": result["config"]["score"],
        }
        semantic_text = result["code"]["semantic_diff_text"]
        return {"score": overall, "sub_scores": sub_scores,
                "semantic_text": semantic_text, "raw": result}
    except Exception as e:
        return {"score": 0.0, "error": str(e), "sub_scores": {}, "semantic_text": ""}

# =========================================================
#  Task wrapper (used by run_experiment)
# =========================================================
def generation_task(item):
    record_inputs = item.input
    api_response = call_ai_api(record_inputs)
    evaluation = validator_evaluator(item, api_response)

    # combine metrics + semantic diff metadata
    overall = evaluation["score"]
    scores = [{"name": "overall_weighted_score", "value": overall,
               "metadata": {"semantic_diff": evaluation["semantic_text"]}}]
    for name, val in evaluation["sub_scores"].items():
        scores.append({
            "name": name,
            "value": val,
            "metadata": {"semantic_diff": evaluation["semantic_text"]}
        })

    return {
        "output": api_response,
        "score": overall,
        "scores": scores,
        "metadata": {
            "semantic_diff": evaluation["semantic_text"],
            "validator_result": evaluation["raw"],
        },
    }

# =========================================================
#  Runner
# =========================================================
def run_generation(dataset_name="ai_script_test_suite", experiment_name=None):
    with open("dataset_info.json") as f:
        info = json.load(f)
    dataset_name = info.get("dataset_name", dataset_name)

    if not experiment_name:
        experiment_name = f"{dataset_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"🚀 Starting Langfuse experiment: {experiment_name}")
    dataset = langfuse.get_dataset(name=dataset_name)

    experiment_result = langfuse.run_experiment(
        name=experiment_name,
        description="Validator evaluation + semantic diff metadata",
        data=dataset.items,
        task=generation_task,
    )

    print(f"🎯 Experiment complete: {experiment_name}")
    print("Langfuse Experiment URL:", getattr(experiment_result, "dataset_run_url", "Check dashboard"))
    print("Description:", getattr(experiment_result, "description", {}))

# =========================================================
#  Main
# =========================================================
if __name__ == "__main__":
    run_generation()



    
