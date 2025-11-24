import mlflow
import pandas as pd
import difflib
from pathlib import Path
from mlflow.models import EvaluationResult
from mlflow.entities import SpanType


# =========================================================
# Helper: HTML diff generator
# =========================================================
def generate_html_diff(expected: str, generated: str, section: str, iteration=None):
    """
    Generate an HTML diff comparing expected and generated text.
    """
    html_diff = difflib.HtmlDiff().make_file(
        expected.splitlines(),
        generated.splitlines(),
        fromdesc=f"Expected {section}",
        todesc=f"Generated {section}",
        context=True,
        numlines=4
    )
    subdir = f"iteration_{iteration}" if iteration else "all_iterations"
    diff_path = Path(f"mlflow_artifacts/{subdir}/{section}_diff.html")
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(html_diff, encoding="utf-8")
    return diff_path


# =========================================================
# Main logging entrypoint
# =========================================================
def log_evaluation_to_mlflow(
    run_name: str,
    results: list,
    metrics: dict = None,
    log_diffs: bool = True,
    iteration: int = None,
    add_trace: bool = True
):
    """
    Logs expected vs generated comparisons as both:
      - MLflow run artifacts (HTML diffs, CSVs)
      - Trace span attributes (visible for assessments)
    """

    # --- Build comparison table ---
    df = pd.DataFrame(results)
    if "semantic_score" not in df.columns:
        df["semantic_score"] = None
    df["match"] = df["expected"] == df["generated"]

    # --- Create evaluation result ---
    evaluation_result = EvaluationResult(
        metrics=metrics or {},
        artifacts={"comparison_table": df}
    )

    # --- Determine iteration label ---
    iteration_label = f"iteration_{iteration}" if iteration else "all_iterations"

    # --- Begin trace-aware logging ---
    if add_trace:
        span_name = f"Evaluation_{iteration_label}"
        with mlflow.start_span(span_name, span_type=SpanType.EVALUATOR) as span:
            _log_eval_core(run_name, df, metrics, log_diffs, iteration, evaluation_result, span)
    else:
        _log_eval_core(run_name, df, metrics, log_diffs, iteration, evaluation_result)


# =========================================================
# Core: Shared logic for trace + artifacts
# =========================================================
def _log_eval_core(run_name, df, metrics, log_diffs, iteration, evaluation_result, span=None):
    iteration_label = f"iteration_{iteration}" if iteration else "all_iterations"

    # --- Log metrics ---
    if metrics:
        mlflow.log_metrics(metrics)
        if span:
            for k, v in metrics.items():
                span.set_attribute(f"metric.{k}", v)

    # --- Basic params ---
    sample = df.iloc[0].to_dict() if not df.empty else {}
    for key in ["language", "framework"]:
        val = sample.get(key)
        if val:
            mlflow.log_param(key, val)
            if span:
                span.set_attribute(key, val)

    # --- Save CSV table as artifact ---
    out_dir = Path("mlflow_artifacts")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"comparison_table_{iteration_label}.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path), artifact_path="tables")
    if span:
        span.set_attribute("artifact.comparison_table", str(csv_path))

    # --- Log HTML diffs ---
    if log_diffs:
        diff_summaries = []
        for _, row in df.iterrows():
            section = row.get("section")
            expected = row.get("expected", "")
            generated = row.get("generated", "")
            if not expected or not generated:
                continue

            html_path = generate_html_diff(expected, generated, section, iteration)
            mlflow.log_artifact(str(html_path), artifact_path=f"diffs/{section}/{iteration_label}")

            summary = f"DIFF ({section}): expected len={len(expected)}, generated len={len(generated)}"
            diff_summaries.append(summary)

            # ✅ Add event for trace (version-safe)
            if span:
                event_name = f"diff_{section}"
                event_attrs = {
                    "expected_snippet": expected[:300],
                    "generated_snippet": generated[:300],
                    "section": section,
                    "iteration": iteration or "all"
                }

                try:
                    # Preferred way: MLflow >= 2.15
                    from mlflow.entities import Event
                    event = Event(name=event_name, attributes=event_attrs)
                    span.add_event(event)
                except Exception:
                    try:
                        # Fallback for older versions: use underlying OTEL span
                        if hasattr(span, "_span") and hasattr(span._span, "add_event"):
                            span._span.add_event(event_name, event_attrs)
                        else:
                            # Ultimate fallback: store as attributes
                            for k, v in event_attrs.items():
                                span.set_attribute(f"event.{event_name}.{k}", v)
                    except Exception as e:
                        # Safe fallback — no crash
                        span.set_attribute(f"diff_event_error.{event_name}", str(e))

                    if span:
                        span.set_attribute("diff_summary", "\n".join(diff_summaries))

    # --- Log evaluation metrics JSON ---
    metrics_json_path = Path(f"mlflow_artifacts/evaluation_metrics_{iteration_label}.json")
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(str(evaluation_result.metrics), encoding="utf-8")
    mlflow.log_artifact(str(metrics_json_path), artifact_path="metrics")
    if span:
        span.set_attribute("artifact.metrics_json", str(metrics_json_path))

    # --- Semantic score in trace ---
    if "semantic_score" in df.columns and span:
        avg_sem = df["semantic_score"].dropna().mean() if df["semantic_score"].notnull().any() else None
        if avg_sem is not None:
            span.set_attribute("semantic_score_mean", round(avg_sem, 3))

    print(f"✅ Logged evaluation (trace + artifacts) for {run_name} [{iteration_label}]")
