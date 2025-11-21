import mlflow
import pandas as pd
import difflib
from pathlib import Path
from mlflow.models import EvaluationResult


# =========================================================
# Diff helpers
# =========================================================
def generate_html_diff(expected: str, generated: str, label: str):
    """
    Generates an HTML diff file comparing expected vs generated text.
    """
    html_diff = difflib.HtmlDiff().make_file(
        expected.splitlines(),
        generated.splitlines(),
        fromdesc=f"Expected {label}",
        todesc=f"Generated {label}",
        context=True,
        numlines=4
    )
    diff_path = Path(f"mlflow_artifacts/{label}_diff.html")
    diff_path.parent.mkdir(exist_ok=True)
    diff_path.write_text(html_diff, encoding="utf-8")
    return diff_path


# =========================================================
# Evaluation table + artifact logger (with semantic similarity)
# =========================================================
def log_evaluation_to_mlflow(
    run_name: str,
    results: list,
    metrics: dict = None,
    log_diffs: bool = True
):
    """
    Logs expected vs generated comparisons as an MLflow evaluation table.

    Parameters
    ----------
    run_name : str
        The name of the MLflow run (e.g. dataset name or experiment ID)
    results : list of dict
        Each dict should contain at least:
            {
                "chat_id": str,
                "section": "code" | "imports" | "config",
                "expected": str,
                "generated": str,
                "score": float,
                "semantic_score": float,  # optional
                "language": str,
                "framework": str
            }
    metrics : dict
        Optional global metrics (aggregated scores, averages, etc.)
    log_diffs : bool
        If True, logs per-section HTML diffs as artifacts
    """

    # Build evaluation DataFrame
    df = pd.DataFrame(results)
    if "semantic_score" not in df.columns:
        df["semantic_score"] = None

    # Compute match boolean
    df["match"] = df["expected"] == df["generated"]

    # Create evaluation result
    evaluation_result = EvaluationResult(
        metrics=metrics or {},
        artifacts={"comparison_table": df}#,
        #tables={"comparison": df},
    )

    run_id = mlflow.active_run().info.run_id

    # Log global metrics
    if metrics:
        mlflow.log_metrics(metrics)

    # Log basic metadata (language, framework)
    sample = results[0] if results else {}
    for key in ["language", "framework"]:
        if key in sample:
            mlflow.log_param(key, sample[key])

    # Compute and log average semantic score if available
    if "semantic_score" in df.columns and df["semantic_score"].notnull().any():
        avg_semantic = float(df["semantic_score"].dropna().mean())
        mlflow.log_metric("avg_semantic_score", round(avg_semantic, 3))

    # Save and log the comparison table
    out_dir = Path("mlflow_artifacts")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path), artifact_path="tables")

    # Optionally log HTML diffs
    if log_diffs:
        for _, row in df.iterrows():
            section = row.get("section")
            expected = row.get("expected", "")
            generated = row.get("generated", "")
            if expected or generated:
                html_path = generate_html_diff(expected, generated, section)
                mlflow.log_artifact(str(html_path), artifact_path=f"diffs/{section}")

    # Save and log evaluation table as CSV artifact
    out_dir = Path("mlflow_artifacts")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "evaluation_table.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path), artifact_path="tables")

    # Log overall metrics
    if metrics:
        mlflow.log_metrics(metrics)

    # ✅ Optionally log the evaluation result object as a JSON artifact instead
    mlflow.log_dict(
        evaluation_result.metrics,
        artifact_file="evaluation_metrics.json"
    )

    print(f"✅ Logged evaluation for run '{run_name}' ({run_id})")

    return evaluation_result
