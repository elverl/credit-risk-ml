"""Register frozen Baseline-vs-RAG results in local MLflow or DagsHub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.tracking import (  # noqa: E402
    EXPERIMENT_NAME,
    RAG_EXPERIMENT_NAME,
    configure_tracking,
    rag_run_payloads,
)


RAG_DIR = ROOT / "artifacts" / "rag"
SUMMARY_PATH = RAG_DIR / "rag_comparison_summary_v1.json"
RUN_NAMES = {"baseline": "baseline_gpt_oss_20b", "rag": "rag_gpt_oss_20b"}
TRACKING_KEY = "rag_evalset_v1_final"
ARTIFACTS = {
    "baseline": [RAG_DIR / "baseline_responses_v1.jsonl"],
    "rag": [
        RAG_DIR / "rag_responses_v1.jsonl",
        RAG_DIR / "retrieval_eval.json",
        RAG_DIR / "rag_comparison_results_v1.jsonl",
        RAG_DIR / "rag_comparison_summary_v1.json",
    ],
}


def _benchmark_run_count(client: MlflowClient) -> int | None:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None
    return len(client.search_runs([experiment.experiment_id], run_view_type=ViewType.ALL))


def _existing_run(client: MlflowClient, experiment_id: str, role: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.tracking_key = '{TRACKING_KEY}' AND "
            f"tags.evaluation_variant = '{role}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def _artifact_files(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    files: set[str] = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            files.update(_artifact_files(client, run_id, item.path))
        else:
            files.add(item.path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "dagshub"), required=True)
    args = parser.parse_args()
    if not SUMMARY_PATH.exists():
        raise SystemExit("Frozen comparison summary not found.")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    payloads = rag_run_payloads(summary)
    settings, experiment_id = configure_tracking(
        ROOT / "artifacts" / "mlflow",
        mode=args.mode,
        experiment_name=RAG_EXPERIMENT_NAME,
    )
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    benchmark_before = _benchmark_run_count(client)
    run_manifest = {}
    for role in ("baseline", "rag"):
        existing = _existing_run(client, experiment_id, role)
        if existing is None:
            with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAMES[role]) as active:
                mlflow.set_tags({
                    "tracking_key": TRACKING_KEY,
                    "evaluation_variant": role,
                    "pipeline": "genai_rag_evaluation",
                })
                mlflow.log_params(payloads[role]["params"])
                mlflow.log_metrics(payloads[role]["metrics"])
                for artifact in ARTIFACTS[role]:
                    if not artifact.exists():
                        raise FileNotFoundError(artifact)
                    mlflow.log_artifact(str(artifact), artifact_path="evaluation")
                run_id = active.info.run_id
        else:
            run_id = existing.info.run_id

        # Recover a fully logged run left RUNNING if the client process was
        # interrupted after upload but before its final status update.
        current = client.get_run(run_id)
        if current.info.status != "FINISHED":
            client.set_terminated(run_id, status="FINISHED")

        run = client.get_run(run_id)
        for key, expected in payloads[role]["metrics"].items():
            actual = run.data.metrics.get(key)
            if actual is None or abs(actual - expected) > 1e-12:
                raise RuntimeError(f"Metric mismatch for {role}.{key}: {actual} != {expected}")
        artifact_files = _artifact_files(client, run_id)
        expected_files = {f"evaluation/{path.name}" for path in ARTIFACTS[role]}
        missing = expected_files - artifact_files
        if missing:
            raise RuntimeError(f"Missing artifacts in {role} run: {sorted(missing)}")
        run_manifest[role] = {
            "run_name": RUN_NAMES[role],
            "run_id": run_id,
            "status": run.info.status,
            "artifact_files": sorted(expected_files),
        }

    benchmark_after = _benchmark_run_count(client)
    if benchmark_before != benchmark_after:
        raise RuntimeError("Benchmark ML experiment run count changed")
    manifest = {
        "tracking_mode": settings.mode,
        "tracking_uri": settings.manifest_uri,
        "experiment_name": RAG_EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "tracking_key": TRACKING_KEY,
        "runs": run_manifest,
        "benchmark_run_count_before": benchmark_before,
        "benchmark_run_count_after": benchmark_after,
    }
    manifest_path = RAG_DIR / f"rag_mlflow_tracking_{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
