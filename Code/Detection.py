

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm


DEFAULT_ENCODINGS = ("utf-8", "utf-8-sig", "latin1", "cp1252")
THREAD_LOCAL = threading.local()

# Define prompt/schema in-code here.
INLINE_PROMPT_VARIATIONS: list[dict[str, Any]] = [
    {
        "name": "default_inline_prompt",
        "system_prompt": (
            "INPUT PROMPT SPECIFIED IN THE PAPER."
        ),
        "output_schema": {
            "name": "ile_binary_annotation",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "analyzed": {"type": "string", "enum": ["yes", "no"]},
                    "encoding_detected": {"type": "string", "enum": ["yes", "no"]},
                    "encoding_evidence": {"type": "array", "items": {"type": "string"}},
                    "decoded_meaning": {"type": "array", "items": {"type": "string"}},
                    "mechanism": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["encoding_detected", "encoding_evidence", "decoded_meaning", "mechanism"],
            },
        },
    }
]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read CSV with common encoding fallbacks."""
    last_exc: Exception | None = None
    for enc in DEFAULT_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_exc


def normalize_schema_bundle(schema_bundle: dict[str, Any]) -> dict[str, Any]:
    """Ensure schema bundle contains {'name','schema'} shape."""
    if "name" in schema_bundle and "schema" in schema_bundle:
        return schema_bundle

    looks_like_raw_schema = isinstance(schema_bundle.get("type"), str) and isinstance(schema_bundle.get("properties"), dict)
    if looks_like_raw_schema:
        return {"name": "ile_binary_annotation", "schema": schema_bundle}

    raise ValueError("output_schema must contain 'name' and 'schema' (or be a raw JSON schema dict).")


def infer_id_col(columns: list[str]) -> str:
    for col in columns:
        if col.strip().lower() == "id":
            return col
    raise ValueError("Could not find ID column.")


def infer_text_col(columns: list[str]) -> str:
    for col in columns:
        if "content" in col.lower():
            return col
    raise ValueError("Could not infer text/content column from dataset.")


def infer_label_col(columns: list[str]) -> str:
    for col in columns:
        if "ile found" in col.lower():
            return col
    raise ValueError("Could not infer label column containing 'ILE found'.")


def detect_root_dir(script_file: Path) -> Path:
    script_dir = script_file.resolve().parent
    candidates = [script_dir] + list(script_dir.parents)
    for cand in candidates:
        has_dataset = (cand / "Dataset").is_dir()
        has_outputs = (cand / "Model Outputs").is_dir() or (cand / "Model Outputs Final" / "Model Outputs").is_dir()
        has_code = (cand / "Code").is_dir() or (cand / "Codes").is_dir()
        if has_dataset and has_outputs and has_code:
            return cand
    for cand in candidates:
        has_dataset = (cand / "Dataset").is_dir()
        has_outputs = (cand / "Model Outputs").is_dir() or (cand / "Model Outputs Final" / "Model Outputs").is_dir()
        if has_dataset and has_outputs:
            return cand
    if script_dir.name.lower() in {"code", "codes"}:
        return script_dir.parent
    return script_dir


def to_binary_label(value: Any) -> int:
    v = str(value).strip().lower()
    if v == "yes":
        return 1
    if v == "no":
        return 0
    raise ValueError(f"Unexpected label value: {value!r}")


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return "null"


def extract_prediction_fields(parsed: dict[str, Any]) -> tuple[int, str, list[Any], list[Any], list[Any]]:
    detected_raw = str(parsed.get("encoding_detected", "")).strip().lower()
    if detected_raw == "yes":
        y_pred = 1
        label = "yes"
    elif detected_raw == "no":
        y_pred = 0
        label = "no"
    else:
        # Conservative fallback to negative for failures.
        y_pred = 0
        label = "no"

    evidence = parsed.get("encoding_evidence", [])
    decoded = parsed.get("decoded_meaning", [])
    mechanism = parsed.get("mechanism", parsed.get("mechanisms", []))

    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    if not isinstance(decoded, list):
        decoded = [str(decoded)]
    if not isinstance(mechanism, list):
        mechanism = [mechanism]

    return y_pred, label, evidence, decoded, mechanism


def call_model_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    schema_bundle: dict[str, Any],
    text_input: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=f"Input:\n{text_input}",
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_bundle["name"],
                "schema": schema_bundle["schema"],
                "strict": True,
            }
        },
    )

    raw_text = response.output_text or ""
    if not raw_text.strip():
        raise ValueError("Model returned empty output_text.")

    parsed = json.loads(raw_text)
    usage_obj = getattr(response, "usage", None)
    usage = {}
    if usage_obj is not None:
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }

    return parsed, raw_text, usage


def call_model_with_retries(
    client: OpenAI,
    model: str,
    system_prompt: str,
    schema_bundle: dict[str, Any],
    text_input: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_retries: int,
    initial_retry_delay: float,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            parsed, raw_text, usage = call_model_once(
                client=client,
                model=model,
                system_prompt=system_prompt,
                schema_bundle=schema_bundle,
                text_input=text_input,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            return parsed, raw_text, usage, ""
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                sleep_s = initial_retry_delay * (2**attempt)
                time.sleep(sleep_s)
            else:
                break

    assert last_err is not None
    raise last_err


def get_thread_client(api_key: str) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key)
        THREAD_LOCAL.client = client
    return client


def predict_one_row(
    *,
    api_key: str,
    model: str,
    prompt_file_name: str,
    system_prompt: str,
    schema_bundle: dict[str, Any],
    row_id: Any,
    text_input: str,
    y_true: int,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_retries: int,
    initial_retry_delay: float,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    raw_text = ""
    usage: dict[str, Any] = {}
    error_msg = ""
    parse_success = False

    try:
        client = get_thread_client(api_key)
        parsed, raw_text, usage, _ = call_model_with_retries(
            client=client,
            model=model,
            system_prompt=system_prompt,
            schema_bundle=schema_bundle,
            text_input=text_input,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
        )
        parse_success = True
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        # Conservative fallback on failures.
        parsed = {
            "analyzed": "yes",
            "encoding_detected": "no",
            "encoding_evidence": [],
            "decoded_meaning": [],
            "mechanism": [],
        }

    y_pred, pred_label, evidence, decoded, mechanism = extract_prediction_fields(parsed)
    y_score = float(y_pred)

    return {
        "ID": row_id,
        "text": text_input,
        "gold_label": "yes" if y_true == 1 else "no",
        "pred_label": pred_label,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "parse_success": parse_success,
        "error": error_msg,
        "encoding_evidence": safe_json_dumps(evidence),
        "decoded_meaning": safe_json_dumps(decoded),
        "mechanism": safe_json_dumps(mechanism),
        "raw_json": raw_text,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_file": prompt_file_name,
        "model": model,
    }


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, float]:
    """Compute binary-detection metrics and extended summary metrics."""
    accuracy = float(accuracy_score(y_true, y_pred))

    # Primary requested metrics: macro-averaged precision/recall/f1.
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # Also keep binary (positive-class) and weighted averages in summary.
    p_binary, r_binary, f1_binary, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # Per-class metrics for full transparency in summary outputs.
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy),
        # Backward-compatible aliases now mapped to macro metrics.
        "precision": float(p_macro),
        "recall": float(r_macro),
        "f1": float(f1_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_binary": float(p_binary),
        "recall_binary": float(r_binary),
        "f1_binary": float(f1_binary),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_class0": float(p_cls[0]),
        "recall_class0": float(r_cls[0]),
        "f1_class0": float(f1_cls[0]),
        "support_class0": float(sup_cls[0]),
        "precision_class1": float(p_cls[1]),
        "recall_class1": float(r_cls[1]),
        "f1_class1": float(f1_cls[1]),
        "support_class1": float(sup_cls[1]),
    }

    if len(set(y_true)) < 2:
        metrics["auc"] = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, y_score)
            metrics["auc"] = float(auc)
        except Exception:  # noqa: BLE001
            metrics["auc"] = float("nan")

    return metrics


def main() -> None:
    codes_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Benchmark prompt variations with an OpenAI model.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4"), help="OpenAI model name.")
    parser.add_argument(
        "--root-dir",
        default="",
        help="Project root containing `Dataset`, `Model Outputs`, and `Code`/`Codes` folders. "
        "Default: auto-detected from script location.",
    )
    parser.add_argument(
        "--evaluation-dir",
        default="",
        help="Backward-compatible alias for --root-dir.",
    )
    parser.add_argument(
        "--dataset-file",
        default="Dataset/Dataset.csv",
        help="Dataset CSV path (relative to project root).",
    )
    parser.add_argument(
        "--ground-truth-file",
        default="Dataset/GroundTruth.csv",
        help="Ground-truth CSV path (relative to project root).",
    )
    parser.add_argument(
        "--output-root",
        default="Model Outputs",
        help="Root output directory for run artifacts (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--only-prompt",
        default=None,
        help="Run only one inline variation by name (from INLINE_PROMPT_VARIATIONS).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick test runs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads. Set 1 for sequential mode.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per row on API/parse failure.")
    parser.add_argument("--initial-retry-delay", type=float, default=1.5, help="Initial retry delay in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter.")
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="Max output tokens per row.")
    args = parser.parse_args()

    if args.root_dir:
        root_dir = Path(args.root_dir).resolve()
    elif args.evaluation_dir:
        root_dir = Path(args.evaluation_dir).resolve()
    else:
        root_dir = detect_root_dir(Path(__file__))

    dataset_path = Path(args.dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = (root_dir / args.dataset_file).resolve()
    gt_path = Path(args.ground_truth_file)
    if not gt_path.is_absolute():
        gt_path = (root_dir / args.ground_truth_file).resolve()

    if not dataset_path.exists():
        dataset_candidates = [
            root_dir / "Dataset" / "Dataset.csv",
            root_dir / "Dataset" / "data-2065.csv",
            root_dir.parent / "Dataset" / "Dataset.csv",
            root_dir.parent / "Dataset" / "data-2065.csv",
        ]
        for candidate in dataset_candidates:
            if candidate.exists():
                dataset_path = candidate.resolve()
                break

    if not gt_path.exists():
        gt_candidates = [
            root_dir / "Dataset" / "GroundTruth.csv",
            root_dir / "Dataset" / "GroundTruth-Final.csv",
            root_dir / "Dataset" / "GT-2065.csv",
            root_dir.parent / "Dataset" / "GroundTruth.csv",
            root_dir.parent / "Dataset" / "GroundTruth-Final.csv",
            root_dir.parent / "Dataset" / "GT-2065.csv",
        ]
        for candidate in gt_candidates:
            if candidate.exists():
                gt_path = candidate.resolve()
                break

    output_root = Path(args.output_root)
    output_root_arg_is_abs = output_root.is_absolute()
    if not output_root.is_absolute():
        output_root = (root_dir / args.output_root).resolve()
    if not output_root.exists() and not output_root_arg_is_abs:
        output_candidates = [
            (root_dir / "Model Outputs Final" / args.output_root).resolve(),
            (root_dir.parent / args.output_root).resolve(),
            (root_dir.parent / "Model Outputs Final" / args.output_root).resolve(),
        ]
        for candidate in output_candidates:
            if candidate.exists():
                output_root = candidate
                break

    if not dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {dataset_path}")
    if not gt_path.exists():
        raise RuntimeError(f"Ground-truth file not found: {gt_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"benchmark_{run_stamp}_{args.model.replace('/', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Primary expectation: .env is in Codes folder.
    env_candidates = [
        codes_dir / ".env",
        root_dir / "Code" / ".env",
        root_dir / "Codes" / ".env",
        root_dir / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path)
            break
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found. Checked environment and .env candidates: "
            f"{codes_dir / '.env'}, {root_dir / 'Code' / '.env'}, "
            f"{root_dir / 'Codes' / '.env'}, {root_dir / '.env'}"
        )

    dataset_df = read_csv_with_fallback(dataset_path)
    gt_df = read_csv_with_fallback(gt_path)

    dataset_id_col = infer_id_col(dataset_df.columns.tolist())
    dataset_text_col = infer_text_col(dataset_df.columns.tolist())
    gt_id_col = infer_id_col(gt_df.columns.tolist())
    gt_label_col = infer_label_col(gt_df.columns.tolist())
    # Match ONLY by ID.
    dataset_ids = dataset_df[dataset_id_col].fillna("").astype(str).str.strip()
    gt_ids = gt_df[gt_id_col].fillna("").astype(str).str.strip()

    if (dataset_ids == "").any():
        n_empty = int((dataset_ids == "").sum())
        raise RuntimeError(f"Dataset has empty IDs ({n_empty}) in column: {dataset_id_col}")
    if (gt_ids == "").any():
        n_empty = int((gt_ids == "").sum())
        raise RuntimeError(f"Ground truth has empty IDs ({n_empty}) in column: {gt_id_col}")

    if dataset_ids.duplicated().any():
        n_dup = int(dataset_ids.duplicated().sum())
        raise RuntimeError(f"Dataset contains duplicate IDs ({n_dup}) in column: {dataset_id_col}")
    if gt_ids.duplicated().any():
        n_dup = int(gt_ids.duplicated().sum())
        raise RuntimeError(f"Ground truth contains duplicate IDs ({n_dup}) in column: {gt_id_col}")

    dataset_id_set = set(dataset_ids.tolist())
    gt_id_set = set(gt_ids.tolist())
    if dataset_id_set != gt_id_set:
        missing_in_gt = sorted(dataset_id_set - gt_id_set)[:5]
        missing_in_dataset = sorted(gt_id_set - dataset_id_set)[:5]
        raise RuntimeError(
            "ID set mismatch between dataset and ground truth. "
            f"Missing in GT: {len(dataset_id_set - gt_id_set)} {missing_in_gt}; "
            f"Missing in dataset: {len(gt_id_set - dataset_id_set)} {missing_in_dataset}"
        )

    dataset_df = dataset_df.copy()
    gt_df = gt_df.copy()
    dataset_df[dataset_id_col] = dataset_ids
    gt_df[gt_id_col] = gt_ids

    merged = dataset_df[[dataset_id_col, dataset_text_col]].merge(
        gt_df[[gt_id_col, gt_label_col]],
        left_on=dataset_id_col,
        right_on=gt_id_col,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(dataset_df):
        raise RuntimeError(
            f"Unexpected join size mismatch. Expected {len(dataset_df)} rows, got {len(merged)} rows."
        )
    if merged[gt_label_col].isna().any():
        n_missing = int(merged[gt_label_col].isna().sum())
        raise RuntimeError(f"Ground-truth labels missing after ID join for {n_missing} rows.")

    id_merge_ratio = 1.0
    join_mode_used = "id_only"
    join_reason = "strict_exact_id_set"

    if gt_id_col in merged.columns and gt_id_col != dataset_id_col:
        merged = merged.drop(columns=[gt_id_col])

    merged = merged.rename(
        columns={
            dataset_id_col: "ID",
            dataset_text_col: "text",
            gt_label_col: "gold_label",
        }
    )

    if args.limit is not None:
        merged = merged.head(args.limit).copy()

    merged["gold_label"] = merged["gold_label"].astype(str).str.strip().str.lower()
    merged["y_true"] = merged["gold_label"].apply(to_binary_label)
    print(
        "[join] "
        f"used={join_mode_used}, merged_rows={len(merged)}, id_merge_ratio={id_merge_ratio:.4f}"
    )

    prompt_variations = INLINE_PROMPT_VARIATIONS.copy()
    if args.only_prompt:
        target = args.only_prompt.strip().lower()
        prompt_variations = [v for v in prompt_variations if str(v.get("name", "")).strip().lower() == target]
    if not prompt_variations:
        if args.only_prompt:
            raise RuntimeError(f"No inline variation matched --only-prompt={args.only_prompt!r}")
        raise RuntimeError("No inline prompt variations configured. Edit INLINE_PROMPT_VARIATIONS at top of file.")

    metrics_rows: list[dict[str, Any]] = []

    run_meta = {
        "model": args.model,
        "root_dir": str(root_dir),
        "evaluation_dir_arg": str(args.evaluation_dir),
        "dataset_file": str(dataset_path),
        "ground_truth_file": str(gt_path),
        "output_root": str(output_root),
        "inline_prompt_variations": [str(v.get("name", "")) for v in prompt_variations],
        "join_mode_used": join_mode_used,
        "join_reason": join_reason,
        "id_merge_ratio": id_merge_ratio,
        "dataset_rows": int(len(dataset_df)),
        "ground_truth_rows": int(len(gt_df)),
        "row_count": int(len(merged)),
        "prompt_count": int(len(prompt_variations)),
        "workers": int(max(1, args.workers)),
        "top_p": float(args.top_p),
        "created_at_local": datetime.now().isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8-sig")

    for prompt_cfg in prompt_variations:
        variation = str(prompt_cfg["name"])
        print(f"\n=== Running variation: {variation} ===")
        system_prompt = str(prompt_cfg["system_prompt"])
        schema_bundle = normalize_schema_bundle(dict(prompt_cfg["output_schema"]))

        variation_dir = run_dir / variation
        variation_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = variation_dir / "predictions.csv"

        rows = list(merged.itertuples(index=False))
        records: list[dict[str, Any] | None] = [None] * len(rows)
        workers = max(1, int(args.workers))

        if workers == 1:
            for idx, row in enumerate(tqdm(rows, total=len(rows), desc=variation)):
                records[idx] = predict_one_row(
                    api_key=api_key,
                    model=args.model,
                    prompt_file_name=f"INLINE::{variation}",
                    system_prompt=system_prompt,
                    schema_bundle=schema_bundle,
                    row_id=row.ID,
                    text_input=str(row.text),
                    y_true=int(row.y_true),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries,
                    initial_retry_delay=args.initial_retry_delay,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(
                        predict_one_row,
                        api_key=api_key,
                        model=args.model,
                        prompt_file_name=f"INLINE::{variation}",
                        system_prompt=system_prompt,
                        schema_bundle=schema_bundle,
                        row_id=row.ID,
                        text_input=str(row.text),
                        y_true=int(row.y_true),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_output_tokens=args.max_output_tokens,
                        max_retries=args.max_retries,
                        initial_retry_delay=args.initial_retry_delay,
                    ): idx
                    for idx, row in enumerate(rows)
                }
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=variation):
                    idx = future_to_idx[future]
                    try:
                        records[idx] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        row = rows[idx]
                        records[idx] = {
                            "ID": row.ID,
                            "text": str(row.text),
                            "gold_label": "yes" if int(row.y_true) == 1 else "no",
                            "pred_label": "no",
                            "y_true": int(row.y_true),
                            "y_pred": 0,
                            "y_score": 0.0,
                            "parse_success": False,
                            "error": f"WorkerError: {type(exc).__name__}: {exc}",
                            "encoding_evidence": "[]",
                            "decoded_meaning": "[]",
                            "mechanism": "[]",
                            "raw_json": "",
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "prompt_file": f"INLINE::{variation}",
                            "model": args.model,
                        }

        final_records: list[dict[str, Any]] = [r for r in records if r is not None]
        if len(final_records) != len(rows):
            raise RuntimeError(f"Internal error: expected {len(rows)} records, got {len(final_records)}")

        predictions_df = pd.DataFrame.from_records(final_records)
        final_output_cols = ["ID", "y_true", "y_pred", "encoding_evidence", "decoded_meaning", "mechanism"]
        predictions_df = predictions_df[final_output_cols]
        predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

        y_true_list = [int(r["y_true"]) for r in final_records]
        y_pred_list = [int(r["y_pred"]) for r in final_records]
        y_score_list = [float(r["y_score"]) for r in final_records]
        parse_success_count = sum(1 for r in final_records if bool(r.get("parse_success")))

        metrics = compute_metrics(y_true_list, y_pred_list, y_score_list)
        metrics_row = {
            "variation": variation,
            "model": args.model,
            "n_rows": len(final_records),
            "parse_success_rate": parse_success_count / len(final_records) if final_records else 0.0,
            **metrics,
            "predictions_csv": str(predictions_path),
        }
        metrics_rows.append(metrics_row)

        (variation_dir / "metrics.json").write_text(json.dumps(metrics_row, indent=2), encoding="utf-8-sig")
        print(
            "  "
            + ", ".join(
                [
                    f"acc={metrics_row['accuracy']:.4f}",
                    f"macro_p={metrics_row['precision_macro']:.4f}",
                    f"macro_r={metrics_row['recall_macro']:.4f}",
                    f"macro_f1={metrics_row['f1_macro']:.4f}",
                    f"auc={metrics_row['auc']:.4f}" if metrics_row["auc"] == metrics_row["auc"] else "auc=nan",
                    f"parse_success={metrics_row['parse_success_rate']:.4f}",
                ]
            )
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False)
    metrics_csv = run_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    # Write markdown summary without requiring optional pandas/tabulate dependency.
    metrics_md = run_dir / "metrics_summary.md"
    header = "| " + " | ".join(metrics_df.columns.tolist()) + " |\n"
    sep = "| " + " | ".join(["---"] * len(metrics_df.columns)) + " |\n"
    rows = []
    for _, row in metrics_df.iterrows():
        vals = [str(row[col]) for col in metrics_df.columns]
        rows.append("| " + " | ".join(vals) + " |\n")
    metrics_md.write_text(header + sep + "".join(rows), encoding="utf-8-sig")

    print("\nBenchmark complete.")
    print(f"Run directory: {run_dir}")
    print(f"Metrics summary: {metrics_csv}")


if __name__ == "__main__":
    main()
