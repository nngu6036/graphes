from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.registry import available_datasets
from grapher.utils.logging import get_logger

logger = get_logger(__name__)

EXCLUDE_FROM_METRIC_AGG = {
    "dataset",
    "model",
    "metric_family",
    "source_file",
    "runtime_seconds",
    "seed",
    "base_seed",
    "is_aggregate",
    "run_id",
}

METRIC_FAMILY_PRIORITY = {
    "polygraphscore_official": 30,
    "polygraphscore_classifier": 20,
}

METRIC_MODELS = ("grapher", "dhvae")


def _flatten_results(obj: dict[str, Any]) -> dict[str, Any]:
    protocol = obj.get("protocol", {}) or {}
    row: dict[str, Any] = {
        "dataset": obj.get("dataset"),
        "model": obj.get("model"),
        "metric_family": obj.get("metric_family"),
        "runtime_seconds": obj.get("runtime_seconds"),
        "is_aggregate": bool(obj.get("is_aggregate", False)),
    }
    if "seed" in protocol:
        row["seed"] = protocol["seed"]
    if "base_seed" in protocol:
        row["base_seed"] = protocol["base_seed"]
    if obj.get("run_id") is not None:
        row["run_id"] = obj.get("run_id")
    elif protocol.get("run_id") is not None:
        row["run_id"] = protocol.get("run_id")
    for k, v in (obj.get("results", {}) or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[k] = v
    return row


def _row_run_id(row: dict[str, Any]) -> int | None:
    value = row.get("run_id")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_by_run_ids(rows: list[dict[str, Any]], run_ids: set[int] | None) -> list[dict[str, Any]]:
    if run_ids is None:
        return rows
    filtered = []
    for row in rows:
        # Recompute requested subsets from individual run files. Existing
        # aggregate JSONs may cover a different set of run ids.
        if bool(row.get("is_aggregate", False)):
            continue
        row_run_id = _row_run_id(row)
        if row_run_id in run_ids or (row_run_id is None and 0 in run_ids):
            filtered.append(row)
    return filtered


def _filter_by_datasets_and_models(
    rows: list[dict[str, Any]],
    *,
    datasets: set[str] | None,
    models: set[str] | None,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        row_dataset = str(row.get("dataset", "")).lower()
        row_model = str(row.get("model", "")).lower()
        if datasets is not None and row_dataset not in datasets:
            continue
        if models is not None and row_model not in models:
            continue
        filtered.append(row)
    return filtered


def _is_numeric_series(series: pd.Series) -> bool:
    converted = pd.to_numeric(series.dropna(), errors="coerce")
    return len(converted) > 0 and converted.notna().all()


def _aggregate_individual_metric_rows(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0]
    out: dict[str, Any] = {
        "dataset": first.get("dataset"),
        "model": first.get("model"),
        "metric_family": first.get("metric_family"),
        "is_aggregate": True,
        "source_file": ";".join(group.get("source_file", pd.Series(dtype=str)).astype(str).tolist()),
    }
    if "runtime_seconds" in group.columns:
        vals = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
        if len(vals):
            out["runtime_seconds"] = float(vals.sum())
    for col in group.columns:
        if col in EXCLUDE_FROM_METRIC_AGG or col.endswith("_std"):
            continue
        values = group[col].dropna()
        if values.empty:
            continue
        if _is_numeric_series(values):
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            out[col] = mean
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = std
        else:
            unique = list(dict.fromkeys(map(str, values.tolist())))
            if len(unique) == 1:
                out[col] = unique[0]
    return out


def _select_or_build_aggregate_rows(long_df: pd.DataFrame, *, prefer_existing_aggregates: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "model", "metric_family"]
    for _, group in long_df.groupby(group_cols, dropna=False):
        aggregate_rows = group[group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else pd.DataFrame()
        if prefer_existing_aggregates and not aggregate_rows.empty:
            # Use the newest/last aggregate file for this metric family.
            rows.append(aggregate_rows.iloc[-1].to_dict())
        else:
            individual_rows = group[~group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else group
            if not individual_rows.empty:
                rows.append(_aggregate_individual_metric_rows(individual_rows))
    return pd.DataFrame(rows)


def _make_wide(selected_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in selected_df.columns if c not in {"dataset", "model", "metric_family", "source_file"}]
    merged_rows = []
    for (dataset, model), group in selected_df.groupby(["dataset", "model"], dropna=False):
        out: dict[str, Any] = {"dataset": dataset, "model": model}
        if "runtime_seconds" in group:
            vals = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
            if len(vals):
                out["evaluation_runtime_seconds"] = float(vals.sum())
        for col in metric_cols:
            if col in {"runtime_seconds"}:
                continue
            if col not in group:
                continue
            value_rows = group[group[col].notna()].copy()
            if value_rows.empty:
                continue
            value_rows["_metric_priority"] = value_rows["metric_family"].astype(str).map(METRIC_FAMILY_PRIORITY).fillna(0)
            value_rows = value_rows.sort_values(["_metric_priority"], ascending=False, kind="stable")
            vals = value_rows[col].tolist()
            families = value_rows["metric_family"].astype(str).tolist()
            if len(vals) > 1 and len(set(map(str, vals))) > 1:
                logger.warning(
                    "Multiple different values for %s/%s/%s from metric families %s: %s; keeping %s",
                    dataset,
                    model,
                    col,
                    families,
                    vals,
                    families[0],
                )
            out[col] = vals[0]
        merged_rows.append(out)
    return pd.DataFrame(merged_rows).sort_values(["dataset", "model"])


def _debug_metric_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for col in df.columns:
        if col in EXCLUDE_FROM_METRIC_AGG or col.endswith("_std"):
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        if _is_numeric_series(values):
            columns.append(col)
    return columns


def _format_debug_value(value: Any) -> str:
    if value is None:
        return "--"
    try:
        if pd.isna(value):
            return "--"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _relative_std_exceeds_threshold(mean: Any, std: Any, *, threshold: float = 0.20) -> tuple[bool, float | None]:
    try:
        mean_value = float(mean)
        std_value = float(std)
    except (TypeError, ValueError):
        return False, None
    if np.isnan(mean_value) or np.isnan(std_value):
        return False, None
    denominator = abs(mean_value)
    if denominator == 0:
        return std_value > 0, None
    ratio = std_value / denominator
    return ratio > threshold, ratio


def _print_debug_run_statistics(long_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    print("Aggregate debug: statistics used for aggregation")
    for (dataset, model, metric_family), group in long_df.groupby(["dataset", "model", "metric_family"], dropna=False):
        print("")
        print(f"{dataset} / {model} / {metric_family}")
        metric_cols = _debug_metric_columns(group)
        individual_rows = group[~group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else group
        aggregate_rows = group[group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else pd.DataFrame()
        selected_match = selected_df[
            (selected_df["dataset"].astype(str) == str(dataset))
            & (selected_df["model"].astype(str) == str(model))
            & (selected_df["metric_family"].astype(str) == str(metric_family))
        ]
        selected_row = selected_match.iloc[0] if not selected_match.empty else None
        selected_is_existing_aggregate = bool(selected_row is not None and selected_row.get("is_aggregate", False) and not aggregate_rows.empty)

        if selected_is_existing_aggregate:
            print("  aggregation input: existing aggregate row")
            values = [f"{col}={_format_debug_value(selected_row.get(col))}" for col in metric_cols]
            print("    aggregate: " + (", ".join(values) if values else "no aggregated numeric metrics"))
            if selected_row.get("source_file"):
                print(f"      source: {selected_row['source_file']}")
        elif individual_rows.empty:
            print("  contributing run ids: none")
        else:
            print(f"  contributing run ids: {len(individual_rows)}")
            sort_cols = [col for col in ["run_id", "seed"] if col in individual_rows.columns]
            sorted_rows = individual_rows.sort_values(sort_cols) if sort_cols else individual_rows
            for _, row in sorted_rows.iterrows():
                run_id = row.get("run_id")
                run_label = "run_id=default" if pd.isna(run_id) else f"run_id={int(run_id)}"
                values = [f"{col}={_format_debug_value(row.get(col))}" for col in metric_cols]
                source = row.get("source_file")
                print(f"    {run_label}: " + (", ".join(values) if values else "no aggregated numeric metrics"))
                if source:
                    print(f"      source: {source}")

        if selected_row is not None:
            summary_parts = []
            high_relative_std_parts = []
            for col in metric_cols:
                if col in selected_row.index and not pd.isna(selected_row.get(col)):
                    part = f"{col}_mean={_format_debug_value(selected_row.get(col))}"
                    std_col = f"{col}_std"
                    if std_col in selected_row.index and not pd.isna(selected_row.get(std_col)):
                        part += f", {std_col}={_format_debug_value(selected_row.get(std_col))}"
                        exceeds, ratio = _relative_std_exceeds_threshold(selected_row.get(col), selected_row.get(std_col))
                        if exceeds:
                            ratio_text = "undefined" if ratio is None else f"{ratio:.1%}"
                            high_relative_std_parts.append(
                                f"{col}: mean={_format_debug_value(selected_row.get(col))}, "
                                f"std={_format_debug_value(selected_row.get(std_col))}, std/mean={ratio_text}"
                            )
                    summary_parts.append(part)
            if summary_parts:
                print("  selected aggregate:")
                for part in summary_parts:
                    print(f"    {part}")
            if high_relative_std_parts:
                print("  high relative std (>20% of average):")
                for part in high_relative_std_parts:
                    print(f"    {part}")


def _print_aggregate_results(selected_df: pd.DataFrame) -> None:
    if selected_df.empty:
        return
    measurements: list[str] = []
    seen_measurements: set[str] = set()
    for col in selected_df.columns:
        if col in EXCLUDE_FROM_METRIC_AGG or col.endswith("_std") or col.endswith("_mean"):
            continue
        if f"{col}_std" not in selected_df.columns:
            continue
        values = pd.to_numeric(selected_df[col], errors="coerce")
        std_values = pd.to_numeric(selected_df[f"{col}_std"], errors="coerce")
        if values.notna().any() and std_values.notna().any() and col not in seen_measurements:
            seen_measurements.add(col)
            measurements.append(col)

    if measurements:
        print("Measurements:")
        for name in measurements:
            print(f"  {name}")

    print("Aggregate results:")
    for _, row in selected_df.sort_values(["dataset", "model", "metric_family"], kind="stable").iterrows():
        prefix = f"{row.get('dataset')}/{row.get('model')}/{row.get('metric_family')}"
        metric_names = []
        for col in selected_df.columns:
            if col in EXCLUDE_FROM_METRIC_AGG or col.endswith("_std") or col.endswith("_mean"):
                continue
            std_col = f"{col}_std"
            if std_col not in selected_df.columns:
                continue
            value = row.get(col)
            std = row.get(std_col)
            try:
                value = float(value)
                std = float(std)
            except (TypeError, ValueError):
                continue
            if np.isnan(value) or np.isnan(std):
                continue
            metric_names.append((col, value, std))
        for name, value, std in metric_names:
            print(f"{prefix} {name}: {_format_debug_value(value)} +- {_format_debug_value(std)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metric JSON outputs into long, metric-family, and wide CSV tables.")
    parser.add_argument("--metric-dir", type=str, default="outputs/metrics")
    parser.add_argument("--output-dir", type=str, default="outputs/tables")
    parser.add_argument("--datasets", nargs="+", choices=available_datasets(), default=None, help="Datasets to include. Defaults to all discovered datasets.")
    parser.add_argument("--models", nargs="+", choices=METRIC_MODELS, default=None, help="Metric model names to include. Defaults to all models found in metric files.")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Only average these run ids. Existing aggregate JSONs are ignored when this is set.")
    parser.add_argument("--debug", action="store_true", help="Print individual per-run statistics used for aggregation.")
    args = parser.parse_args()

    metric_dir = Path(args.metric_dir)
    output_dir = Path(args.output_dir)
    requested_run_ids = set(args.run_ids) if args.run_ids is not None else None
    requested_datasets = {dataset.lower() for dataset in args.datasets} if args.datasets is not None else None
    requested_models = {model.lower() for model in args.models} if args.models is not None else None
    rows = []
    for path in sorted(metric_dir.glob("**/*.json")):
        # Only metric JSONs should live under outputs/metrics, but skip hidden or
        # editor temp files defensively.
        if path.name.startswith("."):
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "results" not in obj:
            continue
        row = _flatten_results(obj)
        row["source_file"] = str(path)
        rows.append(row)

    rows = _filter_by_datasets_and_models(rows, datasets=requested_datasets, models=requested_models)
    rows = _filter_by_run_ids(rows, requested_run_ids)

    if not rows:
        requested_parts = []
        if requested_datasets is not None:
            requested_parts.append(f"datasets={sorted(requested_datasets)}")
        if requested_models is not None:
            requested_parts.append(f"models={sorted(requested_models)}")
        if requested_run_ids is not None:
            requested_parts.append(f"run_ids={sorted(requested_run_ids)}")
        requested = f" matching {', '.join(requested_parts)}" if requested_parts else ""
        if requested_run_ids is None:
            logger.info("No metric files found under %s%s", metric_dir, requested)
        else:
            logger.info("No metric files found under %s%s", metric_dir, requested)
        return

    long_df = pd.DataFrame(rows)
    long_out = output_dir / "aggregated_results_long.csv"
    long_out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_out, index=False)

    selected_df = _select_or_build_aggregate_rows(long_df, prefer_existing_aggregates=requested_run_ids is None)
    if args.debug:
        _print_debug_run_statistics(long_df, selected_df)
    selected_out = output_dir / "aggregated_results_by_metric_family.csv"
    selected_df.to_csv(selected_out, index=False)

    wide_df = _make_wide(selected_df)
    wide_out = output_dir / "aggregated_results.csv"
    wide_df.to_csv(wide_out, index=False)
    _print_aggregate_results(selected_df)
    logger.info("Saved long results to %s", long_out)
    logger.info("Saved metric-family aggregate results to %s", selected_out)
    logger.info("Saved wide results to %s", wide_out)


if __name__ == "__main__":
    main()
