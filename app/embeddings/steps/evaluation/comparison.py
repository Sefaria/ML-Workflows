from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BayesianComparisonConfig:
    metric: str = "ndcg@10"
    rope: float = 0.005
    posterior_draws: int = 20000
    seed: int = 613
    credible_interval_mass: float = 0.95
    confidence_level: float = 0.95
    batch_size: int = 1000


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_label(run_config: dict, explicit_label: Optional[str], fallback: str) -> str:
    if explicit_label:
        return explicit_label
    workflow = run_config.get("workflow") or {}
    encoder = run_config.get("encoder") or {}
    pieces = [
        workflow.get("model_source"),
        encoder.get("backend"),
        encoder.get("model") or encoder.get("model_path"),
    ]
    label = " / ".join(str(piece) for piece in pieces if piece)
    return label or fallback


def metric_by_query(rows: list[dict], metric: str) -> dict[str, float]:
    values = {}
    for row in rows:
        query_id = str(row["query_id"])
        metrics = row.get("metrics") or {}
        if metric not in metrics:
            raise KeyError(f"Metric {metric!r} missing for query_id={query_id}")
        values[query_id] = float(metrics[metric])
    return values


def build_paired_rows(a_rows: list[dict], b_rows: list[dict], metric: str) -> list[dict]:
    a_by_query = {str(row["query_id"]): row for row in a_rows}
    b_by_query = {str(row["query_id"]): row for row in b_rows}
    a_metrics = metric_by_query(a_rows, metric)
    b_metrics = metric_by_query(b_rows, metric)
    paired_query_ids = sorted(set(a_metrics) & set(b_metrics))
    if not paired_query_ids:
        raise ValueError("No overlapping query_id values found between evaluation reports.")

    paired_rows = []
    for query_id in paired_query_ids:
        a_value = a_metrics[query_id]
        b_value = b_metrics[query_id]
        a_row = a_by_query[query_id]
        b_row = b_by_query[query_id]
        paired_rows.append(
            {
                "query_id": query_id,
                "query_text": a_row.get("query_text") or b_row.get("query_text") or "",
                "query_type": a_row.get("query_type") or b_row.get("query_type"),
                "query_lang": a_row.get("query_lang") or b_row.get("query_lang"),
                "metric": metric,
                "a_value": a_value,
                "b_value": b_value,
                "difference": a_value - b_value,
            }
        )
    return paired_rows


def bayesian_bootstrap_mean_difference(differences: np.ndarray, config: BayesianComparisonConfig) -> np.ndarray:
    if differences.size == 0:
        raise ValueError("At least one paired difference is required.")
    if config.posterior_draws <= 0:
        raise ValueError("posterior_draws must be positive.")
    rng = np.random.default_rng(config.seed)
    draws = np.empty(config.posterior_draws, dtype=float)
    offset = 0
    while offset < config.posterior_draws:
        draw_count = min(config.batch_size, config.posterior_draws - offset)
        weights = rng.exponential(scale=1.0, size=(draw_count, differences.size))
        weights = weights / weights.sum(axis=1, keepdims=True)
        draws[offset : offset + draw_count] = weights @ differences
        offset += draw_count
    return draws


def summarize_posterior(draws: np.ndarray, differences: np.ndarray, config: BayesianComparisonConfig) -> dict:
    lower_q = (1.0 - config.credible_interval_mass) / 2.0
    upper_q = 1.0 - lower_q
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    ties = int(np.sum(differences == 0))
    p_a_practically_better = float(np.mean(draws > config.rope))
    p_b_practically_better = float(np.mean(draws < -config.rope))
    p_practically_equivalent = float(np.mean((draws >= -config.rope) & (draws <= config.rope)))

    if p_a_practically_better >= config.confidence_level:
        conclusion = "evidence_favors_a"
    elif p_b_practically_better >= config.confidence_level:
        conclusion = "evidence_favors_b"
    elif p_practically_equivalent >= config.confidence_level:
        conclusion = "practically_equivalent"
    else:
        conclusion = "inconclusive"

    return {
        "observed_mean_difference": float(np.mean(differences)),
        "observed_median_difference": float(np.median(differences)),
        "posterior_mean_difference": float(np.mean(draws)),
        "posterior_median_difference": float(np.median(draws)),
        "credible_interval_mass": config.credible_interval_mass,
        "credible_interval": {
            "lower": float(np.quantile(draws, lower_q)),
            "upper": float(np.quantile(draws, upper_q)),
        },
        "p_a_mean_greater_than_b": float(np.mean(draws > 0)),
        "p_b_mean_greater_than_a": float(np.mean(draws < 0)),
        "rope": config.rope,
        "p_a_practically_better": p_a_practically_better,
        "p_b_practically_better": p_b_practically_better,
        "p_practically_equivalent": p_practically_equivalent,
        "confidence_level": config.confidence_level,
        "conclusion": conclusion,
        "win_loss_tie": {
            "a_wins": wins,
            "b_wins": losses,
            "ties": ties,
        },
    }


def compare_evaluation_reports(
    *,
    a_per_query_path: str,
    b_per_query_path: str,
    a_summary_path: str,
    b_summary_path: str,
    a_run_config_path: str,
    b_run_config_path: str,
    output_dir: str,
    config: BayesianComparisonConfig,
    a_label: Optional[str] = None,
    b_label: Optional[str] = None,
) -> dict:
    a_rows = read_jsonl(a_per_query_path)
    b_rows = read_jsonl(b_per_query_path)
    a_summary = read_json(a_summary_path)
    b_summary = read_json(b_summary_path)
    a_run_config = read_json(a_run_config_path)
    b_run_config = read_json(b_run_config_path)
    resolved_a_label = model_label(a_run_config, a_label, "model_a")
    resolved_b_label = model_label(b_run_config, b_label, "model_b")

    paired_rows = build_paired_rows(a_rows, b_rows, config.metric)
    differences = np.array([row["difference"] for row in paired_rows], dtype=float)
    draws = bayesian_bootstrap_mean_difference(differences, config)
    posterior_summary = summarize_posterior(draws, differences, config)
    comparison_summary = {
        "status": "success",
        "method": "bayesian_bootstrap_paired_query_metric_difference",
        "metric": config.metric,
        "a_label": resolved_a_label,
        "b_label": resolved_b_label,
        "paired_query_count": len(paired_rows),
        "a_query_count": len(a_rows),
        "b_query_count": len(b_rows),
        "a_aggregate_metric": a_summary.get(config.metric),
        "b_aggregate_metric": b_summary.get(config.metric),
        "aggregate_difference": (
            float(a_summary[config.metric]) - float(b_summary[config.metric])
            if config.metric in a_summary and config.metric in b_summary
            else None
        ),
        "posterior": posterior_summary,
        "config": {
            "posterior_draws": config.posterior_draws,
            "seed": config.seed,
            "rope": config.rope,
            "credible_interval_mass": config.credible_interval_mass,
            "confidence_level": config.confidence_level,
        },
    }
    csv_summary = {
        "metric": config.metric,
        "a_label": resolved_a_label,
        "b_label": resolved_b_label,
        "paired_query_count": len(paired_rows),
        "observed_mean_difference": posterior_summary["observed_mean_difference"],
        "posterior_mean_difference": posterior_summary["posterior_mean_difference"],
        "ci_lower": posterior_summary["credible_interval"]["lower"],
        "ci_upper": posterior_summary["credible_interval"]["upper"],
        "p_a_mean_greater_than_b": posterior_summary["p_a_mean_greater_than_b"],
        "p_a_practically_better": posterior_summary["p_a_practically_better"],
        "p_practically_equivalent": posterior_summary["p_practically_equivalent"],
        "p_b_practically_better": posterior_summary["p_b_practically_better"],
        "conclusion": posterior_summary["conclusion"],
    }

    output_path = Path(output_dir)
    write_json(output_path / "comparison_summary.json", comparison_summary)
    write_csv(output_path / "comparison_summary.csv", [csv_summary])
    write_jsonl(output_path / "per_query_comparison.jsonl", paired_rows)
    return comparison_summary
