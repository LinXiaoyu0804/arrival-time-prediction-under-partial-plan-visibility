"""Fast integrity and claim audit for the code-and-data release."""

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs/TRE_operational_batch_validation_20260913"
SOURCE = ROOT / "data/source/work_experiments/planned_actual_v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_source():
    metadata = json.loads((SOURCE / "audit/source_manifest.json").read_text(encoding="utf8"))
    workbook = SOURCE / "raw" / metadata["filename"]
    require(workbook.stat().st_size == metadata["size_bytes"], "source workbook size mismatch")
    require(sha256(workbook) == metadata["sha256"], "source workbook hash mismatch")
    return metadata["dataset_doi"]


def check_evidence_hashes():
    rows = list(csv.DictReader((EVIDENCE / "SHA256SUMS.csv").open(encoding="utf8")))
    failures = []
    for row in rows:
        path = EVIDENCE / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            failures.append(row["path"])
    require(not failures, f"evidence hash failures: {failures[:5]}")
    return len(rows)


def check_claims():
    comparisons = pd.read_csv(EVIDENCE / "tables/final_audit_driver_cluster_comparisons.csv")
    primary = comparisons.query("scope == 'pooled' and comparison == 'daily_DG_minus_uniform_DG'").iloc[0]
    total = comparisons.query("scope == 'pooled' and comparison == 'daily_DG_minus_no_query'").iloc[0]
    require(abs(primary.difference - 1.0443046073862137) < 1e-12, "primary estimate changed")
    require(primary.ci_low > 0 and abs(primary.ci_low - 0.5409965656183965) < 1e-12, "primary interval changed")
    require(abs(total.difference - 3.575567641154819) < 1e-12 and total.ci_low > 0, "total gain changed")

    repeated_intervals = [
        pd.read_csv(EVIDENCE / "tables/placebo_driver_cluster_comparisons.csv")
        .query("comparison == 'daily_DG_minus_uniform_DG'")
        .iloc[0][["difference", "ci_low", "ci_high"]].to_numpy(float),
        -pd.read_csv(EVIDENCE / "tables/alternative_metric_driver_cluster_comparisons.csv")
        .query("metric == 'mae'")
        .iloc[0][["difference", "ci_high", "ci_low"]].to_numpy(float),
        pd.read_csv(EVIDENCE / "tables/operational_stress_driver_cluster_comparisons.csv")
        .query("condition == 'proxy_clean' and k == 5 and comparison == 'daily_DG_minus_uniform_DG'")
        .iloc[0][["difference", "ci_low", "ci_high"]].to_numpy(float),
    ]
    waves = pd.read_csv(EVIDENCE / "tables/dispatch_wave_driver_cluster_comparisons.csv")
    repeated_intervals.extend(
        row[["difference", "ci_low", "ci_high"]].to_numpy(float)
        for _, row in waves[waves.wave_size.astype(str).isin(["50", "daily"])].iterrows()
    )
    primary_interval = primary[["difference", "ci_low", "ci_high"]].to_numpy(float)
    require(
        all(np.allclose(values, primary_interval, rtol=0, atol=1e-12) for values in repeated_intervals),
        "repeated primary intervals disagree",
    )

    tasks = pd.read_parquet(EVIDENCE / "tables/final_audit_selected_tasks.parquet")
    expected_marginal = tasks.groupby(["allocation", "within_route_priority_rank"]).agg(
        attempts=("clean_gain", "size"),
        mean_gain=("clean_gain", "mean"),
        median_gain=("clean_gain", "median"),
        positive_percent=("clean_gain", lambda x: 100 * x.gt(0).mean()),
        harm_percent=("clean_gain", lambda x: 100 * x.lt(0).mean()),
        over_30_percent=("clean_gain", lambda x: 100 * x.gt(30).mean()),
    ).reset_index()
    reported_marginal = pd.read_csv(EVIDENCE / "tables/final_audit_marginal_query_ordinal.csv")
    marginal_keys = ["allocation", "within_route_priority_rank"]
    reported_marginal = reported_marginal.sort_values(marginal_keys).reset_index(drop=True)
    expected_marginal = expected_marginal.sort_values(marginal_keys).reset_index(drop=True)
    require(reported_marginal[marginal_keys].equals(expected_marginal[marginal_keys]), "marginal table keys changed")
    numeric = [column for column in expected_marginal if column not in marginal_keys]
    require(
        np.allclose(reported_marginal[numeric], expected_marginal[numeric], rtol=0, atol=1e-12),
        "marginal table is not pooled from selected tasks",
    )
    capacity = pd.read_csv(EVIDENCE / "tables/final_audit_batch_capacity.csv")
    require(capacity.attempt_difference.abs().max() == 0, "query workloads do not match")
    integrity = pd.read_csv(EVIDENCE / "integrity_checks.csv")
    passed = integrity.passed.astype(str).str.lower().eq("true")
    require(passed.all() and len(integrity) == 16, "final evidence checks failed")
    return {
        "primary_gain": primary.difference,
        "primary_ci": [primary.ci_low, primary.ci_high],
    }


def check_release_scope():
    forbidden_dirs = {
        "TRE_operational_allocation_reconstructed_20260914",
        "sections",
        "paper",
        "manuscript",
    }
    forbidden_suffixes = {".tex", ".bib"}
    paths = [path for path in ROOT.rglob("*") if ".git" not in path.parts]
    found_dirs = sorted(path.as_posix() for path in paths if path.is_dir() and path.name in forbidden_dirs)
    found_files = sorted(path.as_posix() for path in paths if path.is_file() and path.suffix.lower() in forbidden_suffixes)
    require(not found_dirs and not found_files, "manuscript sources/builds are present in the release")


def main():
    check_release_scope()
    report = {
        "status": "pass",
        "source_doi": check_source(),
        "evidence_files_verified": check_evidence_hashes(),
        **check_claims(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
