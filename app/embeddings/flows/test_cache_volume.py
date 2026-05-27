import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task

from utils.gcs import upload_blob


def _statvfs_report(path: str) -> dict:
    stat = os.statvfs(path)
    return {
        "path": path,
        "bytes_total": stat.f_frsize * stat.f_blocks,
        "bytes_available": stat.f_frsize * stat.f_bavail,
        "bytes_free": stat.f_frsize * stat.f_bfree,
        "bytes_used": stat.f_frsize * (stat.f_blocks - stat.f_bfree),
    }


@task(log_prints=True)
def write_cache_probe(cache_path: str, probe_subdir: str) -> dict:
    root = Path(cache_path)
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / probe_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    probe_payload = {
        "written_at": timestamp,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cache_path": str(root),
        "probe_subdir": str(target_dir),
    }

    probe_file = target_dir / "probe.json"
    probe_file.write_text(json.dumps(probe_payload, ensure_ascii=False, indent=2))
    listing = sorted(path.name for path in target_dir.iterdir())

    return {
        "collected_at": timestamp,
        "cache_path": str(root),
        "probe_directory": str(target_dir),
        "probe_file": str(probe_file),
        "probe_file_exists": probe_file.exists(),
        "probe_directory_listing": listing,
        "statvfs": _statvfs_report(str(root)),
    }


@task(log_prints=True)
def upload_report(report: dict, bucket: str, blob_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        local_path = tmp.name

    try:
        print(f"Uploading cache-volume test report to gs://{bucket}/{blob_path}")
        upload_blob(local_path, bucket, blob_path)
    finally:
        Path(local_path).unlink(missing_ok=True)


@flow(log_prints=True)
def test_cache_volume_flow(
    dest_bucket: str,
    dest_blob: str,
    cache_path: str = "/cache",
    probe_subdir: str = "prefect-cache-probe",
) -> None:
    report = write_cache_probe(cache_path, probe_subdir)
    upload_report(report, dest_bucket, dest_blob)
