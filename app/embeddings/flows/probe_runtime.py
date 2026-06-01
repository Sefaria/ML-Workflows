import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task

from utils.gcs import upload_blob
from utils.slack import notify_workflow_started


def _run_command(command: list[str]) -> dict:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "error": str(exc),
        }


def _statvfs_report(path: str) -> dict:
    stat = os.statvfs(path)
    return {
        "path": path,
        "bytes_total": stat.f_frsize * stat.f_blocks,
        "bytes_available": stat.f_frsize * stat.f_bavail,
        "bytes_free": stat.f_frsize * stat.f_bfree,
        "bytes_used": stat.f_frsize * (stat.f_blocks - stat.f_bfree),
    }


def _memory_report() -> dict:
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    available = {}
    for key in ("SC_AVPHYS_PAGES",):
        if key in os.sysconf_names:
            available[key] = os.sysconf(key)

    report = {
        "page_size": page_size,
        "bytes_total": page_size * total_pages,
    }
    if "SC_AVPHYS_PAGES" in available:
        report["bytes_available"] = page_size * available["SC_AVPHYS_PAGES"]
    return report


@task(log_prints=True)
def collect_runtime_report(scratch_path: str) -> dict:
    scratch_dir = Path(scratch_path)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    probe_file = scratch_dir / "prefect_runtime_probe.tmp"
    probe_bytes = b"prefect-runtime-probe\n"
    probe_file.write_bytes(probe_bytes)

    report = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
        "scratch_path": str(scratch_dir),
        "probe_write": {
            "path": str(probe_file),
            "bytes_written": len(probe_bytes),
            "exists_after_write": probe_file.exists(),
        },
        "environment": {
            "HOME": os.getenv("HOME"),
            "TMPDIR": os.getenv("TMPDIR"),
            "PREFECT_API_URL": os.getenv("PREFECT_API_URL"),
        },
        "statvfs": {
            "scratch_path": _statvfs_report(str(scratch_dir)),
            "tmp": _statvfs_report("/tmp"),
        },
        "commands": {
            "df_h": _run_command(["df", "-h"]),
            "df_h_tmp": _run_command(["df", "-h", str(scratch_dir)]),
            "mount": _run_command(["mount"]),
        },
        "memory": _memory_report(),
    }

    probe_file.unlink(missing_ok=True)
    return report


@task(log_prints=True)
def upload_report(report: dict, bucket: str, blob_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        local_path = tmp.name

    try:
        print(f"Uploading runtime report to gs://{bucket}/{blob_path}")
        upload_blob(local_path, bucket, blob_path)
    finally:
        Path(local_path).unlink(missing_ok=True)


@flow(log_prints=True)
def probe_runtime_flow(
    dest_bucket: str,
    dest_blob: str,
    scratch_path: str = "/tmp",
) -> None:
    notify_workflow_started(
        "probe-runtime",
        {
            "Destination": f"gs://{dest_bucket}/{dest_blob}",
            "Scratch path": scratch_path,
        },
    )
    report = collect_runtime_report(scratch_path)
    upload_report(report, dest_bucket, dest_blob)
