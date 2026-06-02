import ctypes.util
import glob
import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prefect import task

from utils.gcs import upload_blob
from utils.slack import notify_workflow_event, slack_notified_flow


GPU_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "LD_LIBRARY_PATH",
    "PATH",
    "HOSTNAME",
    "PREFECT_API_URL",
    "PREFECT__FLOW_RUN_ID",
)


def _run_command(command: list[str], timeout_seconds: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _read_text_file(path: str, max_chars: int = 4000) -> str | None:
    try:
        return Path(path).read_text(errors="replace")[:max_chars]
    except OSError:
        return None


def _nvidia_device_files() -> list[str]:
    return sorted(glob.glob("/dev/nvidia*"))


def _torch_device_probe(device_index: int, matrix_size: int) -> dict[str, Any]:
    import torch

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    before = {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }

    first = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    second = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    product = first @ second
    checksum = float(product[0, 0].detach().cpu())
    torch.cuda.synchronize(device)

    after = {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }

    del first, second, product
    torch.cuda.empty_cache()

    return {
        "status": "ok",
        "device": str(device),
        "matrix_size": matrix_size,
        "checksum": checksum,
        "memory_before": before,
        "memory_after": after,
    }


def _torch_report(matrix_size: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "import_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = []
    allocation_tests = []

    for index in range(device_count):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
        except Exception as exc:
            devices.append(
                {
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        try:
            allocation_tests.append(_torch_device_probe(index, matrix_size))
        except Exception as exc:
            allocation_tests.append(
                {
                    "status": "failed",
                    "device": f"cuda:{index}",
                    "matrix_size": matrix_size,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "import_ok": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "device_count": device_count,
        "current_device": int(torch.cuda.current_device()) if cuda_available and device_count else None,
        "devices": devices,
        "allocation_tests": allocation_tests,
        "library_paths": {
            "cuda": ctypes.util.find_library("cuda"),
            "cudart": ctypes.util.find_library("cudart"),
            "nvidia_ml": ctypes.util.find_library("nvidia-ml"),
        },
    }

    try:
        from torch.utils.collect_env import get_pretty_env_info

        report["collect_env"] = get_pretty_env_info()
    except Exception as exc:
        report["collect_env_error"] = f"{type(exc).__name__}: {exc}"

    return report


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    torch_report = report.get("torch", {})
    allocation_tests = torch_report.get("allocation_tests") or []
    failed_tests = [test for test in allocation_tests if test.get("status") != "ok"]
    device_names = [
        device.get("name")
        for device in torch_report.get("devices", [])
        if device.get("name")
    ]
    nvidia_smi = report.get("commands", {}).get("nvidia_smi", {})
    nvidia_smi_ok = nvidia_smi.get("returncode") == 0

    cuda_available = bool(torch_report.get("cuda_available"))
    device_count = int(torch_report.get("device_count") or 0)
    allocation_ok = bool(allocation_tests) and not failed_tests
    status = "available" if cuda_available and device_count > 0 and allocation_ok else "unavailable"

    return {
        "status": status,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "allocation_ok": allocation_ok,
        "failed_allocation_tests": len(failed_tests),
        "nvidia_smi_ok": nvidia_smi_ok,
        "torch_cuda_version": torch_report.get("cuda_version"),
        "torch_version": torch_report.get("torch_version"),
        "devices": ", ".join(device_names) if device_names else None,
        "nvidia_visible_devices": report.get("environment", {}).get("NVIDIA_VISIBLE_DEVICES"),
        "cuda_visible_devices": report.get("environment", {}).get("CUDA_VISIBLE_DEVICES"),
    }


@task(log_prints=True)
def collect_gpu_availability_report(matrix_size: int = 512) -> dict[str, Any]:
    report = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "environment": {key: os.getenv(key) for key in GPU_ENV_KEYS},
        "device_files": _nvidia_device_files(),
        "proc_driver_nvidia_version": _read_text_file("/proc/driver/nvidia/version"),
        "commands": {
            "nvidia_smi": _run_command(["nvidia-smi"]),
            "nvidia_smi_list": _run_command(["nvidia-smi", "-L"]),
            "nvidia_smi_query": _run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
        "torch": _torch_report(matrix_size),
    }
    report["summary"] = _summarize_report(report)

    summary = report["summary"]
    print(
        "GPU availability probe: "
        f"status={summary['status']}, "
        f"cuda_available={summary['cuda_available']}, "
        f"device_count={summary['device_count']}, "
        f"allocation_ok={summary['allocation_ok']}, "
        f"nvidia_smi_ok={summary['nvidia_smi_ok']}, "
        f"devices={summary.get('devices')}"
    )

    notify_workflow_event(
        workflow_name="probe-gpu-availability",
        title="probe-gpu-availability result",
        status=summary["status"],
        details=summary,
    )
    return report


@task(log_prints=True)
def upload_report(report: dict[str, Any], bucket: str, blob_path: str) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        local_path = tmp.name

    try:
        print(f"Uploading GPU availability report to gs://{bucket}/{blob_path}")
        upload_blob(local_path, bucket, blob_path)
        return {
            "bucket": bucket,
            "blob": blob_path,
            "uri": f"gs://{bucket}/{blob_path}",
        }
    finally:
        Path(local_path).unlink(missing_ok=True)


@slack_notified_flow(workflow_name="probe-gpu-availability", log_prints=True)
def probe_gpu_availability_flow(
    dest_bucket: str,
    dest_blob: str,
    matrix_size: int = 512,
) -> None:
    report = collect_gpu_availability_report(matrix_size)
    upload = upload_report(report, dest_bucket, dest_blob)
    notify_workflow_event(
        workflow_name="probe-gpu-availability",
        title="probe-gpu-availability report uploaded",
        status="completed",
        details={
            "report_uri": upload["uri"],
            **report["summary"],
        },
    )
