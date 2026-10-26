import json
import re
import time
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback

from utils.gcs import (
    blob_exists,
    delete_prefix,
    download_directory,
    list_blob_names,
    read_blob_text,
    upload_directory,
    write_blob_text,
)


CHECKPOINT_PATTERN = re.compile(r"/checkpoint-(\d+)/")


def latest_marker_blob(prefix: str) -> str:
    return f"{prefix.rstrip('/')}/latest.json"


def restore_latest_checkpoint(bucket: str, prefix: str, local_dir: str) -> Optional[str]:
    """Download the checkpoint named by the latest.json marker. Returns its local path, or None."""
    marker_blob = latest_marker_blob(prefix)
    if not blob_exists(bucket, marker_blob):
        return None
    marker = json.loads(read_blob_text(bucket, marker_blob))
    local_path = str(Path(local_dir) / f"checkpoint-{marker['step']}")
    print(f"Restoring checkpoint step {marker['step']} from gs://{bucket}/{marker['path']}")
    download_directory(bucket, marker["path"], local_path)
    return local_path


class GcsCheckpointCallback(TrainerCallback):
    """Save a checkpoint on a wall-clock interval and mirror it to GCS.

    Spot nodes can disappear with 30 seconds notice, so progress is bounded by
    the interval instead of by training steps. latest.json is written only after
    the upload finishes, so a half-uploaded checkpoint is never restored.
    """

    def __init__(self, bucket: str, prefix: str, interval_seconds: float, keep_last: int = 2):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.interval_seconds = interval_seconds
        self.keep_last = keep_last
        self._last_save = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_save = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self._last_save >= self.interval_seconds:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        remote_prefix = f"{self.prefix}/checkpoint-{step}"
        print(f"Uploading checkpoint step {step} to gs://{self.bucket}/{remote_prefix}")
        upload_directory(str(Path(args.output_dir) / f"checkpoint-{step}"), self.bucket, remote_prefix)
        write_blob_text(
            self.bucket,
            latest_marker_blob(self.prefix),
            json.dumps({"step": step, "path": remote_prefix}),
        )
        self._prune_remote()
        self._last_save = time.monotonic()

    def _prune_remote(self) -> None:
        steps = sorted(
            {int(match.group(1)) for name in list_blob_names(self.bucket, self.prefix) if (match := CHECKPOINT_PATTERN.search(name))}
        )
        for step in steps[: -self.keep_last]:
            delete_prefix(self.bucket, f"{self.prefix}/checkpoint-{step}")
