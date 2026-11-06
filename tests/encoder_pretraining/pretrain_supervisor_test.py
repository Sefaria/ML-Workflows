import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from prefect.states import Completed, Crashed, Failed  # noqa: E402

from encoder_pretraining.flows.pretrain_supervisor import was_interrupted  # noqa: E402


def test_crash_is_an_interruption():
    assert was_interrupted(Crashed(message="Execution was aborted by a termination signal."))


def test_dataloader_worker_killed_by_sigterm_is_an_interruption():
    # What the roshei tevot cross-encoder run ended with after a spot eviction on 2026-11-05.
    assert was_interrupted(Failed(message="Flow run encountered an exception: RuntimeError: DataLoader worker (pid 122) is killed by signal: Terminated. "))


def test_code_errors_and_success_are_not_interruptions():
    assert not was_interrupted(Failed(message="Flow run encountered an exception: ValueError: bad parameter"))
    assert not was_interrupted(Failed(message="DataLoader worker (pid 9) is killed by signal: Killed."))   # OOM: retrying would fail again
    assert not was_interrupted(Completed())
