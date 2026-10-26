import time
from typing import Optional

from prefect.deployments import run_deployment
from prefect.runtime import flow_run

from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "pretrain-supervisor"
A100_NODE_SELECTORS = {
    "spot": {
        "cloud.google.com/gke-accelerator": "nvidia-tesla-a100",
        "cloud.google.com/gke-provisioning": "spot",
    },
    "on_demand": {
        "cloud.google.com/gke-accelerator": "nvidia-tesla-a100",
        "cloud.google.com/gke-provisioning": "standard",
    },
}


def capacity_for_attempt(attempt: int, spot_attempts: int, allow_on_demand: bool) -> str:
    return "spot" if attempt <= spot_attempts or not allow_on_demand else "on_demand"


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def pretrain_supervisor_flow(
    deployment_name: str = "pretrain-modernbert-flow/pretrain-modernbert",
    run_name: Optional[str] = None,
    gpu_parameters: Optional[dict] = None,
    max_attempts: int = 6,
    spot_attempts: int = 4,
    allow_on_demand: bool = False,
    retry_delay_seconds: int = 120,
) -> str:
    """Run the GPU pretraining deployment and start it again after infrastructure crashes.

    A spot eviction or an unschedulable pod ends the GPU flow run as Crashed. The
    GPU flow resumes from its newest GCS checkpoint, so each retry continues the
    same run. With allow_on_demand, retries after spot_attempts crashes move to on-demand A100s.
    A flow that ends Failed (a code error) is not retried.
    """
    run_name = run_name or str(flow_run.id)
    parameters = {**(gpu_parameters or {}), "run_name": run_name}

    for attempt in range(1, max_attempts + 1):
        capacity = capacity_for_attempt(attempt, spot_attempts, allow_on_demand)
        print(f"Attempt {attempt}/{max_attempts} on {capacity} capacity (run_name={run_name})")
        gpu_run = run_deployment(
            name=deployment_name,
            parameters=parameters,
            job_variables={"node_selector": A100_NODE_SELECTORS[capacity]},
            timeout=None,
        )
        state = gpu_run.state
        if state.is_completed():
            return run_name
        if not state.is_crashed():
            raise RuntimeError(f"GPU run ended in state {state.name}; not retrying. Run id: {gpu_run.id}")

        notify_workflow_event(
            workflow_name=WORKFLOW_NAME,
            title=f"{WORKFLOW_NAME} attempt {attempt} crashed",
            status="retrying" if attempt < max_attempts else "giving up",
            details={"run_name": run_name, "capacity": capacity, "gpu_flow_run_id": str(gpu_run.id)},
        )
        if attempt < max_attempts:
            time.sleep(retry_delay_seconds)

    raise RuntimeError(f"Pretraining run {run_name} did not finish after {max_attempts} attempts.")
