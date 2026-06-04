import os
from functools import wraps
from inspect import signature
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests
from prefect import flow
from prefect.context import MissingContextError, get_run_context


class SlackWebhookClient:
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        enabled: bool = True,
        timeout_seconds: int = 20,
        username: Optional[str] = None,
        icon_url: Optional[str] = None,
    ):
        self.webhook_url = webhook_url if webhook_url is not None else os.getenv("SLACK_WEBHOOK_URL")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.username = username
        self.icon_url = icon_url

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.webhook_url)

    def post(self, text: str, **extra_payload: Any) -> bool:
        if not self.is_configured:
            return False

        payload = {"text": text}
        if self.username:
            payload["username"] = self.username
        if self.icon_url:
            payload["icon_url"] = self.icon_url
        payload.update(extra_payload)

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Slack notification failed: {type(exc).__name__}: {exc}")
            return False
        return True


SENSITIVE_DETAIL_PARTS = ("api_key", "apikey", "key", "password", "secret", "token", "webhook")
MAX_DETAIL_VALUE_LENGTH = 200
MAX_SLACK_DETAIL_LENGTH = 120
MAX_SLACK_DETAILS = 12
WORKFLOW_START_DETAIL_KEYS = {
    "chunk-documents": ("source_blob", "dest_blob", "section_limit", "max_workers"),
    "generate-query-dataset": (
        "source_blob",
        "dest_prefix",
        "document_limit",
        "max_workers",
        "model",
        "query_type_sample_seed",
    ),
    "sample-chunk-eval-query-dataset": (
        "source_blob",
        "training_dataset_prefix",
        "dest_prefix",
        "sample_size",
        "sample_seed",
        "model",
        "query_type_sample_seed",
    ),
    "train-embeddings": (
        "source_prefix",
        "model_prefix",
        "epochs",
        "batch_size",
        "learning_rate",
        "validation_fraction",
        "validation_seed",
    ),
    "visualize-chunk-samples": ("source_blob", "dest_blob", "sample_count"),
    "visualize-query-dataset": ("source_prefix", "dest_blob", "sample_count"),
    "visualize-eval-query-dataset": ("source_prefix", "dest_blob"),
    "evaluate-embeddings": (
        "eval_dataset_prefix",
        "report_prefix",
        "evaluation_backend",
        "latest_model_prefix",
        "gcs_model_prefix",
    ),
    "evaluate-embeddings-base": (
        "eval_dataset_prefix",
        "report_prefix",
        "evaluation_backend",
        "base_model_repo_id",
    ),
    "evaluate-embeddings-gemini": (
        "eval_dataset_prefix",
        "report_prefix",
        "evaluation_backend",
        "gemini_cache_path",
        "gemini_max_workers",
    ),
    "compare-embedding-evaluations": (
        "evaluation_a_prefix",
        "evaluation_b_prefix",
        "report_prefix",
        "metric",
        "rope",
        "posterior_draws",
    ),
    "visualize-embedding-retrieval": (
        "eval_dataset_prefix",
        "eval_report_prefix",
        "dest_blob",
        "sample_count",
        "sample_seed",
    ),
    "probe-gpu-availability": ("dest_blob", "matrix_size"),
}
TWEMOJI_BASE_URL = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72"
EVALUATION_SLACK_ICON_URL = (
    "https://static.vecteezy.com/system/resources/previews/056/231/941/large_2x/"
    "cat-wearing-a-detective-hat-and-holding-a-magnifying-glass-showcases-a-curious-expression-in-a-playful-setup-photo.jpeg"
)
WORKFLOW_SLACK_ICON_URLS = {
    "chunk-documents": f"{TWEMOJI_BASE_URL}/2702.png",
    "generate-query-dataset": f"{TWEMOJI_BASE_URL}/1f50d.png",
    "train-embeddings": "https://i.pinimg.com/736x/fb/cc/43/fbcc43003276a217c2c80b241342f53c.jpg",
    "visualize-chunk-samples": f"{TWEMOJI_BASE_URL}/1f4ca.png",
    "visualize-query-dataset": f"{TWEMOJI_BASE_URL}/1f4ca.png",
    "sample-chunk-eval-query-dataset": EVALUATION_SLACK_ICON_URL,
    "visualize-eval-query-dataset": EVALUATION_SLACK_ICON_URL,
    "evaluate-embeddings": EVALUATION_SLACK_ICON_URL,
    "evaluate-embeddings-base": EVALUATION_SLACK_ICON_URL,
    "evaluate-embeddings-gemini": EVALUATION_SLACK_ICON_URL,
    "compare-embedding-evaluations": EVALUATION_SLACK_ICON_URL,
    "visualize-embedding-retrieval": EVALUATION_SLACK_ICON_URL,
    "test-cache-volume": f"{TWEMOJI_BASE_URL}/1f5c3.png",
    "probe-runtime": f"{TWEMOJI_BASE_URL}/1f6f0.png",
    "probe-gpu-availability": "https://i.pinimg.com/736x/fb/cc/43/fbcc43003276a217c2c80b241342f53c.jpg",
    "test-slack-webhook": f"{TWEMOJI_BASE_URL}/1f9ea.png",
    "create-dataset": f"{TWEMOJI_BASE_URL}/1f4da.png",
}


def prefect_flow_run_url() -> Optional[str]:
    try:
        context = get_run_context()
    except MissingContextError:
        return None

    flow_run = getattr(context, "flow_run", None)
    flow_run_id = getattr(flow_run, "id", None)
    if flow_run_id is None:
        task_run = getattr(context, "task_run", None)
        flow_run_id = getattr(task_run, "flow_run_id", None)
    if not flow_run_id:
        return None

    ui_url = os.getenv("PREFECT_UI_URL")
    if not ui_url:
        api_url = os.getenv("PREFECT_API_URL")
        if not api_url:
            return None
        ui_url = api_url.removesuffix("/api").rstrip("/")

    return f"{ui_url.rstrip('/')}/runs/flow-run/{flow_run_id}"


def slack_notified_flow(
    *,
    workflow_name: Optional[str] = None,
    detail_keys: Optional[tuple[str, ...]] = None,
    **flow_kwargs: Any,
) -> Callable:
    def decorator(fn: Callable) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            details = workflow_start_details(fn, args, kwargs, detail_keys)
            details.pop("workflow_name", None)
            bound = signature(fn).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            resolved_workflow_name = bound.arguments.get("workflow_name") or workflow_name or fn.__name__
            notify_workflow_started(resolved_workflow_name, details)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                notify_workflow_failed(resolved_workflow_name, exc, details)
                raise

        return flow(**flow_kwargs)(wrapper)

    return decorator


def workflow_start_details(
    fn: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    detail_keys: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    bound = signature(fn).bind_partial(*args, **kwargs)
    bound.apply_defaults()

    items = bound.arguments.items()
    if detail_keys is not None:
        selected_keys = set(detail_keys)
        items = ((key, value) for key, value in items if key in selected_keys)

    details = {}
    for key, value in items:
        display_value = safe_detail_value(key, value)
        if display_value is not None:
            details[key] = display_value
    return details


def safe_detail_value(key: str, value: Any) -> Optional[Any]:
    normalized_key = key.lower()
    if any(part in normalized_key for part in SENSITIVE_DETAIL_PARTS):
        return "[redacted]"
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_DETAIL_VALUE_LENGTH:
            return value
        return f"{value[:MAX_DETAIL_VALUE_LENGTH]}..."
    return None


def workflow_slack_client(workflow_name: str) -> SlackWebhookClient:
    icon_url = workflow_icon_url(workflow_name)
    return SlackWebhookClient(
        username="ml-workflows",
        icon_url=icon_url,
    )


def workflow_icon_url(workflow_name: str) -> Optional[str]:
    env_key = f"SLACK_ICON_URL_{_workflow_env_key(workflow_name)}"
    return os.getenv(env_key) or WORKFLOW_SLACK_ICON_URLS.get(workflow_name)


def _workflow_env_key(workflow_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in workflow_name.upper())


def _select_workflow_start_details(workflow_name: str, details: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not details:
        return details
    keys = WORKFLOW_START_DETAIL_KEYS.get(workflow_name)
    if not keys:
        return details
    return {key: details[key] for key in keys if key in details and not _skip_slack_detail(details[key])}


def _post_workflow_message(
    client: SlackWebhookClient,
    workflow_name: str,
    text: str,
    blocks: list[dict[str, Any]],
) -> bool:
    icon_url = workflow_icon_url(workflow_name)
    payload = {"blocks": blocks}
    if icon_url:
        payload["icon_url"] = icon_url
    return client.post(text, **payload)


def notify_workflow_started(
    workflow_name: str,
    details: Optional[dict[str, Any]] = None,
    client: Optional[SlackWebhookClient] = None,
) -> bool:
    client = client or workflow_slack_client(workflow_name)
    run_url = prefect_flow_run_url()
    text = f"{workflow_name} run started"
    blocks = _workflow_blocks(
        title=text,
        status="started",
        run_url=run_url,
        details=_select_workflow_start_details(workflow_name, details),
    )
    return _post_workflow_message(client, workflow_name, text, blocks)


def notify_workflow_failed(
    workflow_name: str,
    exc: BaseException,
    details: Optional[dict[str, Any]] = None,
    client: Optional[SlackWebhookClient] = None,
) -> bool:
    client = client or workflow_slack_client(workflow_name)
    run_url = prefect_flow_run_url()
    text = f"{workflow_name} run failed"
    failure_details = {
        "error": f"{type(exc).__name__}: {exc}",
        **(details or {}),
    }
    blocks = _workflow_blocks(
        title=text,
        status="failed",
        run_url=run_url,
        details=failure_details,
    )
    return _post_workflow_message(client, workflow_name, text, blocks)


def notify_training_validation_metric(
    *,
    workflow_name: str,
    epoch: float,
    total_epochs: int,
    steps: int,
    metrics: dict[str, Any],
    details: Optional[dict[str, Any]] = None,
    client: Optional[SlackWebhookClient] = None,
) -> bool:
    client = client or workflow_slack_client(workflow_name)
    run_url = prefect_flow_run_url()
    epoch_label = _format_epoch(epoch)
    progress_label = details.get("epoch_progress") if details else None
    text = f"{workflow_name} progress: epoch {epoch_label}/{total_epochs}"
    if progress_label:
        text = f"{text} {progress_label}"
    metric_details = {
        "epoch": f"{epoch_label}/{total_epochs}",
        "steps": steps,
        **metrics,
        **(details or {}),
    }
    blocks = _workflow_blocks(
        title=text,
        status="validation",
        run_url=run_url,
        details=metric_details,
    )
    return _post_workflow_message(client, workflow_name, text, blocks)


def notify_workflow_event(
    *,
    workflow_name: str,
    title: str,
    status: str,
    details: Optional[dict[str, Any]] = None,
    client: Optional[SlackWebhookClient] = None,
) -> bool:
    client = client or workflow_slack_client(workflow_name)
    blocks = _workflow_blocks(
        title=title,
        status=status,
        run_url=prefect_flow_run_url(),
        details=details,
    )
    return _post_workflow_message(client, workflow_name, title, blocks)


@dataclass
class SlackProgressReporter:
    workflow_name: str
    total_units: int
    client: SlackWebhookClient = field(default_factory=SlackWebhookClient)
    notify_every_fraction: float = 0.1
    unit_label: str = "sections"
    _next_fraction: float = field(init=False, default=0.1)

    def __post_init__(self) -> None:
        if self.total_units < 0:
            raise ValueError("total_units must not be negative for SlackProgressReporter.")

    def notify_start(self, details: Optional[dict[str, Any]] = None) -> None:
        run_url = prefect_flow_run_url()
        text = f"{self.workflow_name} started"
        start_details = {
            f"total_{self.unit_label}": self.total_units,
            **(details or {}),
        }
        blocks = _workflow_blocks(
            title=text,
            status="started",
            run_url=run_url,
            details=start_details,
        )
        _post_workflow_message(self.client, self.workflow_name, text, blocks)

    def notify_progress_if_due(self, completed_units: int, details: Optional[dict[str, Any]] = None) -> None:
        if self.total_units == 0 or completed_units <= 0:
            return

        fraction_complete = min(1.0, completed_units / self.total_units)
        if fraction_complete + 1e-12 < self._next_fraction:
            return

        crossed_fraction = self._next_fraction
        while self._next_fraction <= fraction_complete + 1e-12:
            crossed_fraction = self._next_fraction
            self._next_fraction += self.notify_every_fraction

        percent = min(100, round(crossed_fraction * 100))
        run_url = prefect_flow_run_url()
        text = f"{self.workflow_name} progress: {percent}%"
        progress_details = {
            "progress": f"{percent}%",
            self.unit_label: f"{min(completed_units, self.total_units)}/{self.total_units}",
            **(details or {}),
        }
        blocks = _workflow_blocks(
            title=text,
            status="in progress",
            run_url=run_url,
            details=progress_details,
        )
        _post_workflow_message(self.client, self.workflow_name, text, blocks)

    def notify_success(self, details: Optional[dict[str, Any]] = None) -> None:
        run_url = prefect_flow_run_url()
        text = f"{self.workflow_name} completed"
        blocks = _workflow_blocks(
            title=text,
            status="completed",
            run_url=run_url,
            details=details,
        )
        _post_workflow_message(self.client, self.workflow_name, text, blocks)

    def notify_failure(self, exc: BaseException, details: Optional[dict[str, Any]] = None) -> None:
        run_url = prefect_flow_run_url()
        text = f"{self.workflow_name} failed"
        failure_details = {
            "error": f"{type(exc).__name__}: {exc}",
            **(details or {}),
        }
        blocks = _workflow_blocks(
            title=text,
            status="failed",
            run_url=run_url,
            details=failure_details,
        )
        _post_workflow_message(self.client, self.workflow_name, text, blocks)


def _format_details(details: Optional[dict[str, Any]]) -> list[str]:
    if not details:
        return []
    return [f"{key}: {value}" for key, value in details.items()]


def _format_epoch(epoch: float) -> str:
    if float(epoch).is_integer():
        return str(int(epoch))
    return f"{epoch:.2f}".rstrip("0").rstrip(".")


def _workflow_blocks(
    *,
    title: str,
    status: str,
    run_url: Optional[str],
    details: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{_escape_mrkdwn(title)}*",
            },
        }
    ]

    context_elements = [{"type": "mrkdwn", "text": f"*Status:* `{_escape_mrkdwn(status)}`"}]
    if run_url:
        context_elements.append({"type": "mrkdwn", "text": f"<{run_url}|Open Prefect run>"})
    blocks.append({"type": "context", "elements": context_elements})

    detail_text = _details_text(details)
    if detail_text:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": detail_text,
                },
            }
        )

    return blocks


def _details_text(details: Optional[dict[str, Any]]) -> str:
    if not details:
        return ""

    lines = []
    for key, value in details.items():
        if _skip_slack_detail(value):
            continue
        lines.append(f"*{_escape_mrkdwn(str(key))}:* {_format_detail_value_for_slack(value)}")
        if len(lines) >= MAX_SLACK_DETAILS:
            break
    return "\n".join(lines)


def _skip_slack_detail(value: Any) -> bool:
    return value is None or value == ""


def _format_detail_value_for_slack(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (int, float)):
        return f"`{value}`"

    text = _escape_mrkdwn(str(value))
    if len(text) > MAX_SLACK_DETAIL_LENGTH:
        text = f"{text[:MAX_SLACK_DETAIL_LENGTH]}..."
    if "\n" in text:
        return f"`{text.replace(chr(10), ' ')}`"
    return f"`{text}`"


def _escape_mrkdwn(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
