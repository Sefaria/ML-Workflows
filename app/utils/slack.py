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
    ):
        self.webhook_url = webhook_url if webhook_url is not None else os.getenv("SLACK_WEBHOOK_URL")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.username = username

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.webhook_url)

    def post(self, text: str, **extra_payload: Any) -> bool:
        if not self.is_configured:
            return False

        payload = {"text": text}
        if self.username:
            payload["username"] = self.username
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


def prefect_flow_run_url() -> Optional[str]:
    try:
        context = get_run_context()
    except MissingContextError:
        return None

    flow_run_id = getattr(context.flow_run, "id", None)
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
            resolved_workflow_name = workflow_name or fn.__name__
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
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_DETAIL_VALUE_LENGTH:
            return value
        return f"{value[:MAX_DETAIL_VALUE_LENGTH]}..."
    return None


def notify_workflow_started(
    workflow_name: str,
    details: Optional[dict[str, Any]] = None,
    client: Optional[SlackWebhookClient] = None,
) -> bool:
    client = client or SlackWebhookClient(username="ml-workflows")
    lines = [f"{workflow_name} run started"]

    run_url = prefect_flow_run_url()
    if run_url:
        lines.append(f"Run: {run_url}")

    lines.extend(_format_details(details))
    return client.post("\n".join(lines))


def notify_workflow_failed(
    workflow_name: str,
    exc: BaseException,
    details: Optional[dict[str, Any]] = None,
    client: Optional[SlackWebhookClient] = None,
) -> bool:
    client = client or SlackWebhookClient(username="ml-workflows")
    lines = [
        f"{workflow_name} run failed",
        f"Error: {type(exc).__name__}: {exc}",
    ]

    run_url = prefect_flow_run_url()
    if run_url:
        lines.append(f"Run: {run_url}")

    lines.extend(_format_details(details))
    return client.post("\n".join(lines))


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
        lines = [
            f"{self.workflow_name} started",
            f"Total {self.unit_label}: {self.total_units}",
        ]
        run_url = prefect_flow_run_url()
        if run_url:
            lines.append(f"Run: {run_url}")
        lines.extend(_format_details(details))
        self.client.post("\n".join(lines))

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
        lines = [
            f"{self.workflow_name} progress: {percent}%",
            f"{self.unit_label.title()}: {min(completed_units, self.total_units)}/{self.total_units}",
        ]
        run_url = prefect_flow_run_url()
        if run_url:
            lines.append(f"Run: {run_url}")
        lines.extend(_format_details(details))
        self.client.post("\n".join(lines))

    def notify_success(self, details: Optional[dict[str, Any]] = None) -> None:
        lines = [f"{self.workflow_name} completed"]
        run_url = prefect_flow_run_url()
        if run_url:
            lines.append(f"Run: {run_url}")
        lines.extend(_format_details(details))
        self.client.post("\n".join(lines))

    def notify_failure(self, exc: BaseException, details: Optional[dict[str, Any]] = None) -> None:
        lines = [
            f"{self.workflow_name} failed",
            f"Error: {type(exc).__name__}: {exc}",
        ]
        run_url = prefect_flow_run_url()
        if run_url:
            lines.append(f"Run: {run_url}")
        lines.extend(_format_details(details))
        self.client.post("\n".join(lines))


def _format_details(details: Optional[dict[str, Any]]) -> list[str]:
    if not details:
        return []
    return [f"{key}: {value}" for key, value in details.items()]
