import os
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


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


@dataclass
class SlackProgressReporter:
    workflow_name: str
    total_units: int
    client: SlackWebhookClient = field(default_factory=SlackWebhookClient)
    notify_every_fraction: float = 0.1
    _next_fraction: float = field(init=False, default=0.1)

    def __post_init__(self) -> None:
        if self.total_units < 0:
            raise ValueError("total_units must not be negative for SlackProgressReporter.")

    def notify_start(self, details: Optional[dict[str, Any]] = None) -> None:
        lines = [
            f"{self.workflow_name} started",
            f"Total sections: {self.total_units}",
        ]
        lines.extend(self._format_details(details))
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
            f"Sections: {min(completed_units, self.total_units)}/{self.total_units}",
        ]
        lines.extend(self._format_details(details))
        self.client.post("\n".join(lines))

    def notify_success(self, details: Optional[dict[str, Any]] = None) -> None:
        lines = [f"{self.workflow_name} completed"]
        lines.extend(self._format_details(details))
        self.client.post("\n".join(lines))

    def notify_failure(self, exc: BaseException, details: Optional[dict[str, Any]] = None) -> None:
        lines = [
            f"{self.workflow_name} failed",
            f"Error: {type(exc).__name__}: {exc}",
        ]
        lines.extend(self._format_details(details))
        self.client.post("\n".join(lines))

    def _format_details(self, details: Optional[dict[str, Any]]) -> list[str]:
        if not details:
            return []
        return [f"{key}: {value}" for key, value in details.items()]
