import os
from datetime import datetime, timezone

import requests
from prefect import flow, task


@task(log_prints=True)
def post_slack_test_message(message: str, username: str | None = None) -> dict:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("Missing SLACK_WEBHOOK_URL environment variable.")

    payload = {
        "text": message,
    }
    if username:
        payload["username"] = username

    response = requests.post(webhook_url, json=payload, timeout=20)
    response.raise_for_status()

    print(f"Slack webhook test posted successfully: status_code={response.status_code}")
    return {
        "status": "success",
        "status_code": response.status_code,
        "response_text": response.text,
    }


@flow(log_prints=True)
def test_slack_webhook_flow(
    message: str = "ML Workflows Slack webhook test",
    include_timestamp: bool = True,
    username: str | None = "ml-workflows",
) -> dict:
    if include_timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
        message = f"{message}\nUTC: {timestamp}"

    return post_slack_test_message(message=message, username=username)


if __name__ == "__main__":
    test_slack_webhook_flow()
