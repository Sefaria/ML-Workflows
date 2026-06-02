from datetime import datetime, timezone

from prefect import task

from utils.slack import SlackWebhookClient, slack_notified_flow


@task(log_prints=True)
def post_slack_test_message(message: str, username: str | None = None) -> dict:
    client = SlackWebhookClient(username=username)
    if not client.is_configured:
        raise ValueError("Missing SLACK_WEBHOOK_URL environment variable.")

    posted = client.post(
        message,
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Slack webhook test*\n```{message}```",
                },
            }
        ],
    )
    if not posted:
        raise RuntimeError("Slack webhook test post failed.")

    print("Slack webhook test posted successfully")
    return {
        "status": "success",
    }


@slack_notified_flow(workflow_name="test-slack-webhook", log_prints=True)
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
