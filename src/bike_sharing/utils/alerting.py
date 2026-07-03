import json
import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from bike_sharing.utils.command_utils import run_command

logger = logging.getLogger(__name__)


def create_github_issue(title: str, body: str, labels: list[str], dedup_hours: int = 24) -> None:
    """
    Create a GitHub issue via `gh issue create`, unless an open issue with the
    same primary label was already created within the last dedup_hours — this
    stops an ongoing, already-reported condition (e.g. drift still present
    every hour) from spamming a new issue every run.
    """
    dedup_label = labels[0]
    result = run_command(
        ["gh", "issue", "list", "--state", "open", "--label", dedup_label, "--json", "createdAt"],
        capture_output=True,
    )
    existing = json.loads(result.stdout)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=dedup_hours)
    for issue in existing:
        created_at = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
        if created_at > cutoff:
            logger.info(
                f"Skipping issue creation — an open '{dedup_label}' issue already exists "
                f"from within the last {dedup_hours}h"
            )
            return

    run_command(
        ["gh", "issue", "create", "--title", title, "--body", body, "--label", ",".join(labels)]
    )
    logger.info(f"Created GitHub issue: {title}")


def send_email(subject: str, body: str, to: str) -> None:
    """
    Send an email via Gmail SMTP. A missing/invalid credential is logged and
    swallowed rather than raised — a misconfigured secret shouldn't fail the
    predict/retrain job this alert is attached to.
    """
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    if not username or not password:
        logger.error("SMTP_USERNAME/SMTP_PASSWORD not set — skipping email")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = to

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        logger.info(f"Email sent to {to}: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
