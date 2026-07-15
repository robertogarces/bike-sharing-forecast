import html
import json
import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from bike_sharing.utils.command_utils import run_command

logger = logging.getLogger(__name__)


def _markdown_to_html(body: str) -> str:
    """
    Convert the small markdown subset used in alert bodies (## headers, -
    bullet lists, blank-line-separated paragraphs) into HTML — email clients
    render plain-text markdown syntax literally instead of formatting it.
    """
    lines_html = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            lines_html.append("</ul>")
            in_list = False

    for line in body.split("\n"):
        if line.startswith("## "):
            close_list()
            lines_html.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                lines_html.append("<ul>")
                in_list = True
            lines_html.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip() == "":
            close_list()
        else:
            close_list()
            lines_html.append(f"<p>{html.escape(line)}</p>")

    close_list()
    return "\n".join(lines_html)


def create_github_issue(title: str, body: str, labels: list[str], dedup_hours: int = 24) -> bool:
    """
    Create a GitHub issue via `gh issue create`, unless an open issue with the
    same primary label was already created within the last dedup_hours — this
    stops an ongoing, already-reported condition (e.g. drift still present
    every hour) from spamming a new issue every run.

    Returns
    -------
    bool
        True if a new issue was created, False if skipped as a duplicate.
        Callers use this to also gate the companion email alert, which has no
        dedup mechanism of its own.
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
            return False

    run_command(
        ["gh", "issue", "create", "--title", title, "--body", body, "--label", ",".join(labels)]
    )
    logger.info(f"Created GitHub issue: {title}")
    return True


def send_email(subject: str, body: str, to: str) -> None:
    """
    Send an email via Gmail SMTP. A missing/invalid credential is logged and
    swallowed rather than raised — a misconfigured secret shouldn't fail the
    predict/retrain job this alert is attached to. Also printed as a GitHub
    Actions warning annotation, since a plain log line is easy to miss — the
    run still shows green with no visible sign the alert was never delivered.
    """
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    if not username or not password:
        logger.error("SMTP_USERNAME/SMTP_PASSWORD not set — skipping email")
        print(f"::warning::Email alert '{subject}' not sent — SMTP credentials not set")
        return

    msg = MIMEText(_markdown_to_html(body), "html")
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
        print(f"::warning::Failed to send email alert '{subject}': {e}")
