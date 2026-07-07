import json
from datetime import datetime, timedelta, timezone

from bike_sharing.utils import alerting


class _FakeCompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout


# ── create_github_issue ─────────────────────────────────────────────────────


def test_create_github_issue_creates_when_no_recent_duplicate(monkeypatch):
    calls = []

    def fake_run_command(cmd, capture_output=False):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            return _FakeCompletedProcess(stdout="[]")
        return _FakeCompletedProcess(stdout="")

    monkeypatch.setattr(alerting, "run_command", fake_run_command)

    alerting.create_github_issue("Title", "Body", ["output-drift-alert"])

    assert any(c[:3] == ["gh", "issue", "create"] for c in calls)


def test_create_github_issue_skips_when_recent_duplicate_exists(monkeypatch):
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    calls = []

    def fake_run_command(cmd, capture_output=False):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            return _FakeCompletedProcess(stdout=json.dumps([{"createdAt": recent}]))
        return _FakeCompletedProcess(stdout="")

    monkeypatch.setattr(alerting, "run_command", fake_run_command)

    alerting.create_github_issue("Title", "Body", ["output-drift-alert"], dedup_hours=24)

    assert not any(c[:3] == ["gh", "issue", "create"] for c in calls)


def test_create_github_issue_creates_when_existing_is_older_than_window(monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
    calls = []

    def fake_run_command(cmd, capture_output=False):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            return _FakeCompletedProcess(stdout=json.dumps([{"createdAt": old}]))
        return _FakeCompletedProcess(stdout="")

    monkeypatch.setattr(alerting, "run_command", fake_run_command)

    alerting.create_github_issue("Title", "Body", ["output-drift-alert"], dedup_hours=24)

    assert any(c[:3] == ["gh", "issue", "create"] for c in calls)


# ── _markdown_to_html ────────────────────────────────────────────────────────


def test_markdown_to_html_converts_headers_and_bullets():
    body = "## Section\n\n- one\n- two\n\nplain line"

    result = alerting._markdown_to_html(body)

    assert "<h2>Section</h2>" in result
    assert "<ul>" in result
    assert "<li>one</li>" in result
    assert "<li>two</li>" in result
    assert "</ul>" in result
    assert "<p>plain line</p>" in result


# ── send_email ───────────────────────────────────────────────────────────────


class _FakeSMTP:
    instances = []

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.login_calls = []
        self.sent_messages = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        pass

    def login(self, username, password):
        self.login_calls.append((username, password))

    def send_message(self, msg):
        self.sent_messages.append(msg)


def test_send_email_sends_expected_message(monkeypatch):
    monkeypatch.setenv("SMTP_USERNAME", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    _FakeSMTP.instances = []
    monkeypatch.setattr(alerting.smtplib, "SMTP", _FakeSMTP)

    alerting.send_email("Subject", "Body text", "target@example.com")

    smtp = _FakeSMTP.instances[0]
    assert smtp.login_calls == [("me@gmail.com", "app-password")]
    sent = smtp.sent_messages[0]
    assert sent["Subject"] == "Subject"
    assert sent["To"] == "target@example.com"
    assert sent.get_content_type() == "text/html"
    assert sent.get_payload(decode=True).decode().strip() == "<p>Body text</p>"


def test_send_email_logs_and_continues_when_credentials_missing(monkeypatch, capsys):
    """
    A missing secret must not raise, but it also must not be invisible — a
    plain log line is easy to miss in CI, so it's also printed as a GitHub
    Actions warning annotation (shows up in the run UI without opening logs).
    """
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    instantiated = []
    monkeypatch.setattr(alerting.smtplib, "SMTP", lambda *a, **k: instantiated.append(True))

    alerting.send_email("Subject", "Body", "target@example.com")

    assert instantiated == []
    assert "::warning::" in capsys.readouterr().out


def test_send_email_annotates_when_smtp_raises(monkeypatch, capsys):
    """Same visibility guarantee when SMTP itself fails (bad password, network, etc.)."""
    monkeypatch.setenv("SMTP_USERNAME", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")

    class _RaisingSMTP:
        def __init__(self, *a, **k):
            raise OSError("connection refused")

    monkeypatch.setattr(alerting.smtplib, "SMTP", _RaisingSMTP)

    alerting.send_email("Subject", "Body", "target@example.com")

    assert "::warning::" in capsys.readouterr().out
