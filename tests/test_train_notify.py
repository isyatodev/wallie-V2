"""Training notifications — what the run says after the dashboard is gone.

Training is detached, so the process that knows the result is the supervisor,
not the UI. These tests pin the parts that fail quietly in production:

* the **text** of a finished/failed run (name, epochs, elapsed, the resume hint);
* the **webhook guard** — only Discord is ever contacted, whatever is in .env;
* **Windows** toast escaping plus the tray-balloon fallback;
* **notify_run** never raising and never announcing a user-initiated stop.

No toast is shown and no request leaves the machine: PowerShell and the HTTP
layer are faked, except for one round trip against a local server.
"""
from __future__ import annotations

import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tts import train_notify as tn


def finished(**extra) -> dict:
    job = {
        "status": "done", "voice_name": "pt_BR-luna-medium",
        "epoch": 300, "epochs": 300,
        "started_at": 1000.0, "finished_at": 1000.0 + 754,
        "exported": {"onnx": "C:/runs/pt_BR-luna-medium.onnx"},
        "error": "",
    }
    job.update(extra)
    return job


# ---------------------------------------------------------------------------
# The message
# ---------------------------------------------------------------------------

def test_message_describes_a_finished_run():
    title, body = tn.message(finished())
    assert title == "✓ Piper training finished — pt_BR-luna-medium"
    assert "300/300 epochs" in body
    assert "12m 34s" in body
    assert "pt_BR-luna-medium.onnx" in body


def test_message_describes_a_failure_and_points_at_the_checkpoint():
    job = finished(status="error", epoch=12, error="the dataset preprocess failed")
    job["finished_at"] = 1000.0 + 90
    title, body = tn.message(job)
    assert title.startswith("✗ Piper training failed")
    assert "stopped at 12/300 epochs" in body
    assert "the dataset preprocess failed" in body
    assert "checkpoint" in body


def test_message_is_empty_while_the_run_is_still_going():
    assert tn.message({"status": "running", "running": True}) == ("", "")


# ---------------------------------------------------------------------------
# The webhook guard
# ---------------------------------------------------------------------------

def test_webhook_that_is_not_discord_is_refused_not_posted(monkeypatch):
    assert "TRAIN_DISCORD_WEBHOOK" in tn.validate_webhook("")
    assert tn.validate_webhook("http://127.0.0.1:8000/collect")
    assert tn.validate_webhook("https://discord.com.evil.example/api/webhooks/1/a")
    assert tn.validate_webhook("https://discord.com/api/webhooks/1/a\nX: y")
    assert tn.validate_webhook("https://discord.com/api/webhooks/123/abc") == ""

    calls: list[str] = []
    monkeypatch.setattr(tn, "webhook_url", lambda: "http://127.0.0.1:8000/collect")
    monkeypatch.setattr(tn, "_post_json", lambda url, payload, **kw: calls.append(url) or "")
    status = tn.discord_message("t", "b")
    assert status.startswith("skipped:")
    assert calls == []


def test_discord_message_posts_title_and_body(monkeypatch):
    sent: dict = {}

    def _post(url, payload, timeout=tn.TIMEOUT_SEC):
        sent.update(url=url, payload=payload)
        return ""

    monkeypatch.setattr(tn, "_post_json", _post)
    status = tn.discord_message(
        "✓ Piper training finished — v", "300/300 epochs",
        "https://discord.com/api/webhooks/1/tok",
    )
    assert status == "sent"
    content = sent["payload"]["content"]
    assert "✓ Piper training finished — v" in content
    assert "300/300 epochs" in content
    # Never ping @everyone from a training hook.
    assert sent["payload"]["allowed_mentions"] == {"parse": []}
    assert sent["url"].endswith("/1/tok")


def test_post_json_speaks_http_and_network_errors(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            return b"{}"

    monkeypatch.setattr(tn.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert tn._post_json("https://discord.com/api/webhooks/1/t", {}) == ""

    def _raises(exc):
        def _open(*a, **k):
            raise exc
        return _open

    monkeypatch.setattr(
        tn.urllib.request, "urlopen",
        _raises(urllib.error.HTTPError(
            "u", 404, "Not Found", {}, io.BytesIO(b'{"message": "Unknown Webhook"}'),
        )),
    )
    assert "404" in tn._post_json("https://discord.com/api/webhooks/1/t", {})

    monkeypatch.setattr(
        tn.urllib.request, "urlopen",
        _raises(urllib.error.URLError("getaddrinfo failed")),
    )
    assert "getaddrinfo" in tn._post_json("https://discord.com/api/webhooks/1/t", {})


def test_post_json_repeats_discords_own_rejection(monkeypatch):
    """"Discord said 404" is useless; "Unknown Webhook (code 10015)" is not."""
    def _raises(body: bytes):
        def _open(*a, **k):
            raise urllib.error.HTTPError(
                "u", 404, "Not Found", {}, io.BytesIO(body),
            )
        return _open

    monkeypatch.setattr(
        tn.urllib.request, "urlopen",
        _raises(b'{"message": "Unknown Webhook", "code": 10015}'),
    )
    reason = tn._post_json("https://discord.com/api/webhooks/1/t", {})
    assert "404" in reason
    assert "Unknown Webhook" in reason
    assert "10015" in reason

    # A 401 carries "message" without a meaningful code: no "(code 0)" noise.
    monkeypatch.setattr(
        tn.urllib.request, "urlopen",
        _raises(b'{"message": "401: Unauthorized", "code": 0}'),
    )
    reason = tn._post_json("https://discord.com/api/webhooks/1/t", {})
    assert "401: Unauthorized" in reason and "code 0" not in reason

    # Anything that is not JSON still lands on screen instead of vanishing.
    monkeypatch.setattr(tn.urllib.request, "urlopen", _raises(b"<html>cloudflare</html>"))
    reason = tn._post_json("https://discord.com/api/webhooks/1/t", {})
    assert "404" in reason and "cloudflare" in reason


def test_send_test_posts_the_sample_and_reports_success(monkeypatch):
    sent: dict = {}

    def _post(url, payload, timeout=tn.TIMEOUT_SEC):
        sent.update(url=url, payload=payload)
        return ""

    monkeypatch.setattr(tn, "_post_json", _post)
    ok, detail = tn.send_test("https://discord.com/api/webhooks/1/tok")
    assert ok is True and "check the channel" in detail
    content = sent["payload"]["content"]
    assert tn.TEST_TITLE in content
    # A test must not be mistakable for a real finished run.
    assert "finished" in content and "test" in content.lower()
    assert sent["payload"]["allowed_mentions"] == {"parse": []}


def test_send_test_never_posts_outside_discord(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(tn, "_post_json", lambda url, payload, **kw: calls.append(url) or "")

    ok, detail = tn.send_test("http://127.0.0.1:8000/collect")
    assert ok is False and "not a Discord webhook" in detail
    ok, detail = tn.send_test("")
    assert ok is False and tn.WEBHOOK_ENV in detail
    assert calls == []


def test_send_test_hands_back_the_failure_verbatim(monkeypatch):
    monkeypatch.setattr(
        tn, "_post_json",
        lambda url, payload, **kw: "failed: Discord rejected the webhook: HTTP 404 — Unknown Webhook (code 10015)",
    )
    ok, detail = tn.send_test("https://discord.com/api/webhooks/1/tok")
    assert ok is False
    # The "failed:" prefix is notify_run's status vocabulary, not the UI's.
    assert not detail.startswith("failed:")
    assert "Unknown Webhook" in detail


def test_post_json_reaches_a_real_server():
    """The only un-faked hop: a POST body a Discord webhook would accept."""
    seen: dict = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server naming
            length = int(self.headers.get("Content-Length") or 0)
            seen["body"] = self.rfile.read(length)
            seen["type"] = self.headers.get("Content-Type")
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/hook"
        assert tn._post_json(url, {"content": "oi"}) == ""
    finally:
        server.shutdown()
        server.server_close()
    assert json.loads(seen["body"]) == {"content": "oi"}
    assert seen["type"] == "application/json"


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def test_windows_toast_carries_the_escaped_text(monkeypatch):
    scripts: list[str] = []

    def _ps(script, timeout=tn.TIMEOUT_SEC):
        scripts.append(script)
        return True, ""

    monkeypatch.setattr(tn, "_powershell", _ps)
    assert tn.windows_notification("a & b <c>", "x'y") == "shown"
    assert len(scripts) == 1
    script = scripts[0]
    assert "ToastNotificationManager" in script
    assert "a &amp; b &lt;c&gt;" in script
    # The body is a PowerShell single-quoted literal: an apostrophe must double.
    assert "x''y" in script


def test_windows_falls_back_to_a_tray_balloon(monkeypatch):
    scripts: list[str] = []

    def _ps(script, timeout=tn.TIMEOUT_SEC):
        scripts.append(script)
        return (len(scripts) > 1), ("" if len(scripts) > 1 else "no toast for you")

    monkeypatch.setattr(tn, "_powershell", _ps)
    assert tn.windows_notification("t", "b") == "shown"
    assert len(scripts) == 2
    assert "NotifyIcon" in scripts[1]


def test_windows_reports_when_nothing_could_be_shown(monkeypatch):
    monkeypatch.setattr(tn, "_powershell", lambda script, timeout=tn.TIMEOUT_SEC: (False, "powershell not found"))
    assert tn.windows_notification("t", "b") == "failed: powershell not found"


def test_windows_notifications_are_skipped_off_windows(monkeypatch):
    monkeypatch.setattr(tn.os, "name", "posix")
    assert tn.windows_notification("t", "b").startswith("skipped:")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def test_notify_run_announces_a_finished_run_on_the_channels_that_are_on(tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(tn, "windows_notification", lambda t, b: seen.append("win") or "shown")
    monkeypatch.setattr(tn, "discord_message", lambda t, b, url=None: seen.append("disc") or "sent")

    result = tn.notify_run(tmp_path, finished(), {"windows": True, "discord": True})
    assert result == {"windows": "shown", "discord": "sent"}
    assert seen == ["win", "disc"]
    log = (tmp_path / tn.NOTIFY_LOG).read_text(encoding="utf-8")
    assert "Piper training finished" in log and "discord: sent" in log


def test_notify_run_only_uses_the_channels_the_user_turned_on(tmp_path, monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(tn, "windows_notification", lambda t, b: called.append("win") or "shown")
    monkeypatch.setattr(tn, "discord_message", lambda t, b, url=None: called.append("disc") or "sent")

    assert tn.notify_run(tmp_path, finished(), {"windows": False, "discord": True}) == {"discord": "sent"}
    assert called == ["disc"]
    assert tn.notify_run(tmp_path, finished(), {}) == {}


def test_notify_run_stays_quiet_when_the_user_stopped_the_run(tmp_path):
    assert tn.notify_run(tmp_path, finished(status="cancelled"), {"windows": True, "discord": True}) == {}
    assert not (tmp_path / tn.NOTIFY_LOG).exists()


def test_a_broken_notifier_never_escapes(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("toast api exploded")

    monkeypatch.setattr(tn, "windows_notification", _boom)
    monkeypatch.setattr(tn, "discord_message", _boom)
    result = tn.notify_run(tmp_path, finished(), {"windows": True, "discord": True})
    assert result["windows"].startswith("failed: toast api exploded")
    assert result["discord"].startswith("failed: toast api exploded")


def test_webhook_is_read_from_the_env(monkeypatch):
    monkeypatch.setenv(tn.WEBHOOK_ENV, "https://discord.com/api/webhooks/9/xyz")
    assert tn.webhook_url() == "https://discord.com/api/webhooks/9/xyz"


def test_webhook_falls_back_to_the_dotenv_without_one_in_the_environment(monkeypatch):
    """The supervisor is standalone: .env must be readable from here too."""
    import dotenv

    monkeypatch.delenv(tn.WEBHOOK_ENV, raising=False)
    opened: list[str] = []

    def _load(path, override=False):
        opened.append(path)
        monkeypatch.setenv(tn.WEBHOOK_ENV, "https://discord.com/api/webhooks/7/env")
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", _load)
    assert tn.webhook_url() == "https://discord.com/api/webhooks/7/env"
    assert opened and opened[0].endswith(".env")
