"""Announce a finished training run — Windows notification + optional Discord.

Training runs detached, so the dashboard may well be closed when it ends. The
supervisor therefore calls this module itself, right after the terminal status
is on disk:

* `windows` — a toast notification on this machine (tray balloon if the toast
  API is unavailable);
* `discord` — a message to the webhook in `TRAIN_DISCORD_WEBHOOK` (.env), for
  when the machine is out of sight.

Nothing here may ever break a run: every failure is caught and reported as a
string in the job's ``notified`` block, which the dashboard shows next to the
finished run.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WEBHOOK_ENV = "TRAIN_DISCORD_WEBHOOK"

# A webhook URL is a credential: only ever post it to Discord itself, so a
# hand-edited .env (or a pasted "http://127.0.0.1:…" ) cannot turn the trainer
# into an outbound request for any host.
ALLOWED_PREFIXES = (
    "https://discord.com/api/webhooks/",
    "https://discordapp.com/api/webhooks/",
    "https://ptb.discord.com/api/webhooks/",
    "https://canary.discord.com/api/webhooks/",
)

NOTIFY_LOG = "notify.log"
TIMEOUT_SEC = 12.0


# ---------------------------------------------------------------------------
# What the message says
# ---------------------------------------------------------------------------

def _elapsed(job: dict[str, Any]) -> str:
    start = float(job.get("started_at") or 0)
    end = float(job.get("finished_at") or 0) or time.time()
    if not start:
        return ""
    total = max(0, int(round(end - start)))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


def _model_name(job: dict[str, Any]) -> str:
    onnx = str((job.get("exported") or {}).get("onnx") or "")
    return Path(onnx).name if onnx else ""


def message(job: dict[str, Any]) -> tuple[str, str]:
    """Title + body for a terminal job. Empty title means "do not notify"."""
    name = str(job.get("voice_name") or "the trained voice")
    epoch = int(job.get("epoch") or 0)
    epochs = int(job.get("epochs") or 0)
    progress = f"{epoch}/{epochs} epochs" if epochs else f"{epoch} epochs"
    took = _elapsed(job)
    status = str(job.get("status") or "")

    if job.get("running"):
        return "", ""
    if status == "done":
        body = progress + (f" in {took}" if took else "")
        model = _model_name(job)
        if model:
            body += f" · {model}"
        return f"✓ Piper training finished — {name}", body
    if status == "error":
        error = str(job.get("error") or "the trainer stopped without a result")
        body = f"stopped at {progress}"
        if took:
            body += f" after {took}"
        body += f"\n{error[:200]}"
        body += "\nResume it from the checkpoint in the dashboard."
        return f"✗ Piper training failed — {name}", body
    if status == "cancelled":
        return f"■ Piper training cancelled — {name}", f"stopped at {progress}"
    return "", ""


def webhook_url() -> str:
    """The Discord webhook from .env (loaded here: the supervisor is standalone)."""
    url = (os.getenv(WEBHOOK_ENV) or "").strip()
    if not url:
        try:
            from dotenv import load_dotenv

            from config import BASE_DIR

            load_dotenv(str(BASE_DIR / ".env"), override=False)
        except Exception:  # noqa: BLE001 — a missing .env is not an error
            pass
        url = (os.getenv(WEBHOOK_ENV) or "").strip()
    return url


def validate_webhook(url: str) -> str:
    """Reject anything that is not a Discord webhook URL; returns the reason."""
    if not url:
        return "no webhook configured — set TRAIN_DISCORD_WEBHOOK in API Keys"
    if url != url.strip() or any(c in url for c in "\r\n\t \"'"):
        return "the webhook URL has whitespace or quotes in it"
    if not url.startswith(ALLOWED_PREFIXES):
        return "that is not a Discord webhook URL (https://discord.com/api/webhooks/…)"
    return ""


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

_TOAST_SCRIPT = """\
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml({xml})
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Wallie-Kansai')
$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""

_BALLOON_SCRIPT = """\
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$icon = New-Object System.Windows.Forms.NotifyIcon
$icon.Icon = [System.Drawing.SystemIcons]::Information
$icon.Visible = $true
$icon.ShowBalloonTip(8000, {title}, {body}, [System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Milliseconds 6000
$icon.Dispose()
"""


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _powershell(script: str, timeout: float = TIMEOUT_SEC) -> tuple[bool, str]:
    """Run a PowerShell script inline, hidden, without touching the console."""
    exe = "powershell"
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
             "-EncodedCommand", encoded],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        return False, "powershell not found"
    except subprocess.TimeoutExpired:
        return False, f"powershell timed out after {int(timeout)}s"
    except OSError as e:  # pragma: no cover — unusual, but never fatal
        return False, f"powershell failed: {e}"
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return False, (detail.splitlines() or ["powershell returned %d" % proc.returncode])[-1][:200]
    return True, ""


def windows_notification(title: str, body: str) -> str:
    """Show a toast; fall back to a tray balloon. Returns a status string."""
    if os.name != "nt":
        return "skipped: Windows notifications need Windows"
    toast_xml = (
        '<toast><visual><binding template="ToastGeneric">'
        f"<text>{_xml_escape(title)}</text><text>{_xml_escape(body)}</text>"
        "</binding></visual></toast>"
    )
    ok, reason = _powershell(_TOAST_SCRIPT.format(xml=_ps_quote(toast_xml)))
    if ok:
        return "shown"
    ok, balloon_reason = _powershell(
        _BALLOON_SCRIPT.format(title=_ps_quote(title), body=_ps_quote(body))
    )
    if ok:
        return "shown"
    return f"failed: {reason or balloon_reason}"


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def _discord_rejection(e: urllib.error.HTTPError) -> str:
    """Discord's own words for a rejected webhook: 404 is "Unknown Webhook".

    Guessing here is what makes a bad URL a mystery ("Discord said 404"); the
    response body names the problem, so surface it instead of the status alone.
    """
    raw = ""
    try:
        raw = e.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — the body is a bonus, never the message
        pass
    detail = ""
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            detail = str(payload.get("message") or "").strip()
            code = payload.get("code")
            if detail and code not in (None, "", 0):
                detail += f" (code {code})"
        else:  # a JSON list/string: still better than nothing
            detail = " ".join(raw.split())
    except Exception:  # noqa: BLE001 — plain text or an HTML error page
        detail = " ".join(raw.split())
    detail = detail[:200]
    return f"Discord rejected the webhook: HTTP {e.code}" + (f" — {detail}" if detail else "")


def _post_json(url: str, payload: dict[str, Any], timeout: float = TIMEOUT_SEC) -> str:
    """POST a JSON body; returns "" on success, else a short reason."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "Wallie-Kansai"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            response.read(1024)
    except urllib.error.HTTPError as e:
        # Discord answers 404 for a deleted webhook and 400 for a malformed body.
        return "failed: " + _discord_rejection(e)
    except urllib.error.URLError as e:
        return f"failed: {str(e.reason)[:160]}"
    except Exception as e:  # noqa: BLE001 — timeouts, resets, TLS: all non-fatal
        return f"failed: {str(e)[:160]}"
    return ""


def discord_message(title: str, body: str, url: str | None = None) -> str:
    """Send the run's result to Discord. Returns a status string."""
    url = webhook_url() if url is None else url
    reason = validate_webhook(url)
    if reason:
        return "skipped: " + reason
    content = f"**{title}**" if title else ""
    if body:
        content += "\n" + body
    return _post_json(url, {"content": content[:1900], "allowed_mentions": {"parse": []}}) or "sent"


# Sample posted by the "send test" button in API Keys. A webhook URL is a
# credential that fails quietly otherwise: a deleted or mistyped one is only
# discovered after a training run ends, hours later.
TEST_TITLE = "🔔 Wallie training notifications are connected"
TEST_BODY = ("This is a test message — real ones arrive when a run ends, like this:\n"
             "✓ Piper training finished — pt_BR-luna-medium · 300/300 epochs in 12m 34s")


def send_test(url: str | None = None) -> tuple[bool, str]:
    """Post the sample message now; returns (ok, what to show the user)."""
    url = webhook_url() if url is None else url
    reason = validate_webhook(url)
    if reason:
        return False, reason
    error = _post_json(
        url,
        {"content": f"**{TEST_TITLE}**\n{TEST_BODY}", "allowed_mentions": {"parse": []}},
    )
    if error:
        return False, error.removeprefix("failed: ")
    return True, "test message delivered — check the channel"


# ---------------------------------------------------------------------------
# Entry point used by the supervisor
# ---------------------------------------------------------------------------

def notify_run(
    run_dir: Path | str,
    job: dict[str, Any],
    prefs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Tell the user how the run ended. Never raises, never changes the job."""
    prefs = prefs or {}
    # A stop is something the user just did; only a result worth interrupting
    # for — finished or failed — gets announced.
    if str(job.get("status") or "") in ("", "idle", "running", "cancelled"):
        return {}
    title, body = message(job)
    if not title:
        return {}
    results: dict[str, str] = {}
    if prefs.get("windows"):
        try:
            results["windows"] = windows_notification(title, body)
        except Exception as e:  # noqa: BLE001 — a notification must not kill the run
            results["windows"] = f"failed: {str(e)[:160]}"
    if prefs.get("discord"):
        try:
            results["discord"] = discord_message(title, body)
        except Exception as e:  # noqa: BLE001
            results["discord"] = f"failed: {str(e)[:160]}"
    if results:
        _record(Path(run_dir), title, body, results)
    return results


def _record(run_dir: Path, title: str, body: str, results: dict[str, str]) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {title} — " + ", ".join(
        f"{k}: {v}" for k, v in results.items()
    )
    try:
        with open(run_dir / NOTIFY_LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\n" + body.replace("\n", " | ") + "\n")
    except OSError:
        pass
    # The supervisor's own stdout is supervisor.log — keep a trace there too,
    # since that is the file to look at when a run dies silently.
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001 — a closed stdout must not matter
        pass


if __name__ == "__main__":  # manual smoke test: python -m tts.train_notify
    demo = {
        "status": "done", "voice_name": "pt_BR-luna-medium", "epoch": 300,
        "epochs": 300, "started_at": time.time() - 754, "finished_at": time.time(),
        "exported": {"onnx": "C:/tmp/pt_BR-luna-medium.onnx"},
    }
    print(notify_run(Path("."), demo, {"windows": True, "discord": False}))
    sys.exit(0)
