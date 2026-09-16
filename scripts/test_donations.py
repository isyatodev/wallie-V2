"""Mock mode — inject fake donations into a RUNNING Wallie. No real payment.

Requires the dashboard to be up (python -m wallie --dashboard) with the
orchestrator started. Talks to the same /api/test/donation endpoint the
dashboard's Donations → Test strip uses, so the event flows through the REAL
pipeline (queue → orchestrator → LLM → TTS → avatar).

Usage:
  python scripts/test_donations.py livepix     --donor Maria --amount 10 --message "manda salve"
  python scripts/test_donations.py streamlabs  --donor Joao  --amount 5  --message "gg"
  python scripts/test_donations.py both
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8765"  # DASHBOARD_HOST/DASHBOARD_PORT defaults


def _post_donation(source: str, donor: str, amount: float, message: str) -> None:
    payload = {
        "source": source,
        "donor": donor,
        "amount": amount,
        "currency": "BRL",
        "message": message,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/test/donation",
        data=__import__("json").dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = __import__("json").loads(resp.read().decode("utf-8"))
            print(f"✓ [{source}] queued: {donor} — {amount:.2f} (event {body.get('event_id', '?')})")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        print(f"✗ [{source}] HTTP {e.code}: {detail}")
        if e.code == 400 and "not running" in detail.lower():
            print("  → start Wallie first (dashboard → Start, or `python -m wallie`)")
        sys.exit(1)
    except Exception as e:
        print(f"✗ [{source}] {e}")
        print(f"  → is the dashboard running at {BASE_URL}? (python -m wallie --dashboard)")
        sys.exit(1)


def main() -> None:
    global BASE_URL
    ap = argparse.ArgumentParser(description="Inject fake donations (mock mode)")
    ap.add_argument("source", choices=["livepix", "streamlabs", "both"])
    ap.add_argument("--donor", default="TestDonor")
    ap.add_argument("--amount", type=float, default=10.0)
    ap.add_argument("--message", default="test donation — ignore")
    ap.add_argument("--base-url", default=BASE_URL, help="dashboard URL")
    args = ap.parse_args()

    BASE_URL = args.base_url.rstrip("/")

    sources = ["livepix", "streamlabs"] if args.source == "both" else [args.source]
    for src in sources:
        _post_donation(src, args.donor, args.amount, args.message)


if __name__ == "__main__":
    main()
