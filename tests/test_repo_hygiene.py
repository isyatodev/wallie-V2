"""Repo hygiene — the fork must not ship anyone's personal profiles.

Both the README and the guide promise that only ``profiles/default.yaml`` is
versioned and that every other profile (with its voice library and memory
files) is ignored. A promise in prose is not a guarantee: one ``git add -f`` or
one accidentally committed profile is enough to publish a persona, the voice
ids it cloned and what the streamer remembers about someone's chat.

These checks run against the checkout itself, so they are skipped when the
tests are unpacked without git (a ZIP download, a wheel, an export).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")

# The profiles that belong to whoever runs this checkout. They are the ones the
# .gitignore block names; a machine with its own personas follows the same rule.
PERSONAL_PROFILES = ("luna", "conspiracy", "gremlin", "memes", "minecraft", "p-voices")

# Everything a profile accumulates next to its YAML — voices carry cloned voice
# ids, the two JSON files carry what the streamer learned.
PROFILE_FILES = (".yaml", ".voices.json", ".longterm.json", ".memory.json")

# Local secrets and runtime state: a tracked copy leaks API keys or the
# active-profile marker into a fork, a PR or a ZIP someone shares.
SECRET_FILES = (".env", ".wallie_state.json")

# The engines' voices tree: downloaded models, ported packages and the user's
# own trained voices — big binaries plus cloned-voice data.
VOICES_PATHS = (
    "voices/en_US-amy-medium.onnx",
    "voices/en_US-amy-medium.onnx.json",
    "voices/kokoro/pf_dora.pt",
    "voices/ported/luna-piper/manifest.json",
    "voices/local-src/luna/take1.wav",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    assert GIT is not None
    return subprocess.run(
        [GIT, *args], cwd=REPO, capture_output=True, text=True, timeout=30,
    )


@pytest.fixture(scope="module", autouse=True)
def git_checkout() -> None:
    probe = _git("rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("not a git checkout — profile versioning cannot be checked")


def test_only_the_default_profile_is_tracked():
    out = _git("ls-files", "--", "profiles")
    assert out.returncode == 0, out.stderr
    tracked = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    assert tracked == ["profiles/default.yaml"], (
        "personal profiles must stay out of git — only the template ships; "
        f"tracked: {tracked}"
    )


@pytest.mark.parametrize("name", PERSONAL_PROFILES)
def test_every_file_of_a_personal_profile_is_ignored(name: str):
    for suffix in PROFILE_FILES:
        path = f"profiles/{name}{suffix}"
        out = _git("check-ignore", "-q", "--", path)
        assert out.returncode == 0, (
            f"{path} is not ignored — a plain `git add` would commit it "
            "(add the profile to the .gitignore block)"
        )


def test_the_default_template_is_not_ignored():
    out = _git("check-ignore", "-q", "--", "profiles/default.yaml")
    assert out.returncode == 1, "profiles/default.yaml must stay trackable"


# ---------------------------------------------------------------------------
# Secrets and runtime state
# ---------------------------------------------------------------------------

def test_local_secrets_and_runtime_state_are_never_tracked():
    out = _git("ls-files", "--", *SECRET_FILES)
    assert out.returncode == 0, out.stderr
    tracked = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    assert tracked == [], (
        f"{tracked} must never be versioned — they hold API keys and local state; "
        "keep them in .gitignore"
    )


@pytest.mark.parametrize("path", SECRET_FILES)
def test_local_secrets_and_runtime_state_are_ignored(path: str):
    out = _git("check-ignore", "-q", "--", path)
    assert out.returncode == 0, (
        f"{path} is not ignored — a plain `git add -A` would commit it"
    )


def test_the_env_template_is_still_tracked():
    """Guarding .env must not take the shareable example with it."""
    out = _git("ls-files", "--", ".env.example")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ".env.example"


# ---------------------------------------------------------------------------
# The voices tree
# ---------------------------------------------------------------------------

def test_no_file_of_the_voices_tree_is_tracked():
    out = _git("ls-files", "--", "voices")
    assert out.returncode == 0, out.stderr
    tracked = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    assert tracked == [], (
        f"{tracked} must never be versioned — models and ported packages are "
        "downloaded/generated locally, not shipped"
    )


@pytest.mark.parametrize("path", VOICES_PATHS)
def test_the_voices_tree_is_ignored(path: str):
    out = _git("check-ignore", "-q", "--", path)
    assert out.returncode == 0, (
        f"{path} is not ignored — downloaded models and trained voices would "
        "end up in a commit"
    )


# ---------------------------------------------------------------------------
# Credentials that leaked into a tracked file
# ---------------------------------------------------------------------------

# Shapes of real credentials. Masked previews ("sk-•••xyz") and short
# placeholders do not match; the fixtures that do are allow-listed below, with
# the reason they are there.
KEY_PATTERNS = {
    "OpenAI-compatible key": r"sk-[A-Za-z0-9_-]{24,}",
    "Groq key": r"gsk_[A-Za-z0-9]{30,}",
    "Google API key": r"AIza[0-9A-Za-z_-]{35}",
    "Slack token": r"xox[baprs]-[0-9A-Za-z-]{10,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{36,}",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "JWT": r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}",
    "private key block": r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
}

# Deliberate dummy keys, used by the tests that prove keys are scrubbed from
# provider errors and logs. They are not credentials — keep this list tiny and
# justified; anything else that looks like a key belongs in .env, never in git.
ALLOWED_DUMMY_KEYS = {
    ("tests/test_vision_test.py", "sk-abcdefghijklmnopqrstuvwxyz"),
}


def _final_project_path(raw: str) -> str:
    """git grep prints repo-relative paths; normalise the separator."""
    return raw.strip().replace("\\", "/")


@pytest.mark.parametrize("label,pattern", sorted(KEY_PATTERNS.items()))
def test_no_api_key_is_committed(label: str, pattern: str):
    """A key in a tracked file is a leaked key — history keeps it forever.

    Because git history is append-only in practice, the fix is never only to
    edit the file: rotate the credential at the provider and then remove it
    from the working copy (and from history if the repo is public).
    """
    # -e keeps a pattern that starts with dashes (the PEM header) from being
    # read as an option; the patterns are POSIX ERE, which git grep speaks.
    out = _git("grep", "-I", "-n", "-E", "-e", pattern)
    assert out.returncode in (0, 1), out.stderr
    if out.returncode == 1:
        return  # no match anywhere — nothing leaked for this shape

    hits: list[str] = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        path, _, rest = line.partition(":")
        lineno, _, content = rest.partition(":")
        path = _final_project_path(path)
        if any(path == allowed_path and dummy in content
               for allowed_path, dummy in ALLOWED_DUMMY_KEYS):
            continue
        # Never echo the match itself into CI logs — file and line are enough.
        hits.append(f"{path}:{lineno}")
    assert hits == [], (
        f"a real {label} is committed in {hits}. The value is in git history: "
        "rotate it at the provider, replace it with a placeholder, and purge it "
        "from history before publishing"
    )
