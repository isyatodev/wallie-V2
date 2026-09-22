"""Engagement gate — should Wallie spend LLM credits on this input?

Cost control: every user-initiated turn (chat message, hearing turn, chat
mention fused into a vision turn) costs an LLM call. When nobody is talking
TO Wallie, most of those turns are the model answering past an empty room.
With ``persona.require_engagement`` on, inputs that don't show the user is
engaging Wallie are dropped before they ever reach the LLM:

- addressed by name/handle/alias
- replying to what Wallie just said (lexical overlap, answer markers)
- continuing a live thread (the user Wallie last addressed, or the user
  Wallie last replied to, within the conversation window)
- directly on Wallie's recent topic (mirrors what she was just saying)
- a question clearly ABOUT Wallie ("what do you think?", "você joga bem?")

Matching is deliberately cheap (string work, no LLM): the gate's whole point
is saving calls, so it must never spend one. Bias: false negatives (dropping
a message nobody expected a reply to) are cheap; false positives cost money.
"""
from __future__ import annotations

import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

_WORD = re.compile(r"[a-z0-9]+")
_EMOTE_ONLY = re.compile(r"^(?:[\U0001F000-\U0001FAFF\u2600-\u27BF]|\s|:\w+:)+$")
# Pure-reaction chatter ("haha", "kkkk", "gg") — nobody expects a reply.
_CREDITS = {
    "haha", "hahaha", "kkkk", "kkkkk", "lol", "lmao", "omg", "wow", "bruh",
    "nice", "gg", "pog", "poggers", "clap", "cringe", "based", "true",
    "real", "oop", "oof", "yay", "wooo", "woooow", "ez", "kappa",
}
# Direct-reply markers: only trusted DURING a live conversation window.
_REPLY_MARKERS = re.compile(
    r"^(?:sim|si|yes|yeah|yep|no|nope|nah|nao|verdade|real|concordo|discordo|ok|okay)\b"
)
# Short canned acknowledgements: when the gate skips an input, one of these MAY
# be spoken instead of full silence — proof of life without spending an LLM call.
# Deliberately generic (no answer, no opinion) and pronoun-agnostic.
_ACKS = (
    "hmm, boa.",
    "interessante...",
    "boa, anotado.",
    "deixa eu pensar nisso.",
    "hmm.",
    "entendi.",
    "good point.",
    "hm, interesting.",
    "noted.",
    "let me think about that.",
    "hmm.",
    "mm-hmm.",
)
# Never fire the same ack twice in a row.
_recent_acks: list[str] = []
# Too generic to prove topical overlap on their own (EN + PT).
_OVERLAP_STOP = {
    "this", "that", "with", "have", "from", "will", "your", "just", "like",
    "there", "then", "what", "when", "they", "them", "about", "really",
    "isso", "essa", "esse", "mais", "para", "como", "quando", "muito",
    "sobre", "antes", "depois", "aqui", "assim", "entao", "tambem",
}


def _norm_words(text: str) -> list[str]:
    """Lowercased word tokens, accents folded ('Wallié' → 'wallie')."""
    s = unicodedata.normalize("NFD", (text or "").casefold())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return [t.rstrip("_") for t in _WORD.findall(s)]


def pick_ack(rng: Optional[random.Random] = None) -> str:
    """Pick a short acknowledgement line, avoiding immediate repeats."""
    r = rng or random
    global _recent_acks
    pool = [a for a in _ACKS if a not in _recent_acks] or list(_ACKS)
    choice = r.choice(pool)
    _recent_acks = ([choice] + _recent_acks)[:2]   # remember the last two
    return choice


@dataclass
class EngagementState:
    """Live conversation context. Nothing persists — a fresh session starts
    cold, and only real interactions build the thread."""
    last_reply_text: str = ""        # what Wallie last SAID (spoken or typed)
    last_reply_ts: float = 0.0
    last_input_text: str = ""        # the input Wallie last REPLIED to
    last_input_ts: float = 0.0
    last_input_user: str = ""
    last_reply_user: str = ""        # who Wallie last addressed
    last_reply_user_ts: float = 0.0
    last_topic_words: frozenset = field(default_factory=frozenset)
    last_topic_ts: float = 0.0


class EngagementGate:
    WINDOW_SEC = 90.0        # a conversation thread dies after this silence
    TOPIC_WINDOW_SEC = 120.0

    def __init__(self, *, name: str = "", aliases: Optional[list[str]] = None) -> None:
        names = {name, *(a for a in (aliases or []))}
        self._names = {n.strip().lower() for n in names if n and n.strip()}
        self.state = EngagementState()

    # ---- recording (cheap, always on — even when the gate drops) ----
    def note_reply(self, text: str) -> None:
        """Record what Wallie just said (a spoken sentence or typed line)."""
        self.state.last_reply_text = text or ""
        self.state.last_reply_ts = time.time()

    def note_reply_to(self, username: str) -> None:
        """Record who Wallie just addressed — they own the thread for a while."""
        if username and username.strip():
            self.state.last_reply_user = username
            self.state.last_reply_user_ts = time.time()

    def note_input(self, text: str, username: str = "") -> None:
        """Record the input Wallie actually replied to."""
        self.state.last_input_text = text or ""
        self.state.last_input_ts = time.time()
        if username:
            self.state.last_input_user = username

    def note_topic(self, text: str) -> None:
        """Snapshot Wallie's current topic words (from her own monologue/vision)."""
        words = frozenset(w for w in _norm_words(text) if len(w) >= 5)
        if words:
            self.state.last_topic_words = words
            self.state.last_topic_ts = time.time()

    # ---- deciding ----
    def should_reply(self, text: str, *, username: str = "", streamer: bool = False) -> tuple[bool, str]:
        """Does this input deserve an LLM turn? Returns (verdict, reason).

        ``streamer=True`` bypasses the gate entirely (the broadcaster is part
        of the show and may steer Wallie at will).
        """
        st = self.state
        if streamer:
            return True, "streamer"
        tokens = _norm_words(text)
        low_tokens = set(tokens)
        now = time.time()
        in_reply_window = bool(st.last_reply_ts) and now - st.last_reply_ts <= self.WINDOW_SEC

        # 1. Addressed by name / handle / alias.
        if self._names and (self._names & low_tokens):
            return True, "named"

        # 2. Thread continuity — the user Wallie addressed, or the user whose
        #    message Wallie last replied to, may keep talking to her.
        if username:
            u = username.lower()
            if (st.last_reply_user and u == st.last_reply_user.lower()
                    and st.last_reply_user_ts and now - st.last_reply_user_ts <= self.WINDOW_SEC):
                return True, "thread-you-addressed"
            if (st.last_input_user and u == st.last_input_user.lower()
                    and st.last_input_ts and now - st.last_input_ts <= self.WINDOW_SEC):
                return True, "thread-your-turn"

        # 3. Conversational back-and-forth: shares real words with what Wallie
        #    just said ("that boss is INSANE" right after she raged about it).
        if in_reply_window and self._overlaps(tokens, _norm_words(st.last_reply_text)):
            return True, "replying-to-her"

        # 4. On Wallie's recent topic — the thing she herself was talking about.
        if (st.last_topic_words and st.last_topic_ts
                and now - st.last_topic_ts <= self.TOPIC_WINDOW_SEC
                and self._overlaps(tokens, list(st.last_topic_words))):
            return True, "on-her-topic"

        # 5. Question clearly ABOUT Wallie ("what do you think?", "você é bot?").
        if self._question_about(tokens):
            return True, "question-about-her"

        # 6. Direct answer markers ("sim", "concordo") only inside a live window.
        if in_reply_window and tokens and _REPLY_MARKERS.match(" ".join(tokens[:2])):
            return True, "answer-marker"

        return False, "no-engagement"

    def _question_about(self, tokens: list[str]) -> bool:
        if len(tokens) < 3:
            return False
        about = self._names | {"you", "your", "yours", "u", "vc", "voce", "tu", "teu", "sua", "seu"}
        return any(w in about for w in tokens[:4])

    @staticmethod
    def _overlaps(a: list[str], b: list[str], *, min_len: int = 4, need: int = 1) -> bool:
        aw = {x for x in a if len(x) >= min_len and x not in _OVERLAP_STOP}
        bw = {y for y in b if len(y) >= min_len and y not in _OVERLAP_STOP}
        return len(aw & bw) >= need


def is_low_value_chat(text: str) -> bool:
    """Pure-reaction chatter (emotes only, 'kkkk', 'gg') — never worth a turn."""
    t = re.sub(r":\w+:", " ", (text or ""))  # strip :emote: tokens first
    t = t.strip()
    if not t or _EMOTE_ONLY.match(t):
        return True
    words = _norm_words(t)
    return bool(words) and len(words) <= 3 and all(w in _CREDITS for w in words)
