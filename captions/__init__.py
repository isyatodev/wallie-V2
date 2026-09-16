"""Captions — TTS speech rendered as a browser-source caption overlay.

Wallie's spoken text is pushed here as sentences are TTS'd; connected caption
clients (browser sources in OBS) receive them over SSE, and the box is CLEARED
when the speech ends so stale text never lingers on stream.
"""
from .hub import CaptionHub

__all__ = ["CaptionHub"]
