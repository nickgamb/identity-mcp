"""
Detect degenerate reasoning loops in Letta/Ollama streams and signal when to
stop surfacing further thinking — without aborting the upstream generation.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# --- Pattern repetition (exact tail repeat) ---
REPETITION_BUF_SIZE = 600
REPETITION_MIN_CHARS = 200
REPETITION_MIN_PATTERN = 5
REPETITION_MAX_PATTERN = 80
REPETITION_MIN_REPEATS = 6

# --- Meta-token density (e.g. chatlog "(Writing)." / "(End)." loops) ---
MARKER_WINDOW = 500
MARKER_MIN_CHARS = 150
MARKER_MIN_COUNT = 14
_META_LINE_RE = re.compile(
    r"^\s*\((?:Writing|End|Generating|Wait|Going|Start|Proceeding|"
    r"Self-correction|Final check|Okay|Perfect|Done|End of)\)[.\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

# --- Hard cap: close thinking if monologue runs too long without assistant ---
MAX_REASONING_CHARS = int(os.getenv("LETTA_MAX_REASONING_CHARS", "24000"))


class LoopReason(str, Enum):
    PATTERN = "pattern"
    MARKERS = "markers"
    MAX_LENGTH = "max_length"


@dataclass
class ReasoningGuardState:
    suppressed: bool = False
    close_reason: Optional[LoopReason] = None
    total_chars: int = 0


class ReasoningLoopGuard:
    """
    Accumulates reasoning text. When a loop is detected, sets suppressed=True.
    Callers should close the thinking UI block and stop forwarding reasoning,
    but keep reading the Letta stream for tool calls and assistant_message.
    """

    __slots__ = ("_buf", "state")

    def __init__(self) -> None:
        self._buf = ""
        self.state = ReasoningGuardState()

    @property
    def suppressed(self) -> bool:
        return self.state.suppressed

    def note_suppressed(self, reason: LoopReason) -> None:
        if not self.state.suppressed:
            self.state.suppressed = True
            self.state.close_reason = reason

    def feed(self, text: str) -> Optional[LoopReason]:
        """
        Process a reasoning chunk. Returns a LoopReason when thinking should
        be closed; returns None while thinking may continue.
        """
        if self.state.suppressed or not text:
            return None

        self.state.total_chars += len(text)
        self._buf += text
        if len(self._buf) > REPETITION_BUF_SIZE:
            self._buf = self._buf[-REPETITION_BUF_SIZE:]

        if self.state.total_chars >= MAX_REASONING_CHARS:
            self.note_suppressed(LoopReason.MAX_LENGTH)
            return LoopReason.MAX_LENGTH

        if len(self._buf) >= REPETITION_MIN_CHARS and self._check_pattern():
            self.note_suppressed(LoopReason.PATTERN)
            return LoopReason.PATTERN

        if len(self._buf) >= MARKER_MIN_CHARS and self._check_markers():
            self.note_suppressed(LoopReason.MARKERS)
            return LoopReason.MARKERS

        return None

    def _check_pattern(self) -> bool:
        tail = self._buf[-REPETITION_BUF_SIZE:]
        for plen in range(REPETITION_MIN_PATTERN, REPETITION_MAX_PATTERN + 1):
            if plen > len(tail) // REPETITION_MIN_REPEATS:
                break
            pattern = tail[-plen:]
            check_len = plen * REPETITION_MIN_REPEATS
            segment = tail[-check_len:]
            if segment == pattern * REPETITION_MIN_REPEATS:
                return True
        return False

    def _check_markers(self) -> bool:
        tail = self._buf[-MARKER_WINDOW:]
        return len(_META_LINE_RE.findall(tail)) >= MARKER_MIN_COUNT


def trim_reasoning_for_display(reasoning: str) -> tuple[str, bool]:
    """
    Non-streaming: walk reasoning and truncate if a loop is detected.
    Returns (safe_reasoning, was_trimmed).
    """
    if not reasoning:
        return "", False
    guard = ReasoningLoopGuard()
    out: list[str] = []
    trimmed = False
    for line in reasoning.split("\n"):
        chunk = line + "\n"
        reason = guard.feed(chunk)
        if reason is not None:
            trimmed = True
            break
        out.append(chunk)
    return "".join(out).rstrip(), trimmed


THINKING_CLOSE_NOTE = {
    LoopReason.PATTERN: "\n\n_(Planning loop detected — continuing with your reply.)_\n",
    LoopReason.MARKERS: "\n\n_(Repetitive planning tokens — continuing with your reply.)_\n",
    LoopReason.MAX_LENGTH: "\n\n_(Long internal planning — continuing with your reply.)_\n",
}
