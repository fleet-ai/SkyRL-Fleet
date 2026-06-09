"""Rolling feedback buffer for Direction 3: in-context reward adaptation.

Stores textual critiques of reward model failures and injects them into
the judge prompt. No weight updates — purely in-context.

Usage:
    buf = FeedbackBuffer(max_size=10)
    buf.add("keywords:existence", "Model used synonym 'huge' instead of required keyword 'large'.")
    buf.add("length_constraints:number_words", "Model wrote 87 words but constraint required exactly 100.")

    # In judge prompt:
    prompt += buf.format_for_prompt()
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional


@dataclass
class FeedbackEntry:
    instruction_type: str   # e.g. "keywords:existence"
    critique: str           # free-form natural language
    source: str = "auto"    # "auto" | "human"
    step: int = 0           # training step when added (for eviction ordering)

    def to_dict(self) -> dict:
        return {
            "instruction_type": self.instruction_type,
            "critique": self.critique,
            "source": self.source,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEntry":
        return cls(**d)


class FeedbackBuffer:
    """FIFO rolling buffer of textual critiques, keyed by instruction type.

    Deduplication: if the buffer already has an entry for the same
    instruction_type and the critique text is nearly identical (>80% token
    overlap), the new entry replaces the old one rather than appending.
    """

    def __init__(self, max_size: int = 12):
        self.max_size = max_size
        self._entries: Deque[FeedbackEntry] = deque(maxlen=max_size)

    def add(
        self,
        instruction_type: str,
        critique: str,
        source: str = "auto",
        step: int = 0,
    ) -> None:
        # Replace existing entry for same instruction_type if similar enough.
        for i, e in enumerate(self._entries):
            if e.instruction_type == instruction_type and self._similar(e.critique, critique):
                # Replace in-place by rebuilding (deque doesn't support index assignment).
                entries = list(self._entries)
                entries[i] = FeedbackEntry(instruction_type, critique, source, step)
                self._entries = deque(entries, maxlen=self.max_size)
                return
        self._entries.append(FeedbackEntry(instruction_type, critique, source, step))

    def add_entry(self, entry: FeedbackEntry) -> None:
        self.add(entry.instruction_type, entry.critique, entry.source, entry.step)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[FeedbackEntry]:
        return list(self._entries)

    def format_for_prompt(self, header: bool = True) -> str:
        """Return a string suitable for appending to a judge system prompt."""
        if not self._entries:
            return ""
        lines = []
        if header:
            lines.append(
                "## Recent feedback on past scoring errors\n"
                "The following critiques were collected from cases where the reward "
                "signal was wrong or ambiguous. Apply them when scoring similar constraints.\n"
            )
        for e in self._entries:
            tag = f"[{e.instruction_type}]" if e.instruction_type else "[general]"
            lines.append(f"- {tag} {e.critique}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    @classmethod
    def load(cls, path: str | Path, max_size: int = 12) -> "FeedbackBuffer":
        buf = cls(max_size=max_size)
        with open(path) as f:
            entries = json.load(f)
        for d in entries:
            buf.add_entry(FeedbackEntry.from_dict(d))
        return buf

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
        """Very lightweight token-overlap similarity."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return False
        overlap = len(ta & tb) / max(len(ta), len(tb))
        return overlap >= threshold


# ---------------------------------------------------------------------------
# Auto-critique generation from programmatic failure attribution
# ---------------------------------------------------------------------------

# Human-readable description of each instruction type for critique generation.
_INSTRUCTION_DESCRIPTIONS = {
    "keywords:existence": "include specific required keywords",
    "keywords:frequency": "use a keyword a specific number of times",
    "keywords:forbidden_words": "avoid using certain forbidden words",
    "keywords:letter_frequency": "use a letter a specific number of times",
    "language:response_language": "respond in a specific language",
    "length_constraints:number_sentences": "write a specific number of sentences",
    "length_constraints:number_paragraphs": "write a specific number of paragraphs",
    "length_constraints:number_words": "write a specific number of words",
    "length_constraints:nth_paragraph_first_word": "start a specific paragraph with a given word",
    "detectable_content:number_placeholders": "include a specific number of placeholders like [NAME]",
    "detectable_content:postscript": "include a postscript (P.S.) section",
    "detectable_format:number_bullet_lists": "include a specific number of bullet points",
    "detectable_format:constrained_response": "respond with only one of a set of allowed values",
    "detectable_format:number_highlighted_sections": "include highlighted text using *asterisks*",
    "detectable_format:multiple_sections": "organize the response into labeled sections",
    "detectable_format:json_format": "format the entire response as valid JSON",
    "detectable_format:title": "include a title wrapped in <<double angle brackets>>",
    "combination:two_responses": "provide two distinct responses separated by ******",
    "combination:repeat_prompt": "begin by repeating the original prompt",
    "startend:end_checker": "end the response with a specific phrase",
    "change_case:capital_word_frequency": "capitalize a specific number of words",
    "change_case:english_capital": "write the entire response in uppercase",
    "change_case:english_lowercase": "write the entire response in lowercase",
    "punctuation:no_comma": "avoid using any commas",
}


def auto_critique(
    instruction_id: str,
    passed: bool,
    response: str,
    kwargs: dict,
    step: int = 0,
) -> Optional[FeedbackEntry]:
    """Generate a short auto-critique for a failed (or unexpectedly passed) instruction.

    Returns None if no useful critique can be generated.
    """
    desc = _INSTRUCTION_DESCRIPTIONS.get(instruction_id, instruction_id)

    if passed:
        # No critique needed for passes.
        return None

    # Build a specific, concise critique based on instruction type.
    critique = _build_specific_critique(instruction_id, response, kwargs)
    if critique is None:
        critique = (
            f"The response failed to {desc}. "
            "Check this constraint carefully — it is easy to miss."
        )

    return FeedbackEntry(
        instruction_type=instruction_id,
        critique=critique,
        source="auto",
        step=step,
    )


def _build_specific_critique(instruction_id: str, response: str, kwargs: dict) -> Optional[str]:
    """Return an instruction-specific critique string, or None to use the generic fallback."""
    import re

    if instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", []) or []
        missing = [kw for kw in keywords if str(kw).lower() not in response.lower()]
        if missing:
            return (
                f"The response was missing required keyword(s): {missing}. "
                "The model must include the exact word (case-insensitive). Synonyms do not count."
            )

    elif instruction_id == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", []) or []
        found = [w for w in forbidden if re.search(r"\b" + re.escape(str(w)) + r"\b", response, re.IGNORECASE)]
        if found:
            return (
                f"The response used forbidden word(s): {found}. "
                "The model must avoid these words entirely, including in parenthetical or quoted text."
            )

    elif instruction_id == "length_constraints:number_words":
        n = kwargs.get("num_words", "?")
        relation = kwargs.get("relation", "at least")
        words = len(re.findall(r"\b\w+\b", response))
        return (
            f"The response had {words} words but the constraint required {relation} {n}. "
            "The model should count words carefully, including words in lists and headings."
        )

    elif instruction_id == "length_constraints:number_sentences":
        n = kwargs.get("num_sentences", "?")
        relation = kwargs.get("relation", "at least")
        sents = len([p for p in re.split(r"[.!?]+", response) if p.strip()])
        return (
            f"The response had ~{sents} sentences but the constraint required {relation} {n}. "
            "Note: bullet points and list items each count as a sentence."
        )

    elif instruction_id == "detectable_format:json_format":
        return (
            "The response was not valid JSON. The entire response must be a single JSON object or array, "
            "with no surrounding prose. Markdown code fences (```json ... ```) are acceptable."
        )

    elif instruction_id == "change_case:english_lowercase":
        upper_words = re.findall(r"\b[A-Z][A-Z]+\b", response)
        if upper_words:
            return (
                f"The response contained uppercase letters ({upper_words[:3]}...). "
                "The entire response must be in lowercase, including proper nouns and the first word of sentences."
            )

    elif instruction_id == "change_case:english_capital":
        lower_words = [w for w in re.findall(r"\b[a-z]+\b", response) if len(w) > 1]
        if lower_words:
            return (
                f"The response contained lowercase letters. "
                "The entire response must be in UPPERCASE."
            )

    elif instruction_id == "detectable_content:postscript":
        marker = kwargs.get("postscript_marker", "P.S.")
        return (
            f"The response did not contain the required postscript marker '{marker}'. "
            "Add a postscript section at the end of the response using that exact marker."
        )

    elif instruction_id == "startend:end_checker":
        phrase = kwargs.get("end_phrase", "")
        return (
            f"The response did not end with the required phrase: '{phrase}'. "
            "The very last characters of the response (ignoring trailing whitespace) must match exactly."
        )

    elif instruction_id == "punctuation:no_comma":
        comma_count = response.count(",")
        return (
            f"The response contained {comma_count} comma(s). "
            "The constraint requires zero commas — rewrite sentences to avoid them entirely."
        )

    elif instruction_id == "detectable_format:number_bullet_lists":
        n = kwargs.get("num_bullets", "?")
        found = len([l for l in response.splitlines() if re.match(r"^\s*[\*\-]\s+\S", l)])
        return (
            f"The response had {found} bullet point(s) but the constraint required exactly {n}. "
            "Count each '- ' or '* ' line as one bullet."
        )

    return None
