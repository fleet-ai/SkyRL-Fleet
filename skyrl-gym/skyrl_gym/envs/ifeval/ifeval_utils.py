"""Self-contained IFEval (Instruction-Following Evaluation) checker.

Implements the verifiable instruction checks described in the IFEval paper
(Zhou et al., 2023, "Instruction-Following Evaluation for Large Language Models").

The public entry point is :func:`compute_score`, which returns the *fraction* of
verifiable constraints that a response satisfies (a soft reward in ``[0.0, 1.0]``).
This gives a richer RL signal than a strict all-or-nothing binary reward.

Only the standard library + ``re`` are required. ``langdetect`` is used opportunistically
for the ``language:response_language`` instruction; if it is not installed, that single
instruction is treated as satisfied (pass) so it does not unfairly penalize the response.
"""

import json
import re

try:  # optional dependency, only used for language detection
    from langdetect import detect as _detect_language  # type: ignore

    _LANGDETECT_AVAILABLE = True
except Exception:  # pragma: no cover - langdetect not installed
    _LANGDETECT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _relation_satisfied(count: int, n: int, relation: str) -> bool:
    """Compare ``count`` against ``n`` according to an IFEval relation string."""
    relation = (relation or "").strip().lower()
    if relation == "at least":
        return count >= n
    if relation == "at most":
        return count <= n
    if relation == "exactly":
        return count == n
    # Default to "at least" semantics if the relation is missing/unknown.
    return count >= n


def _split_paragraphs(text: str):
    """Split into non-empty paragraphs delimited by blank lines (double newlines)."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    return len([p for p in parts if p.strip()])


# ---------------------------------------------------------------------------
# Individual instruction checkers. Each returns a bool.
# ---------------------------------------------------------------------------
def _check_keywords_existence(response: str, kwargs: dict) -> bool:
    keywords = kwargs.get("keywords", []) or []
    low = response.lower()
    return all(str(kw).lower() in low for kw in keywords)


def _check_keywords_frequency(response: str, kwargs: dict) -> bool:
    keyword = kwargs.get("keyword", "")
    n = int(kwargs.get("frequency", 0))
    relation = kwargs.get("relation", "at least")
    if not keyword:
        return True
    count = len(re.findall(re.escape(keyword), response, flags=re.IGNORECASE))
    return _relation_satisfied(count, n, relation)


def _check_keywords_forbidden_words(response: str, kwargs: dict) -> bool:
    forbidden = kwargs.get("forbidden_words", []) or []
    for word in forbidden:
        if re.search(r"\b" + re.escape(str(word)) + r"\b", response, flags=re.IGNORECASE):
            return False
    return True


def _check_keywords_letter_frequency(response: str, kwargs: dict) -> bool:
    letter = str(kwargs.get("letter", "")).lower()
    n = int(kwargs.get("let_frequency", 0))
    relation = kwargs.get("let_relation", "at least")
    if not letter:
        return True
    count = response.lower().count(letter)
    return _relation_satisfied(count, n, relation)


def _check_response_language(response: str, kwargs: dict) -> bool:
    expected = str(kwargs.get("language", "")).lower()
    if not expected:
        return True
    if not _LANGDETECT_AVAILABLE:
        # Cannot verify language without the optional dependency -> pass.
        return True
    if not response.strip():
        return False
    detected = _detect_language(response).lower()
    return detected == expected


def _check_number_sentences(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("num_sentences", 0))
    relation = kwargs.get("relation", "at least")
    return _relation_satisfied(_count_sentences(response), n, relation)


def _check_number_paragraphs(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("num_paragraphs", 0))
    return len(_split_paragraphs(response)) == n


def _check_number_words(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("num_words", 0))
    relation = kwargs.get("relation", "at least")
    return _relation_satisfied(_count_words(response), n, relation)


def _check_nth_paragraph_first_word(response: str, kwargs: dict) -> bool:
    num_paragraphs = int(kwargs.get("num_paragraphs", 0))
    nth = int(kwargs.get("nth_paragraph", 1))
    first_word = str(kwargs.get("first_word", "")).strip().lower()
    paragraphs = _split_paragraphs(response)
    if len(paragraphs) != num_paragraphs:
        return False
    if nth < 1 or nth > len(paragraphs):
        return False
    target = paragraphs[nth - 1].strip()
    # Strip leading punctuation/markup before grabbing the first word.
    match = re.search(r"[A-Za-z0-9']+", target)
    if not match:
        return False
    return match.group(0).lower() == first_word


def _check_number_placeholders(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("num_placeholders", 0))
    placeholders = re.findall(r"\[[^\[\]\n]+\]", response)
    return len(placeholders) >= n


def _check_postscript(response: str, kwargs: dict) -> bool:
    marker = str(kwargs.get("postscript_marker", "P.S.")).strip()
    if not marker:
        return True
    # Allow optional whitespace between the marker characters (e.g. "P. S.").
    flexible = r"\s*".join(re.escape(ch) for ch in marker if not ch.isspace())
    return re.search(flexible, response, flags=re.IGNORECASE) is not None


def _check_number_bullet_lists(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("num_bullets", 0))
    count = 0
    for line in response.splitlines():
        if re.match(r"^\s*[\*\-]\s+\S", line):
            count += 1
    return count == n


def _check_constrained_response(response: str, kwargs: dict) -> bool:
    constraint = kwargs.get("constraint", "")
    if isinstance(constraint, list):
        options = [str(c).strip() for c in constraint]
        return response.strip() in options
    return response.strip() == str(constraint).strip()


def _check_number_highlighted_sections(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("num_highlights", 0))
    # Match **double** highlights first, then *single* highlights.
    matches = re.findall(r"\*\*[^\*\n]+\*\*|\*[^\*\n]+\*", response)
    count = len([m for m in matches if m.strip("*").strip()])
    return count == n


def _check_multiple_sections(response: str, kwargs: dict) -> bool:
    spliter = str(kwargs.get("section_spliter", "Section"))
    n = int(kwargs.get("num_sections", 0))
    pattern = re.escape(spliter) + r"\s+\d+\s*[:.]"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    return len(matches) >= n


def _check_json_format(response: str, kwargs: dict) -> bool:
    text = response.strip()
    # Strip an optional surrounding markdown code fence.
    fence = re.match(r"^```[a-zA-Z0-9]*\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text:
        return False
    json.loads(text)
    return True


def _check_title(response: str, kwargs: dict) -> bool:
    matches = re.findall(r"<<([^\n]+?)>>", response)
    return any(m.strip() for m in matches)


def _check_two_responses(response: str, kwargs: dict) -> bool:
    parts = [p.strip() for p in response.split("******")]
    parts = [p for p in parts if p]
    return len(parts) == 2 and parts[0] != parts[1]


def _check_repeat_prompt(response: str, kwargs: dict) -> bool:
    prompt = str(kwargs.get("prompt_to_repeat", "")).strip().lower()
    if not prompt:
        return True
    resp = response.strip().lower()
    if resp.startswith(prompt):
        return True
    # Allow the prompt to appear near the start of the response.
    return prompt in resp[: len(prompt) + 200]


def _check_end_checker(response: str, kwargs: dict) -> bool:
    phrase = str(kwargs.get("end_phrase", "")).strip().lower()
    if not phrase:
        return True
    return response.strip().lower().endswith(phrase)


def _check_capital_word_frequency(response: str, kwargs: dict) -> bool:
    n = int(kwargs.get("capital_frequency", 0))
    relation = kwargs.get("capital_relation", "at least")
    words = re.findall(r"\b\w+\b", response)
    count = len([w for w in words if len(w) > 1 and w.isupper()])
    return _relation_satisfied(count, n, relation)


def _check_english_capital(response: str, kwargs: dict) -> bool:
    if not any(ch.isalpha() for ch in response):
        return False
    return response == response.upper()


def _check_english_lowercase(response: str, kwargs: dict) -> bool:
    if not any(ch.isalpha() for ch in response):
        return False
    return response == response.lower()


def _check_no_comma(response: str, kwargs: dict) -> bool:
    return "," not in response


# Dispatch table: instruction id -> checker function.
_CHECKERS = {
    "keywords:existence": _check_keywords_existence,
    "keywords:frequency": _check_keywords_frequency,
    "keywords:forbidden_words": _check_keywords_forbidden_words,
    "keywords:letter_frequency": _check_keywords_letter_frequency,
    "language:response_language": _check_response_language,
    "length_constraints:number_sentences": _check_number_sentences,
    "length_constraints:number_paragraphs": _check_number_paragraphs,
    "length_constraints:number_words": _check_number_words,
    "length_constraints:nth_paragraph_first_word": _check_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _check_number_placeholders,
    "detectable_content:postscript": _check_postscript,
    "detectable_format:number_bullet_lists": _check_number_bullet_lists,
    "detectable_format:constrained_response": _check_constrained_response,
    "detectable_format:number_highlighted_sections": _check_number_highlighted_sections,
    "detectable_format:multiple_sections": _check_multiple_sections,
    "detectable_format:json_format": _check_json_format,
    "detectable_format:title": _check_title,
    "combination:two_responses": _check_two_responses,
    "combination:repeat_prompt": _check_repeat_prompt,
    "startend:end_checker": _check_end_checker,
    "change_case:capital_word_frequency": _check_capital_word_frequency,
    "change_case:english_capital": _check_english_capital,
    "change_case:english_lowercase": _check_english_lowercase,
    "punctuation:no_comma": _check_no_comma,
}


def _check_instruction(instruction_id: str, kwargs: dict, response: str) -> bool:
    """Evaluate a single instruction. Unknown instructions pass; errors fail."""
    checker = _CHECKERS.get(instruction_id)
    if checker is None:
        # Unknown instruction -> do not penalize.
        return True
    try:
        return bool(checker(response, kwargs or {}))
    except Exception:
        return False


def compute_score(response: str, ground_truth_json: str) -> float:
    """Returns fraction of instructions satisfied (0.0 to 1.0).

    Args:
        response: the model's generated text.
        ground_truth_json: a JSON string encoding
            ``{"instruction_id_list": [...], "kwargs": [...]}``.

    Returns:
        The fraction of verifiable instructions the response satisfies, in ``[0.0, 1.0]``.
    """
    try:
        spec = json.loads(ground_truth_json)
    except Exception:
        return 0.0

    instruction_ids = spec.get("instruction_id_list", []) or []
    kwargs_list = spec.get("kwargs", []) or []

    if not instruction_ids:
        return 0.0

    response = response if isinstance(response, str) else str(response)

    satisfied = 0
    for i, instruction_id in enumerate(instruction_ids):
        kwargs = kwargs_list[i] if i < len(kwargs_list) and isinstance(kwargs_list[i], dict) else {}
        if _check_instruction(instruction_id, kwargs, response):
            satisfied += 1

    return satisfied / len(instruction_ids)
