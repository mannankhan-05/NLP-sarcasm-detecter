"""Text cleaning. Sarcasm-preserving tokenization vs. classical TF-IDF cleanup."""

from __future__ import annotations

import re
import string

INTENSIFIERS = frozenset(
    """
    so very really extremely totally literally actually obviously clearly
    definitely absolutely certainly surely completely utterly highly
    incredibly amazingly ridiculously perfectly wonderful great fantastic
    awesome brilliant lovely nice just
    """.split()
)

HYPERBOLE = frozenset(
    """
    always never ever everyone nobody everything nothing forever
    million billion infinite dying dead kill love hate worst best
    """.split()
)

# Compact stoplist for the classical baseline only (sarcasm models keep function words).
STOPWORDS = frozenset(
    """
    a an the and or but if while of at by for to in on from with as is are
    was were be been being it this that these those i you he she we they
    me him her us them my your his our their
    """.split()
)


def preserve_text(text: str) -> str:
    """Keep casing and punctuation — both carry sarcasm signal."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def aggressive_clean(text: str) -> str:
    """Lowercase, strip punctuation, drop stopwords, Porter-like suffix stem."""
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return " ".join(_crude_stem(t) for t in tokens)


def _crude_stem(token: str) -> str:
    for suffix in ("ing", "ingly", "edly", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text or "")


_IRONY_RE = re.compile(
    r"(could have been|worked out so well|so well last time|best part of my (day|week|life)|"
    r"love being stuck|love being|completely trust|totally trust|oh,? sure|yeah,? right|"
    r"another meeting|why don't we get|do you hear yourself|tiny garage sale|cop a feel|"
    r"as if|yeah right|oh great|oh wonderful|how wonderful)",
    re.I,
)
_GREETING_RE = re.compile(
    r"^(hi|hii+|hello|hey|yo|sup|thanks|thank you|ok|okay|yes|no|hmm+|lol|haha|"
    r"how are you|how's it going|what's up|good morning|good night)\b",
    re.I,
)
_SIMPLE_COMPLIMENT_RE = re.compile(
    r"^(is )?(?P<name>[\w'.-]+)( \w+){0,3} (is|are|am|'s) (a |an )?"
    r"(good|nice|fine|kind|sweet|great|cool|smart|honest|best)( \w+){0,3}\??$",
    re.I,
)
_FACTUAL_RE = re.compile(
    r"^[\w'.-]+( [\w'.-]+){0,6} (is|are|am|was|were|'s) "
    r"(a |an |the |doing |going |studying |working |living |taking |making )",
    re.I,
)
_SIMPLE_EVAL_RE = re.compile(
    r"^(that |this |the )?[\w'-]+ (is|was|are|were) (good|nice|fine|ok|okay|bad|great)\.?$",
    re.I,
)
_CUE_OPENER_RE = re.compile(
    r"^(yeah|oh|sure),?\s+(because|right|sure|great|another)\b",
    re.I,
)


def isolated_text_kind(text: str) -> str:
    """How to treat a typed-only UI line: force_ns | allow | boost_s.

    Everyday names and copular facts have no sarcasm markers. The sitcom
    embedding head over-predicts sarcastic on that OOD text, so we require
    surface cues (or a long dialogue-like line) before trusting a sarcastic label.
    """
    text = preserve_text(text)
    words = [t for t in tokenize_words(text) if re.search(r"[A-Za-z]", t)]
    n = len(words)
    if n == 0:
        return "force_ns"
    if _IRONY_RE.search(text) or _CUE_OPENER_RE.search(text):
        return "boost_s"
    if n <= 3:
        return "force_ns"
    if _GREETING_RE.search(text):
        return "force_ns"
    if _SIMPLE_COMPLIMENT_RE.search(text) or _SIMPLE_EVAL_RE.search(text) or _FACTUAL_RE.search(text):
        return "force_ns"
    if n < 10:
        # Short unmarked statements default to literal.
        return "force_ns"
    return "allow"


def apply_isolated_prior(p_sarc: float, text: str) -> tuple[float, str]:
    kind = isolated_text_kind(text)
    p = max(0.0, min(1.0, float(p_sarc)))
    if kind == "force_ns":
        return min(p, 0.22), (
            "No sarcasm markers in this wording, so it is treated as non-sarcastic. "
            "Everyday names and simple facts are not sitcom sarcasm."
        )
    if kind == "boost_s":
        return max(0.02, min(0.98, max(p, 0.66))), ""
    return max(0.02, min(0.98, p)), ""

