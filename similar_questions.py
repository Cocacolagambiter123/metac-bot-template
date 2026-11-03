"""
Finds questions like the one being forecast that have already resolved on
Metaculus, so the forecaster can see how they turned out.

In Metaculus's survey of fall 2026 bot makers, 34% of the winning bots looked
up similar resolved questions, and none of the other bots did. Bot
tournaments reuse many question templates ("How many mass shootings will the
U.S. experience in November 2026?" follows the same question for October), so
earlier versions of a question are often the best base rate there is.

Once per question:
1. A cheap AI call writes up to three short search phrases.
2. Metaculus is searched for resolved questions matching each phrase.
3. The same cheap AI picks, from the titles alone, the ones that really ask
   about the same thing, and notes how each differs.
4. Their resolutions are looked up and the list is added to the research.
   Counts such as "4 of 6 resolved Yes" are worked out here, not by the AI.

Metaculus only shows an account the resolution of a question it has forecast
(or of questions in projects it has been given access to); for any other
question the API returns the resolution, description and resolution criteria
as null (its API gateway rules, documented in Metaculus PR #5261). So main.py
searches only questions the bot itself has forecast. The list grows as the
bot's own forecasts resolve: recurring questions in MiniBench and FutureEval
resolve within days or weeks, so earlier versions of the same question soon
become available.

Anything that goes wrong just means a shorter list or none: this must never
cost a question.

This module doesn't depend on forecasting-tools, so it can be tested offline.
main.py supplies the AI and the Metaculus search.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

MAX_QUERIES = 3
MAX_CANDIDATES = 30
MAX_SELECTED = 8


@dataclass(frozen=True)
class TargetQuestion:
    """The question being forecast."""

    title: str
    resolution_criteria: str = ""
    question_id: int | None = None
    # For one subquestion of a group question: its label.
    group_option: str = ""


@dataclass(frozen=True)
class PastQuestion:
    """A question that has resolved (or is expected to have, if `resolution` is None)."""

    title: str
    url: str
    question_type: str  # "binary", "numeric", "discrete", "multiple_choice" or "date"
    resolution: str | None  # readable, e.g. "Yes" or "412 cases"; None if not known yet
    resolved_on: date | None
    post_id: int | None = None
    question_id: int | None = None
    resolved_yes: bool | None = None  # for yes/no questions


@dataclass(frozen=True)
class SimilarQuestion:
    past: PastQuestion
    note: str  # how it relates to the question being forecast


Ask = Callable[[str], Awaitable[str]]
Search = Callable[[str], Awaitable[list[PastQuestion]]]
# Fills in a resolution that the search results left out. None if there is none.
Resolve = Callable[[PastQuestion], Awaitable["PastQuestion | None"]]


def _json_object(text: str) -> dict:
    found = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not found:
        raise ValueError("No JSON object in the answer")
    data = json.loads(found.group(0))
    if not isinstance(data, dict):
        raise ValueError("The answer is not a JSON object")
    return data


def _describe(target: TargetQuestion) -> str:
    text = f"Question: {target.title}"
    if target.group_option:
        text += f'\n(This is one subquestion of a group of related questions, about: "{target.group_option}")'
    if target.resolution_criteria:
        text += f"\nResolution criteria: {target.resolution_criteria[:800]}"
    return text


def build_query_prompt(target: TargetQuestion) -> str:
    return f"""
Write up to {MAX_QUERIES} short search phrases (2 to 6 words each) for finding questions on Metaculus that ask about the same thing as the question below and have already resolved: for example the same recurring question for an earlier month, or the same kind of event for a similar subject. Use the key words of the question and leave out dates and numbers.

{_describe(target)}

Answer with only one JSON object, no other text:
{{"queries": ["...", "..."]}}
""".strip()


def parse_queries(text: str) -> list[str]:
    raw = _json_object(text).get("queries")
    if not isinstance(raw, list):
        raise ValueError("No list of queries")
    queries: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split())
        if 1 <= len(query.split()) <= 8 and query.lower() not in (q.lower() for q in queries):
            queries.append(query)
    return queries[:MAX_QUERIES]


_MONTHS = set(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_FILLER = set(
    "a an the of in on at by for to from and or will be is are was were what how many much which "
    "who whose when where before after between than more less over under this that these those "
    "during within as with per its their".split()
)


def fallback_query(title: str) -> str:
    """The title's key words without dates, numbers or filler words."""
    words = [w.strip(".'-") for w in re.findall(r"[a-z][a-z.'&-]*", title.lower())]
    keep = [w for w in words if w and w not in _FILLER and w not in _MONTHS]
    return " ".join(keep[:6])


def build_selection_prompt(target: TargetQuestion, candidates: list[PastQuestion]) -> str:
    # Outcomes are left out on purpose: the choice should depend on how alike
    # the questions are, not on how they turned out.
    lines = "\n".join(
        f"{i}. [{c.question_type}] {c.title}" + (f" (resolved {c.resolved_on})" if c.resolved_on else "")
        for i, c in enumerate(candidates, 1)
    )
    return f"""
You help a forecaster find base rates. Below is the question to forecast, then numbered questions that have already resolved on Metaculus.

Pick the resolved questions whose outcomes would help forecast this one: the same recurring question for earlier periods, or the same quantity or kind of event with similar thresholds and definitions. Leave out questions that only share a topic.

{_describe(target)}

Resolved questions:
{lines}

Answer with only one JSON object, no other text. List at most {MAX_SELECTED}, most useful first, each with a few words on how it relates to or differs from the question to forecast:
{{"similar": [{{"index": 1, "note": "..."}}]}}
If none would help, answer {{"similar": []}}.
""".strip()


def parse_selection(text: str, n_candidates: int) -> list[tuple[int, str]]:
    """(candidate position, note) pairs, positions counted from 0."""
    raw = _json_object(text).get("similar")
    if not isinstance(raw, list):
        raise ValueError("No list of similar questions")
    picks: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if not 1 <= index <= n_candidates or index in seen:
            continue
        seen.add(index)
        picks.append((index - 1, " ".join(str(item.get("note", "")).split())[:200]))
    return picks[:MAX_SELECTED]


async def find_similar_questions(
    target: TargetQuestion,
    ask: Ask,
    search: Search,
    resolve: Resolve,
    as_of: date,
) -> list[SimilarQuestion]:
    """
    Resolved questions similar to `target`. Only questions resolved on or
    before `as_of` count: today in live use, or the day before a closed
    question opened when testing on it, so nothing later leaks in.
    """
    try:
        queries = parse_queries(await ask(build_query_prompt(target)))
    except Exception as error:
        logger.info(f"Couldn't read the search phrases ({error}), so the title's key words are used")
        queries = []
    if not queries:
        fallback = fallback_query(target.title)
        queries = [fallback] if fallback else []

    candidates: list[PastQuestion] = []
    seen: set[object] = set()
    for query in queries:
        try:
            hits = await search(query)
        except Exception as error:
            logger.warning(f"Metaculus search for {query!r} failed: {error}")
            continue
        for hit in hits:
            key = hit.question_id if hit.question_id is not None else hit.url
            if key in seen:
                continue
            seen.add(key)
            if target.question_id is not None and hit.question_id == target.question_id:
                continue
            if hit.resolved_on is None or hit.resolved_on > as_of:
                continue
            candidates.append(hit)
    candidates = candidates[:MAX_CANDIDATES]
    if not candidates:
        return []

    try:
        picks = parse_selection(await ask(build_selection_prompt(target, candidates)), len(candidates))
    except Exception as error:
        logger.warning(f"Couldn't read the choice of similar questions: {error}")
        return []

    similar: list[SimilarQuestion] = []
    for position, note in picks:
        candidate = candidates[position]
        try:
            past = candidate if candidate.resolution is not None else await resolve(candidate)
        except Exception as error:
            logger.warning(f"Couldn't look up how {candidate.url} resolved: {error}")
            continue
        if past is None or past.resolution is None:
            continue  # annulled, ambiguous, or not actually resolved
        if past.resolved_on is None or past.resolved_on > as_of:
            continue
        similar.append(SimilarQuestion(past, note))
    return similar


def format_for_prompt(similar: list[SimilarQuestion]) -> str:
    """The list as research text for the forecaster, newest first."""
    ordered = sorted(similar, key=lambda s: s.past.resolved_on or date.min, reverse=True)
    lines = [
        "Similar questions that have already resolved on Metaculus (found automatically, "
        "newest first). Check how each differs from this question (dates, thresholds, "
        "definitions, sources) before using it as a base rate:"
    ]
    for s in ordered:
        when = s.past.resolved_on.isoformat() if s.past.resolved_on else "date unknown"
        note = f" ({s.note})" if s.note else ""
        lines.append(f'- "{s.past.title}", resolved {when}: {s.past.resolution}.{note}')
    yes_no = [s.past.resolved_yes for s in ordered if s.past.resolved_yes is not None]
    if len(yes_no) >= 2:
        lines.append(f"Of these yes/no questions, {sum(yes_no)} of {len(yes_no)} resolved Yes.")
    return "\n".join(lines)
