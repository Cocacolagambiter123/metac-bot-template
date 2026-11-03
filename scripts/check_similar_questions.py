"""
Checks the similar-question lookup (similar_questions.py) on CLOSED questions.

For each closed or resolved question in the given tournaments, this runs the
lookup as it would have run the day before the question opened: only
questions that had resolved by then can be found, so nothing later leaks in.
It prints the search phrases and how many results each found, the questions
chosen and how they resolved, and how the question itself resolved. That shows
whether Metaculus's search works and whether the choices are useful.

Metaculus only shows an account the outcomes of questions it has forecast, so
the lookup (like the live bot) searches only the bot's own past questions.
Until the bot has forecast questions that later resolved, this finds nothing.

It submits nothing and never looks at open questions: tuning a bot on open
questions in a scored tournament breaks Metaculus's "no human in the loop"
rule.

Usage (needs METACULUS_TOKEN and an AI key, as for the bot):
    python scripts/check_similar_questions.py [--tournaments 33121,minibench] [--limit 30]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting_tools import MetaculusClient, MetaculusQuestion  # noqa: E402
from forecasting_tools.helpers.metaculus_client import ApiFilter  # noqa: E402

from main import (  # noqa: E402
    past_question,
    pick_models,
    resolve_past_question,
    search_resolved_questions,
    target_question,
)
from similar_questions import PastQuestion, find_similar_questions  # noqa: E402


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())}


def raw_resolution_fields(post_id: int) -> str:
    """What the API returns about a post's resolution, to explain a missing one."""
    found = MetaculusClient().get_question_by_post_id(post_id)
    post = found.api_json if not isinstance(found, list) else found[0].api_json
    questions = [post["question"]] if post.get("question") else list(
        (post.get("group_of_questions") or {}).get("questions", [])
    )
    shown = {
        key: post.get(key) for key in ("status", "resolved", "curation_status") if key in post
    }
    for i, question in enumerate(questions[:2]):
        shown[f"question[{i}]"] = {
            key: value for key, value in question.items()
            if "resol" in key or key in ("status", "actual_close_time", "type")
        }
    return str(shown)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournaments", default="")
    parser.add_argument("--limit", type=int, default=30, help="how many closed questions to check")
    args = parser.parse_args()

    client = MetaculusClient()
    tournaments: list[str | int] = [
        int(t) if t.strip().isdigit() else t.strip()
        for t in args.tournaments.split(",") if t.strip()
    ] or [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]
    questions = await client.get_questions_matching_filter(
        ApiFilter(
            allowed_statuses=["closed", "resolved"],
            allowed_tournaments=tournaments,
            group_question_mode="unpack_subquestions",
        ),
        num_questions=args.limit,
        error_if_question_target_missed=False,
    )
    now = datetime.now(timezone.utc)
    questions = [q for q in questions if q.close_time is not None and q.close_time <= now]
    print(f"Checking {len(questions)} closed questions from {tournaments}\n")

    ask = pick_models()["parser"].invoke
    blocks: list[str] = []
    explained_missing = False
    with_similar = 0
    total_similar = 0
    hit_count = 0
    relevant_hits = 0
    for question in questions:
        searches: list[tuple[str, list[PastQuestion]]] = []

        async def search(query: str) -> list[PastQuestion]:
            hits = await search_resolved_questions(query)
            searches.append((query, hits))
            return hits

        lookups: list[tuple[PastQuestion, PastQuestion | None]] = []

        async def resolve(candidate: PastQuestion) -> PastQuestion | None:
            result = await resolve_past_question(candidate)
            lookups.append((candidate, result))
            return result

        opened = question.open_time or question.published_time or now
        as_of = opened.date() - timedelta(days=1)
        try:
            similar = await find_similar_questions(
                target_question(question), ask, search, resolve, as_of
            )
        except Exception as error:
            blocks.append(f"ERROR  {question.question_text}: {error}")
            continue
        own = await own_resolution(question)
        title = question.question_text + (f" [{question.group_question_option}]" if question.group_question_option else "")
        lines = [f"{title}", f"  {question.page_url} (opened {opened:%Y-%m-%d}; resolved {own})"]
        for query, hits in searches:
            shared = [h for h in hits if words(h.title) & words(query)]
            hit_count += len(hits)
            relevant_hits += len(shared)
            lines.append(f"  search {query!r}: {len(hits)} results, {len(shared)} sharing a word with the search")
        missing = [c for c, r in lookups if r is None or r.resolution is None]
        if lookups:
            lines.append(f"  chosen: {len(lookups)} needed their resolution looked up, {len(missing)} had none")
        if missing and not explained_missing and missing[0].post_id is not None:
            explained_missing = True
            try:
                lines.append(f"  API fields for {missing[0].url}: {raw_resolution_fields(missing[0].post_id)}")
            except Exception as error:
                lines.append(f"  couldn't fetch {missing[0].url}: {error}")
        if similar:
            with_similar += 1
            total_similar += len(similar)
        for s in similar:
            lines.append(f"  -> {s.past.title} | resolved {s.past.resolved_on}: {s.past.resolution} | {s.note}")
        blocks.append("\n".join(lines))

    summary = (
        f"{with_similar} of {len(questions)} questions got similar resolved questions "
        f"({total_similar} in all). Search results sharing a word with their search phrase: "
        f"{relevant_hits} of {hit_count}."
    )
    report = "\n\n".join(blocks)
    print(report + "\n\n" + summary)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write(f"## Similar-question lookup on closed questions\n\n{summary}\n\n```\n{report}\n```\n")
    return 0


async def own_resolution(question: MetaculusQuestion) -> str | None:
    converted = past_question(question)
    if converted is None:
        return None
    try:
        resolved = await resolve_past_question(converted)
    except Exception:
        return None
    return resolved.resolution if resolved else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
