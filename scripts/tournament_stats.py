"""
Counts how many questions tournaments have opened per day recently, and what
forecasting them would cost the bot. Use it to plan before switching on the
scheduled tournament workflow, or to budget a new season.

Group questions count once per subquestion, since the bot forecasts each one.
The default cost per question is what the Test Bot run measured on
2026-11-03 (Claude Opus 5.5 at high effort, 5 forecasts per question, AskNews
research): $7.03 for 10 questions. AskNews's own cost isn't included.

It only reads question lists: no AI calls, nothing submitted.

Usage (needs METACULUS_TOKEN):
    python scripts/tournament_stats.py [--tournaments minibench,33121] [--days 14] [--cost-per-question 0.70]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting_tools import MetaculusClient  # noqa: E402
from forecasting_tools.helpers.metaculus_client import ApiFilter  # noqa: E402


async def opened_since(client: MetaculusClient, tournament: str | int, since: datetime) -> list:
    return await client.get_questions_matching_filter(
        ApiFilter(
            allowed_tournaments=[tournament],
            open_time_gt=since,
            group_question_mode="unpack_subquestions",
            other_url_parameters={"with_cp": "false", "include_conditional_cps": "false"},
        ),
        num_questions=3000,
        error_if_question_target_missed=False,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournaments", default="")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--cost-per-question", type=float, default=0.70)
    args = parser.parse_args()

    client = MetaculusClient()
    tournaments: list[str | int] = [
        int(t) if t.strip().isdigit() else t.strip()
        for t in args.tournaments.split(",") if t.strip()
    ] or [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)

    lines: list[str] = []
    for tournament in tournaments:
        try:
            questions = await opened_since(client, tournament, since)
        except Exception as error:
            lines.append(f"{tournament}: couldn't load questions ({error})")
            continue
        opened = [q for q in questions if q.open_time is not None and since <= q.open_time <= now]
        upcoming = [q for q in questions if q.open_time is not None and q.open_time > now]
        per_day = Counter(q.open_time.date() for q in opened)
        average = len(opened) / args.days
        lines.append(
            f"{tournament}: {len(opened)} questions opened in the last {args.days} days "
            f"({average:.1f} a day), {len(upcoming)} more scheduled to open. "
            f"At ${args.cost_per_question:.2f} a question: about ${average * args.cost_per_question:.2f} "
            f"a day, ${average * args.cost_per_question * 30:.0f} a month."
        )
        for day in sorted(per_day):
            lines.append(f"  {day}: {per_day[day]}")

    report = "\n".join(lines)
    print(report)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write(f"## Questions opened per day\n\n```\n{report}\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
