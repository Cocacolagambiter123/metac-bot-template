"""
Offline tests for the similar-question lookup (similar_questions.py), with
stand-ins for the AI, the Metaculus search and the resolution lookup.

The last test class checks the conversions from forecasting-tools questions in
main.py; it needs forecasting-tools and is skipped without it (it runs in
GitHub Actions).

Run from the repository root with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from similar_questions import (  # noqa: E402
    MAX_SELECTED,
    PastQuestion,
    SimilarQuestion,
    TargetQuestion,
    fallback_query,
    find_similar_questions,
    format_for_prompt,
    parse_queries,
    parse_selection,
)

TODAY = date(2026, 11, 3)
TARGET = TargetQuestion(
    title="How many mass shootings will the U.S. experience in November 2026?",
    resolution_criteria="Resolves to the Gun Violence Archive count for November 2026.",
    question_id=500,
)


def past(qid: int, title: str, resolution: str | None, resolved_on: date | None, yes: bool | None = None,
         kind: str = "numeric") -> PastQuestion:
    return PastQuestion(
        title=title,
        url=f"https://www.metaculus.com/questions/{qid}",
        question_type=kind,
        resolution=resolution,
        resolved_on=resolved_on,
        post_id=qid,
        question_id=qid,
        resolved_yes=yes,
    )


OCTOBER = past(401, "How many mass shootings will the U.S. experience in October 2026?", "38", date(2026, 11, 2))
SEPTEMBER = past(301, "How many mass shootings will the U.S. experience in September 2026?", None, date(2026, 10, 2))
ANNULLED = past(201, "How many mass shootings will the U.S. experience in August 2026?", None, date(2026, 9, 2))
LATER = past(601, "How many mass shootings will the U.S. experience in December 2026?", "35", date(2027, 1, 2))
ITSELF = past(500, TARGET.title, None, date(2026, 12, 2))
UNRELATED = past(77, "Will a mass shooting occur at a U.S. school in 2026?", "Yes", date(2026, 6, 1), True, "binary")


class ParsingTest(unittest.TestCase):
    def test_parse_queries(self) -> None:
        text = 'Here: {"queries": ["mass shootings", "Mass  shootings", "gun violence archive count", 7, "' + "x " * 12 + '"]}'
        self.assertEqual(parse_queries(text), ["mass shootings", "gun violence archive count"])
        with self.assertRaises(ValueError):
            parse_queries("no json")
        with self.assertRaises(ValueError):
            parse_queries('{"queries": "mass shootings"}')

    def test_fallback_query(self) -> None:
        query = fallback_query(TARGET.title)
        self.assertTrue(query.startswith("mass shootings"))
        self.assertNotIn("november", query)
        self.assertNotIn("2026", query)
        self.assertNotIn("how", query.split())

    def test_parse_selection(self) -> None:
        items = [{"index": 2, "note": "same, a month earlier"}, {"index": 2, "note": "dup"}, {"index": 0},
                 {"index": 9}, {"index": True}, {"index": "1"}, {"index": 1}]
        self.assertEqual(parse_selection(json.dumps({"similar": items}), 3), [(1, "same, a month earlier"), (0, "")])
        many = [{"index": i} for i in range(1, 20)]
        self.assertEqual(len(parse_selection(json.dumps({"similar": many}), 30)), MAX_SELECTED)
        self.assertEqual(parse_selection('{"similar": []}', 3), [])


class FakeWorld:
    """Stand-ins for the AI, the search and the resolution lookup."""

    def __init__(self, hits: dict[str, list[PastQuestion]], picks: list[int] | None = None,
                 queries: list[str] | None = None, broken_ai: bool = False) -> None:
        self.hits = hits
        self.picks = picks
        self.queries = queries if queries is not None else list(hits)
        self.broken_ai = broken_ai
        self.prompts: list[str] = []
        self.searched: list[str] = []
        self.resolved: list[int | None] = []

    async def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.broken_ai:
            return "sorry, no idea"
        if "search phrases" in prompt:
            return json.dumps({"queries": self.queries})
        count = len(re.findall(r"^\d+\. \[", prompt, flags=re.MULTILINE))
        picks = self.picks if self.picks is not None else range(1, count + 1)
        return json.dumps({"similar": [{"index": i, "note": f"note {i}"} for i in picks]})

    async def search(self, query: str) -> list[PastQuestion]:
        self.searched.append(query)
        if query not in self.hits:
            raise ConnectionError("Metaculus is down")
        return self.hits[query]

    async def resolve(self, question: PastQuestion) -> PastQuestion | None:
        self.resolved.append(question.question_id)
        if question.question_id == SEPTEMBER.question_id:
            return past(301, SEPTEMBER.title, "41", date(2026, 10, 2))
        return None  # e.g. annulled

    def run(self, target: TargetQuestion = TARGET, as_of: date = TODAY) -> list[SimilarQuestion]:
        return asyncio.run(find_similar_questions(target, self.ask, self.search, self.resolve, as_of))


class FindSimilarQuestionsTest(unittest.TestCase):
    def test_end_to_end(self) -> None:
        world = FakeWorld(
            {"mass shootings": [OCTOBER, ITSELF, LATER, SEPTEMBER], "gun violence archive": [OCTOBER, ANNULLED, UNRELATED]},
            picks=[1, 2, 3],  # October, September, August (annulled)
        )
        found = world.run()
        self.assertEqual([s.past.question_id for s in found], [401, 301])
        self.assertEqual(found[1].past.resolution, "41")  # looked up separately
        self.assertEqual(found[0].note, "note 1")
        # Only the picks without a known resolution were looked up.
        self.assertEqual(world.resolved, [301, 201])
        selection_prompt = world.prompts[1]
        self.assertNotIn(TARGET.title + " (", selection_prompt.split("Resolved questions:")[1])  # itself left out
        self.assertNotIn("December 2026", selection_prompt)  # resolved after as_of: left out
        self.assertEqual(selection_prompt.count("October 2026"), 1)  # duplicates removed
        self.assertNotIn(": 38", selection_prompt)  # outcomes aren't shown while choosing

    def test_nothing_resolved_after_the_cutoff_is_used(self) -> None:
        # Testing on a question that opened on 2026-10-10: October's version
        # resolved later, so it must not appear.
        world = FakeWorld({"mass shootings": [OCTOBER, SEPTEMBER]})
        found = world.run(as_of=date(2026, 10, 9))
        self.assertEqual([s.past.question_id for s in found], [301])

    def test_falls_back_to_title_words_when_the_ai_fails(self) -> None:
        world = FakeWorld({fallback_query(TARGET.title): [OCTOBER]}, broken_ai=True)
        self.assertEqual(world.run(), [])  # the choice can't be read either
        self.assertEqual(world.searched, [fallback_query(TARGET.title)])

    def test_search_failures_are_skipped(self) -> None:
        world = FakeWorld({"mass shootings": [OCTOBER]}, queries=["broken search", "mass shootings"])
        self.assertEqual([s.past.question_id for s in world.run()], [401])
        world = FakeWorld({}, queries=["broken search"])
        self.assertEqual(world.run(), [])

    def test_group_label_is_in_the_prompts(self) -> None:
        target = TargetQuestion(title="What will these indices close at on December 31?", group_option="Nasdaq-100")
        world = FakeWorld({"nasdaq": []}, queries=["nasdaq"])
        world.run(target)
        self.assertIn('about: "Nasdaq-100"', world.prompts[0])


class FormatTest(unittest.TestCase):
    def test_newest_first_with_yes_count(self) -> None:
        first = past(1, "Will A happen by June?", "No", date(2026, 6, 30), False, "binary")
        second = past(2, "Will A happen by September?", "Yes", date(2026, 9, 30), True, "binary")
        text = format_for_prompt([SimilarQuestion(first, "earlier deadline"), SimilarQuestion(second, "")])
        self.assertLess(text.index("September"), text.index("June"))
        self.assertIn('"Will A happen by June?", resolved 2026-06-30: No. (earlier deadline)', text)
        self.assertIn("1 of 2 resolved Yes", text)
        self.assertNotIn("resolved Yes.", format_for_prompt([SimilarQuestion(second, "")]))


try:
    from forecasting_tools import BinaryQuestion, NumericQuestion

    import main

    HAVE_BOT = True
except ImportError:
    HAVE_BOT = False


@unittest.skipUnless(HAVE_BOT, "forecasting-tools is not installed")
class ConversionTest(unittest.TestCase):
    """The conversions from forecasting-tools questions in main.py."""

    def numeric(self, resolution: str | None, label: str | None = None) -> "NumericQuestion":
        return NumericQuestion(
            question_text="How many cases will be reported?",
            id_of_post=10, id_of_question=11, page_url="https://www.metaculus.com/questions/10",
            lower_bound=0.0, upper_bound=500.0, open_lower_bound=False, open_upper_bound=True,
            unit_of_measure="cases", resolution_string=resolution, group_question_option=label,
            actual_resolution_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
        )

    def test_numeric(self) -> None:
        converted = main.past_question(self.numeric("412"))
        assert converted is not None
        self.assertEqual(converted.resolution, "412 cases")
        self.assertEqual(converted.resolved_on, date(2026, 10, 2))
        self.assertEqual(converted.question_type, "numeric")
        above = main.past_question(self.numeric("above_upper_bound"))
        assert above is not None
        self.assertEqual(above.resolution, "above the question's range (above 500 cases)")

    def test_unknown_and_annulled(self) -> None:
        unknown = main.past_question(self.numeric(None))
        assert unknown is not None
        self.assertIsNone(unknown.resolution)  # looked up later
        self.assertIsNone(main.past_question(self.numeric("annulled")))

    def test_binary_and_group_label(self) -> None:
        question = BinaryQuestion(
            question_text="Will it happen?", resolution_string="yes", group_question_option="by June",
            actual_resolution_time=datetime(2026, 6, 30, tzinfo=timezone.utc),
        )
        converted = main.past_question(question)
        assert converted is not None
        self.assertEqual((converted.resolution, converted.resolved_yes), ("Yes", True))
        self.assertEqual(converted.title, "Will it happen? (by June)")


if __name__ == "__main__":
    unittest.main()
