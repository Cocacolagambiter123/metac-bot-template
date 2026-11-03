"""
Regression test: researching questions must keep working when main.py runs a
second tournament in a new event loop (asyncio.run is called once per
tournament).

Needs forecasting-tools, so it is skipped where the bot's dependencies aren't
installed. Run from the repository root with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import main

    HAVE_BOT = True
except ImportError:
    HAVE_BOT = False


@unittest.skipUnless(HAVE_BOT, "forecasting-tools is not installed")
class ResearchLimiterTest(unittest.TestCase):
    def test_works_in_a_second_event_loop(self) -> None:
        # No __init__ needed: the limiter only uses class attributes.
        bot = main.FallTemplateBot2026.__new__(main.FallTemplateBot2026)

        async def research() -> None:
            async with bot._research_limiter():
                await asyncio.sleep(0.01)  # held across an await, as real research is

        async def tournament() -> list[BaseException]:
            results = await asyncio.gather(*(research() for _ in range(3)), return_exceptions=True)
            return [r for r in results if isinstance(r, BaseException)]

        self.assertEqual(asyncio.run(tournament()), [])  # e.g. FutureEval
        self.assertEqual(asyncio.run(tournament()), [])  # e.g. MiniBench, in a new loop

    def test_limits_concurrency_to_the_class_setting(self) -> None:
        bot = main.FallTemplateBot2026.__new__(main.FallTemplateBot2026)
        active = 0
        peak = 0

        async def research() -> None:
            nonlocal active, peak
            async with bot._research_limiter():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        async def run() -> None:
            await asyncio.gather(*(research() for _ in range(5)))

        asyncio.run(run())
        self.assertEqual(peak, main.FallTemplateBot2026._max_concurrent_questions)


if __name__ == "__main__":
    unittest.main()
