import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Literal

import dotenv
import numpy as np

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    exit_if_any_failed,
    has_asknews_key,
    has_real_env,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DiscreteQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

from numeric_model.blend import blend_cdfs
from numeric_model.matching import (
    MatchResult,
    QuestionInfo,
    anchor_text,
    match_question,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class ThrottledLlm(GeneralLlm):
    """
    A GeneralLlm that caps how many AI requests are in flight at once.

    The bot forecasts every open question at the same time, 5 times each, so it
    can send dozens of requests at once. New Claude Console accounts allow only
    a few concurrent requests and reject the rest with a rate-limit error (2 of
    50 forecasts failed this way in the first test run). Every ThrottledLlm
    shares one cap, because the provider counts requests across all models.
    Set MAX_CONCURRENT_AI_CALLS to change it (default 4).
    """

    _limiter_loop: asyncio.AbstractEventLoop | None = None
    _limiter: asyncio.Semaphore | None = None

    @classmethod
    def _get_limiter(cls) -> asyncio.Semaphore:
        # The script calls asyncio.run() once per tournament, and a semaphore
        # can't be shared between event loops, so make one per loop.
        loop = asyncio.get_running_loop()
        if cls._limiter is None or cls._limiter_loop is not loop:
            raw = os.getenv("MAX_CONCURRENT_AI_CALLS", "").strip()
            limit = int(raw) if raw.isdigit() and int(raw) > 0 else 4
            cls._limiter_loop = loop
            cls._limiter = asyncio.Semaphore(limit)
        return cls._limiter

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        # Wraps each single API call, so a slot is freed while a failed call
        # waits to be retried.
        async with self._get_limiter():
            return await super()._mockable_direct_call_to_model(prompt)


def pick_models() -> dict[str, str | GeneralLlm]:
    """
    Choose the AI models based on which API key is set.

    - ANTHROPIC_API_KEY (a key from the Claude Console) is used first.
    - OPENROUTER_API_KEY (e.g. the free credits Metaculus hands out) is the fallback.

    The forecasts themselves use Claude Opus 5.5 at high reasoning effort: in
    Metaculus's spring 2026 benchmark, high-reasoning versions beat their
    standard versions in all 8 model pairs tested. A cheap model (Claude Haiku
    5.5) handles the side tasks: turning answers into numbers and summarising.

    Set FORECAST_MODEL or HELPER_MODEL (litellm model names, e.g.
    "anthropic/claude-fable-5-1") to override either choice without editing code,
    and FORECAST_EFFORT (low, medium, high, xhigh or max) to change how hard the
    forecasting model thinks.
    """
    if has_real_env("ANTHROPIC_API_KEY"):
        forecast_model = "anthropic/claude-opus-5-5"
        helper_model = "anthropic/claude-haiku-5-5"
    elif has_real_env("OPENROUTER_API_KEY"):
        forecast_model = "openrouter/anthropic/claude-opus-5.5"
        helper_model = "openrouter/anthropic/claude-haiku-5.5"
    else:
        forecast_model = ""
        helper_model = ""
    forecast_model = os.getenv("FORECAST_MODEL", "").strip() or forecast_model
    helper_model = os.getenv("HELPER_MODEL", "").strip() or helper_model
    if not forecast_model or not helper_model:
        print(
            "❌  No model chosen. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY, "
            "or set both FORECAST_MODEL and HELPER_MODEL."
        )
        sys.exit(1)

    # Reasoning effort is "high" unless FORECAST_EFFORT says otherwise.
    # Metaculus found high-reasoning versions beat standard ones; there is no
    # published evidence yet that "xhigh" or "max" beat "high", and they cost
    # more and take longer.
    effort = os.getenv("FORECAST_EFFORT", "").strip().lower() or "high"
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        print(
            "❌  FORECAST_EFFORT must be low, medium, high, xhigh or max "
            f"(got {effort!r})."
        )
        sys.exit(1)
    # Thinking counts towards max_tokens, and higher effort thinks for longer,
    # so give it more room and more time.
    max_tokens, timeout_seconds = {
        "xhigh": (32000, 1200),
        "max": (48000, 1800),
    }.get(effort, (16000, 600))

    # Current Claude models reject the `temperature` setting, so none is sent.
    forecaster = ThrottledLlm(
        model=forecast_model,
        reasoning_effort=effort,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        allowed_tries=3,
    )
    helper = ThrottledLlm(
        model=helper_model,
        max_tokens=8000,
        timeout=180,
        allowed_tries=3,
    )
    if has_asknews_key():
        researcher: str | GeneralLlm = "asknews/news-summaries"
    else:
        # check_environment() has already warned that forecasts will be made
        # without news research until an AskNews key is added.
        researcher = "no_research"
    return {
        "default": forecaster,
        "summarizer": helper,
        "researcher": researcher,
        "parser": helper,
    }


def tournaments_to_forecast(client: MetaculusClient) -> list[str | int]:
    """
    The tournaments 'tournament' mode forecasts on.

    By default these are the current FutureEval season and MiniBench. Set the
    BOT_TOURNAMENTS variable to a comma-separated list of tournament IDs or slugs
    (e.g. "fall-futureeval-2026" to leave MiniBench out, or the new season's ID
    when it starts) to change this without editing code.
    """
    raw = os.getenv("BOT_TOURNAMENTS", "").strip()
    if not raw:
        return [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    return [int(part) if part.isdigit() else part for part in ids]


def numeric_model_settings() -> tuple[bool, float]:
    """
    Whether to use the statistical model for numeric questions about known
    data series (see numeric_model/), and how much weight to give it.

    - USE_NUMERIC_MODEL: "true" to turn it on. Off by default until it has
      been validated.
    - NUMERIC_MODEL_WEIGHT: 0 to 1, default 0. At 0 the model's forecast is
      only shown to the AI in the prompt, as a starting point. Above 0 the
      final distribution is also a mixture: this share from the model, the
      rest from the AI.
    """
    use = os.getenv("USE_NUMERIC_MODEL", "").strip().lower() in ("1", "true", "yes", "on")
    raw = os.getenv("NUMERIC_MODEL_WEIGHT", "").strip()
    try:
        weight = float(raw) if raw else 0.0
    except ValueError:
        weight = -1.0
    if not 0.0 <= weight <= 1.0:
        print(f"❌  NUMERIC_MODEL_WEIGHT must be a number from 0 to 1 (got {raw!r}).")
        sys.exit(1)
    return use, weight


def question_info(question: NumericQuestion) -> QuestionInfo:
    """The fields of a forecasting-tools question that the numeric model needs."""
    return QuestionInfo(
        title=question.question_text,
        background=question.background_info or "",
        resolution_criteria=question.resolution_criteria or "",
        fine_print=question.fine_print or "",
        unit_of_measure=question.unit_of_measure or "",
        lower_bound=float(question.lower_bound),
        upper_bound=float(question.upper_bound),
        open_lower_bound=question.open_lower_bound,
        open_upper_bound=question.open_upper_bound,
        is_discrete=isinstance(question, DiscreteQuestion),
        open_time=question.open_time,
        scheduled_resolution_time=question.scheduled_resolution_time,
        page_url=question.page_url or "",
        group_option=question.group_question_option or "",
    )


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    # Made per event loop by _research_limiter().
    _research_limiter_loop: asyncio.AbstractEventLoop | None = None
    _research_limiter_semaphore: asyncio.Semaphore | None = None
    _structure_output_validation_samples = 2
    # One numeric-model match per question, shared by its 5 forecasts.
    _numeric_matches: dict[str, MatchResult] = {}
    _numeric_match_locks: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}

    ##################################### RESEARCH #####################################

    @classmethod
    def _research_limiter(cls) -> asyncio.Semaphore:
        """
        Caps how many questions are researched at once.

        main.py calls asyncio.run() once per tournament (FutureEval, then
        MiniBench). A semaphore that made a task wait in one event loop raises
        "bound to a different event loop" when a task waits on it in the next
        one. The research cap used to be one semaphore shared by the whole
        class, so every MiniBench question that had to wait its turn for
        research would fail. Each event loop now gets its own.
        """
        loop = asyncio.get_running_loop()
        if cls._research_limiter_semaphore is None or cls._research_limiter_loop is not loop:
            cls._research_limiter_loop = loop
            cls._research_limiter_semaphore = asyncio.Semaphore(cls._max_concurrent_questions)
        return cls._research_limiter_semaphore

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._research_limiter():
            researcher = self.get_llm("researcher")
            prompt = self._get_research_prompt(question, researcher)
            try:
                research = await self._call_researcher(researcher, prompt)
            except Exception as error:
                # A failed news search should not cost us the whole question
                # (missed questions are expensive), so forecast without it.
                logger.warning(
                    f"Research failed for URL {question.page_url}, so it will be "
                    f"forecast without news research. Error: {error}"
                )
                research = ""
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        if not research.strip():
            # Nothing to summarise. Asking the model anyway makes it post a
            # confused "the research didn't come through" reply in the comment.
            return "No news research was used for this forecast."
        return await super().summarize_research(question, research)

    async def _call_researcher(
        self, researcher: str | GeneralLlm, prompt: str
    ) -> str:
        if isinstance(researcher, GeneralLlm):
            return await researcher.invoke(prompt)
        if (
            researcher == "asknews/news-summaries"
            or researcher == "asknews/deep-research/low-depth"
            or researcher == "asknews/deep-research/medium-depth"
            or researcher == "asknews/deep-research/high-depth"
        ):
            return await AskNewsSearcher().call_preconfigured_version(
                researcher, prompt
            )
        if researcher.startswith("smart-searcher"):
            model_name = researcher.removeprefix("smart-searcher/")
            searcher = SmartSearcher(
                model=model_name,
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
                use_advanced_filters=False,
            )
            return await searcher.invoke(prompt)
        if not researcher or researcher == "None" or researcher == "no_research":
            return ""
        return await self.get_llm("researcher", "llm").invoke(prompt)

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            {self._metaculus_conventions(question)}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            {self._metaculus_conventions(question)}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        match = await self._numeric_model_match(question)
        anchor = ""
        if match is not None and match.accepted:
            anchor = "Statistical model:\n" + anchor_text(
                match, question.unit_of_measure or ""
            )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            {self._metaculus_conventions(question)}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            {anchor}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        prediction = await self._numeric_prompt_to_forecast(question, prompt)
        if match is not None and match.accepted:
            prediction = self._blend_with_numeric_model(question, prediction, match)
        return prediction

    async def _numeric_model_match(
        self, question: NumericQuestion
    ) -> MatchResult | None:
        """
        Matches the question to a known data series, once per question (the
        5 forecasts share the result). None if USE_NUMERIC_MODEL is off.
        """
        use, _ = numeric_model_settings()
        if not use:
            return None
        key = str(question.id_of_question or question.page_url or question.question_text)
        loop = asyncio.get_running_loop()
        held = self._numeric_match_locks.get(key)
        if held is None or held[0] is not loop:
            held = (loop, asyncio.Lock())
            self._numeric_match_locks[key] = held
        async with held[1]:
            if key not in self._numeric_matches:
                helper = self.get_llm("parser", "llm")
                try:
                    match = await match_question(
                        question_info(question),
                        helper.invoke,
                        as_of=datetime.now(timezone.utc).date(),
                    )
                except Exception as error:
                    # The model is optional: never let it cost us the question.
                    match = MatchResult(accepted=False, reasons=[f"matching failed: {error}"])
                logger.info(f"{match.describe()} (URL {question.page_url})")
                self._numeric_matches[key] = match
            return self._numeric_matches[key]

    def _blend_with_numeric_model(
        self,
        question: NumericQuestion,
        prediction: ReasonedPrediction[NumericDistribution],
        match: MatchResult,
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Mixes the AI's distribution with the model's, if NUMERIC_MODEL_WEIGHT
        is above 0. Falls back to the AI's forecast if anything goes wrong.
        """
        _, weight = numeric_model_settings()
        if weight <= 0:
            return prediction
        try:
            ai_cdf = prediction.prediction_value.get_cdf()
            grid = np.array([point.value for point in ai_cdf])
            heights = np.array([point.percentile for point in ai_cdf])
            blended = blend_cdfs(
                heights,
                grid,
                match.scaled_samples,
                weight,
                question.open_lower_bound,
                question.open_upper_bound,
            )
            distribution = NumericDistribution.from_question(
                [
                    Percentile(value=float(value), percentile=float(height))
                    for value, height in zip(grid, blended)
                ],
                question,
            )
            distribution.get_cdf()  # validate now, so a failure falls back below
        except Exception as error:
            logger.warning(
                f"Couldn't blend with the numeric model for URL {question.page_url}, "
                f"so the AI's forecast is used alone. Error: {error}"
            )
            return prediction
        reasoning = (
            f"{prediction.reasoning}\n\nThe final distribution mixes this forecast "
            f"({1 - weight:.0%}) with the statistical model's ({weight:.0%})."
        )
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            {self._metaculus_conventions(question)}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _metaculus_conventions(self, question: MetaculusQuestion) -> str:
        """
        Metaculus questions assume the outcome was still undecided when the
        question opened. One bot maker reported losing 90 points in spring 2026
        by not telling their bot this, so every forecasting prompt includes it.
        """
        opened = ""
        if question.open_time is not None:
            opened = f" It opened at {question.open_time.isoformat(timespec='minutes')}."
        return clean_indents(
            f"""
            Metaculus convention: treat this question as undecided at the moment it opened.{opened}
            If your research seems to show the outcome is already settled, check the dates: news from before the question opened does not settle it, because the question would not have been asked about something already decided.
            Always give a full forecast, even when you think the outcome is close to certain.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    use_numeric_model, numeric_model_weight = numeric_model_settings()
    if use_numeric_model:
        print(
            "📈  Numeric model on for questions about known data series "
            f"(blend weight {numeric_model_weight:g}).\n"
        )
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. The models are pinned in pick_models() above, because
    # forecasting-tools' built-in defaults are old models (e.g. GPT-4o) that
    # scored below zero in recent FutureEval seasons. Keep 5 forecasts per
    # question: bot makers who cut this to save money reported it backfired.
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=pick_models(),
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        tournaments = tournaments_to_forecast(client)
        print(f"🗂️   Tournaments: {', '.join(str(t) for t in tournaments)}\n")
        if os.getenv("BOT_TOURNAMENTS", "").strip():
            TOURNAMENT_URLS["tournament"] = (
                f"https://www.metaculus.com/tournament/{tournaments[0]}/"
            )
        forecast_reports = []
        for tournament_id in tournaments:
            forecast_reports += asyncio.run(
                template_bot.forecast_on_tournament(
                    tournament_id, return_exceptions=True
                )
            )
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
    # Mark the run as failed (red in GitHub Actions) if any question failed,
    # so problems such as a used-up spending limit don't go unnoticed.
    exit_if_any_failed(forecast_reports)
