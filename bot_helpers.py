"""
Runtime helpers for the template bot: environment validation, startup/result
banners, and suppression of noisy upstream warnings.

Kept separate from main.py so that file can focus on the bot's forecasting
logic. main_with_no_framework.py keeps its own inline copies on purpose --
it's meant to be a single-file reference implementation.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Sequence


# Placeholder values shipped in .env.template. If a real env var still equals
# one of these the user forgot to replace it; we'd rather fail loudly here than
# inside the SDK three layers down.
_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def _is_real_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip() and val.strip() not in _PLACEHOLDER_ENV_VALUES)


def has_real_env(name: str) -> bool:
    """True if the environment variable is set to a real (non-placeholder) value."""
    return _is_real_env(name)


def has_asknews_key() -> bool:
    """True if AskNews news search is configured (API key, or client ID + secret)."""
    return _is_real_env("ASKNEWS_API_KEY") or (
        _is_real_env("ASKNEWS_CLIENT_ID") and _is_real_env("ASKNEWS_SECRET")
    )


def silence_noisy_dependencies() -> None:
    """
    Quiet warnings from transitive deps that fire on import and confuse new
    users. Must be called *before* importing forecasting_tools.
    """
    warnings.filterwarnings(
        "ignore", message=r".*does not support cost tracking.*"
    )
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(
        logging.ERROR
    )
    # Streamlit installs its own logger hierarchy; suppress via its own API.
    try:
        from streamlit.logger import set_log_level

        set_log_level("error")
    except ImportError:
        pass
    # LiteLLM is verbose at INFO; its WARNING level is enough for us.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


def check_environment(strict: bool = True) -> None:
    """
    Verify METACULUS_TOKEN and at least one LLM key are set. On failure with
    strict=True, exits the process with a non-zero status.
    """
    problems: list[str] = []

    if not _is_real_env("METACULUS_TOKEN"):
        problems.append(
            "METACULUS_TOKEN is missing or still a placeholder. "
            "Get one at https://www.metaculus.com/futureeval/participate/"
        )

    has_llm_key = any(
        _is_real_env(k)
        for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_llm_key:
        problems.append(
            "No LLM key set. Set OPENROUTER_API_KEY, OPENAI_API_KEY or "
            "ANTHROPIC_API_KEY (free OpenRouter credits: "
            "https://forms.gle/aQdYMq9Pisrf1v7d8). The Metaculus LLM proxy that "
            "the bot used to fall back to is no longer available."
        )

    if not has_asknews_key():
        # Not fatal: the bot still forecasts, but without news research, which
        # scored noticeably worse in Metaculus's comparisons.
        print(
            "⚠️   No AskNews key (ASKNEWS_API_KEY) is set, so the bot will forecast "
            "without news research. Free AskNews access for bot makers: "
            "https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/\n"
        )

    if problems:
        print("❌  Setup problems:")
        for p in problems:
            print(f"    • {p}")
        if strict:
            sys.exit(1)


def print_startup_banner(run_mode: str, will_publish: bool) -> None:
    publish = "publish=yes" if will_publish else "publish=no (dry run)"
    print(f"🤖  Running mode={run_mode}, {publish}\n")


def print_run_summary_banner(
    forecast_reports: Sequence[Any],
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    """
    End-of-run summary printed via print() (not logger) so it survives log
    filtering. Shows count, per-question URLs, and any failure tracebacks.
    If tournament_url is given, it's included as a footer link.
    """
    # Lazy import so this module is usable in contexts where forecasting_tools
    # isn't installed (e.g. unit tests of the banner format).
    from forecasting_tools import ForecastReport

    valid = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]
    banner = "=" * 80

    print()
    print(banner)

    if not forecast_reports:
        print("ℹ️   No new questions to forecast on this run.")
        print(banner)
        print()
        return

    if valid and not exceptions:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"🎉  Bot {verb} {len(valid)} forecast(s).")
    elif valid and exceptions:
        print(
            f"⚠️   Partial — {len(valid)} succeeded, {len(exceptions)} failed."
        )
    else:
        print(f"❌  All {len(exceptions)} attempt(s) failed.")

    if valid:
        print()
        for r in valid:
            note = f"  (with {len(r.errors)} minor error(s))" if r.errors else ""
            print(f"  ✅ {r.question.page_url}{note}")
        costs = [r.price_estimate for r in valid if r.price_estimate]
        if costs:
            # An estimate: it leaves out failed questions and any model the
            # cost tracker doesn't have prices for.
            print(f"\n  💵 Estimated AI cost of these forecasts: ${sum(costs):.2f}")
        if will_publish and tournament_url:
            print(f"\n  Tournament: {tournament_url}")

    if exceptions:
        print()
        for exc in exceptions:
            msg = str(exc)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"  ❌ {type(exc).__name__}: {msg}")

    print(banner)
    print()


# Phrases the AI providers use when a spending limit is hit or credits run out.
_SPENDING_LIMIT_PHRASES = (
    "api usage limit",  # Claude Console: your own spend limit, or the tier's monthly cap
    "enforced_spend_limit_reached",
    "credit balance is too low",  # Claude Console: prepaid credits used up
    "insufficient credits",  # OpenRouter
    "requires more credits",  # OpenRouter
)


def _exception_texts(exc: BaseException, seen: set[int] | None = None) -> list[str]:
    """The messages of an exception and of every exception it wraps."""
    seen = set() if seen is None else seen
    if id(exc) in seen:
        return []
    seen.add(id(exc))
    texts = [f"{type(exc).__name__}: {exc}"]
    for inner in (exc.__cause__, exc.__context__):
        if inner is not None:
            texts += _exception_texts(inner, seen)
    for inner in getattr(exc, "exceptions", None) or []:  # ExceptionGroup members
        if isinstance(inner, BaseException):
            texts += _exception_texts(inner, seen)
    return texts


def exit_if_any_failed(forecast_reports: Sequence[Any]) -> None:
    """
    Exit with a non-zero status if any question failed, so the GitHub Actions
    run shows as failed instead of quietly succeeding. If the cause looks like a
    spending limit or empty credit balance, say so plainly.
    """
    failures = [r for r in forecast_reports if isinstance(r, BaseException)]
    if not failures:
        return
    all_text = "\n".join(
        text for exc in failures for text in _exception_texts(exc)
    ).lower()
    if any(phrase in all_text for phrase in _SPENDING_LIMIT_PHRASES):
        print(
            "💸  The AI provider refused requests because a spending limit was "
            "reached or the credits ran out. Raise the limit or add credits "
            "(Claude Console: https://platform.claude.com/settings/billing). "
            "Later runs will retry the missed questions while they're still open."
        )
    print(f"❌  {len(failures)} question(s) failed, so this run is marked as failed.")
    sys.exit(1)
