"""V8 — Commentary generation sanity check (spec §6).

Feeds hand-constructed events through the real LangGraph + Claude path and prints
the output for manual review, plus the mechanical fabrication check.
"""

from __future__ import annotations

import os
import sys

from courtvision.commentary import (
    AnthropicNarrator,
    format_timestamp,
    generate_commentary,
)
from courtvision.config import Config
from courtvision.types import Event

# Hand-built events covering the interesting cases: a possession change, a null
# holder (loose ball), and both teams.
EVENTS = [
    Event(0.0, 7, "A", "dribble", False),
    Event(2.0, 7, "A", "pass", False),
    Event(4.0, 3, "A", "shot", True),
    Event(6.5, None, None, "rebound", False),
    Event(8.0, 11, "B", "dribble", True),
    Event(10.5, 11, "B", "shot", False),
]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "V8 note — ANTHROPIC_API_KEY not set; relying on an `ant auth login` "
            "profile. If this fails to authenticate, set the key and re-run."
        )

    config = Config()
    try:
        lines, errors = generate_commentary(EVENTS, AnthropicNarrator(config), config)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        print(f"V8 FAIL — narrator raised {type(exc).__name__}: {exc}")
        return 1

    print(f"--- commentary ({config.llm_model}) ---")
    for line in lines:
        print(f"{format_timestamp(line.time_s)} — {line.text}")
    print("--- end ---")

    ok = not errors and len(lines) == len(EVENTS)
    verdict = "PASS" if ok else "FAIL"
    print(f"V8 {verdict} — {len(lines)}/{len(EVENTS)} lines, {len(errors)} fabrication errors")
    for error in errors:
        print(f"  {error}")
    if ok:
        print("  Now read the lines above: do they describe these events, and read naturally?")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
