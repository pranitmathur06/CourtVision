"""Stage 8 — natural-language play-by-play from structured events.

The LLM's job is narration, not decision-making: every fact it needs is already
computed upstream. That keeps hallucination risk low (spec §3), and
`validate_commentary` enforces it mechanically — any player or team named in a
line that isn't in the corresponding event is an error, and the graph retries.

LangGraph owns the control flow (narrate -> validate -> retry or finish); the
Anthropic SDK is called directly inside the narrate node.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from courtvision.config import Config
from courtvision.types import Event

PLAYER_MENTION = re.compile(r"\bPlayer\s+(\d+)\b", re.IGNORECASE)
TEAM_MENTION = re.compile(r"\bTeam\s+([AB])\b", re.IGNORECASE)
MAX_TIMESTAMP_DRIFT_S = 0.5

SYSTEM_PROMPT = """You are a basketball play-by-play commentator.

You will receive a JSON list of events that were computed by a computer vision
pipeline. Write exactly one line of commentary per event, in the same order.

Rules, which are absolute:
- Describe ONLY what is in the events. Never invent a player, team, action, score,
  foul, or game situation that is not present in the input.
- Refer to a player as "Player <track_id>" using the exact track_id from the event.
- Refer to a team as "Team A" or "Team B" using the exact team from the event.
- If an event has a null track_id, do not name any player in that line.
- Set each line's time_s to exactly the event's time_s.
- Vary the phrasing so it reads like live commentary, but never at the cost of accuracy.
"""


class CommentaryLine(BaseModel):
    time_s: float = Field(description="Timestamp in seconds, copied from the event")
    text: str = Field(description="One sentence of play-by-play commentary")


class Commentary(BaseModel):
    lines: list[CommentaryLine]


def format_timestamp(seconds: float) -> str:
    """Seconds to M:SS, for the human-readable log."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def validate_commentary(
    events: Sequence[Event], lines: Sequence[CommentaryLine]
) -> list[str]:
    """Return a list of fabrication/consistency errors. Empty means the output is faithful."""
    errors: list[str] = []
    if len(lines) != len(events):
        errors.append(
            f"line count {len(lines)} does not match event count {len(events)}"
        )
        return errors

    for index, (event, line) in enumerate(zip(events, lines)):
        if abs(line.time_s - event.time_s) > MAX_TIMESTAMP_DRIFT_S:
            errors.append(
                f"line {index}: timestamp {line.time_s} drifts from event {event.time_s}"
            )

        for mentioned in PLAYER_MENTION.findall(line.text):
            if event.track_id is None or int(mentioned) != event.track_id:
                errors.append(
                    f"line {index}: names Player {mentioned}, "
                    f"but the event's player is {event.track_id}"
                )

        for mentioned in TEAM_MENTION.findall(line.text):
            if event.team is None or mentioned.upper() != event.team:
                errors.append(
                    f"line {index}: names Team {mentioned.upper()}, "
                    f"but the event's team is {event.team}"
                )

    return errors


def events_to_payload(events: Sequence[Event]) -> str:
    return json.dumps(
        [
            {
                "time_s": round(e.time_s, 2),
                "track_id": e.track_id,
                "team": e.team,
                "action": e.action,
                "possession_change": e.possession_change,
            }
            for e in events
        ],
        indent=2,
    )


class Narrator(Protocol):
    def narrate(self, events: Sequence[Event]) -> Commentary: ...


class AnthropicNarrator:
    """Calls Claude to narrate a list of events."""

    def __init__(self, config: Config) -> None:
        import anthropic

        # Credentials resolve from ANTHROPIC_API_KEY or an `ant auth login` profile.
        self._client = anthropic.Anthropic()
        self._config = config

    def narrate(self, events: Sequence[Event]) -> Commentary:
        response = self._client.messages.parse(
            model=self._config.llm_model,
            max_tokens=self._config.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": events_to_payload(events)}],
            output_format=Commentary,
        )
        # A refusal returns HTTP 200 with no usable content — check before reading.
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused to narrate: {response.stop_details}")
        return response.parsed_output


class CommentaryState(TypedDict):
    events: list[Event]
    lines: list[CommentaryLine]
    errors: list[str]
    attempts: int


def build_graph(narrator: Narrator, config: Config):
    """narrate -> validate -> (retry if fabricated and attempts remain, else finish)."""

    def narrate(state: CommentaryState) -> dict:
        commentary = narrator.narrate(state["events"])
        return {"lines": list(commentary.lines), "attempts": state["attempts"] + 1}

    def validate(state: CommentaryState) -> dict:
        return {"errors": validate_commentary(state["events"], state["lines"])}

    def route(state: CommentaryState) -> str:
        if not state["errors"]:
            return "done"
        if state["attempts"] >= config.llm_max_attempts:
            return "done"
        return "retry"

    graph = StateGraph(CommentaryState)
    graph.add_node("narrate", narrate)
    graph.add_node("validate", validate)
    graph.add_edge(START, "narrate")
    graph.add_edge("narrate", "validate")
    graph.add_conditional_edges("validate", route, {"retry": "narrate", "done": END})
    return graph.compile()


def generate_commentary(
    events: Sequence[Event], narrator: Narrator, config: Config
) -> tuple[list[CommentaryLine], list[str]]:
    """Run the graph. Returns (lines, errors); non-empty errors mean it gave up."""
    if not events:
        return [], []

    final = build_graph(narrator, config).invoke(
        {"events": list(events), "lines": [], "errors": [], "attempts": 0}
    )
    return final["lines"], final["errors"]
