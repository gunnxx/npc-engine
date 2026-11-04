from dataclasses import dataclass, field
import logging
import re

from openai import OpenAI

from src.common.budget import LLM_MODEL, MESSAGE_OVERHEAD_TOKENS, RECENT_TOKENS, SUMMARY_TOKENS, TOKEN_COUNTER

logger = logging.getLogger(__name__)

PROTECTED_RECENT_MESSAGES = 2
"""Newest messages that are always kept verbatim (the latest player/NPC exchange)."""

# =========
# Memory
# =========


@dataclass
class Message:
    """One turn of dialogue stored in a session."""

    role: str
    """``user`` for the player, ``assistant`` for the NPC."""

    content: str
    """Text of the turn."""

    tokens: int
    """Tokens the turn occupies in a prompt, overhead included."""

    importance: int
    """Heuristic score used to pick which turns to drop first when over budget."""

    def to_dict(self) -> dict:
        """Convert to the chat-message shape used in the protocol.

        Returns:
            Dictionary with ``role`` and ``content`` keys.
        """
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """Conversation memory of one NPC: a rolling summary and the verbatim recent turns."""

    summary: str = ""
    """Condensed account of everything that was folded out of ``messages``."""

    messages: list[Message] = field(default_factory=list)
    """Recent turns kept verbatim, oldest first."""

    def recent_tokens(self) -> int:
        """Count the tokens occupied by the verbatim messages.

        Returns:
            Sum of the per-message token counts.
        """
        return sum(message.tokens for message in self.messages)


class ConversationMemory:
    """In-process store of every session, with the budget enforcement logic."""

    def __init__(self):
        """Constructor."""
        self._sessions: dict[str, Session] = {}

    # ==============
    # Public methods
    # ==============

    def get(self, session_id: str) -> Session:
        """Return the session for an id, creating it on first use.

        Args:
            session_id: Conversation identifier.

        Returns:
            The session's memory.
        """
        return self._sessions.setdefault(session_id, Session())

    def add_message(self, session_id: str, role: str, content: str) -> Session:
        """Append a turn and bring the session back within budget.

        Args:
            session_id: Conversation identifier.
            role: ``user`` or ``assistant``.
            content: Text of the turn.

        Returns:
            The updated session.
        """
        session = self.get(session_id)
        session.messages.append(
            Message(
                role=role,
                content=content,
                tokens=TOKEN_COUNTER.count(content) + MESSAGE_OVERHEAD_TOKENS,
                importance=score_importance(content),
            )
        )
        if session.recent_tokens() > RECENT_TOKENS:
            self._drop_low_importance(session)
        if session.recent_tokens() > RECENT_TOKENS:
            self.summarize(session_id)
        return session

    def summarize(self, session_id: str) -> int:
        """Fold older turns into the summary, keeping as many recent turns verbatim as the budget allows.

        Args:
            session_id: Conversation identifier.

        Returns:
            Number of messages that were folded into the summary.
        """
        session = self.get(session_id)
        keep = self._count_recent_that_fit(session)
        folded = session.messages[: len(session.messages) - keep]
        if folded:
            session.summary = summarize_with_llm(session.summary, [message.to_dict() for message in folded])
            session.messages = session.messages[len(folded):]
        logger.info("Summarized %d messages in session %r (%d tokens of summary).",
                    len(folded), session_id, TOKEN_COUNTER.count(session.summary))
        return len(folded)

    def reset(self, session_id: str):
        """Forget a session entirely.

        Args:
            session_id: Conversation identifier.
        """
        self._sessions.pop(session_id, None)

    # ===============
    # Private methods
    # ===============

    def _drop_low_importance(self, session: Session):
        """Remove low-importance turns outside the protected window until the budget fits or none are left.

        Args:
            session: Session to trim in place.
        """
        while session.recent_tokens() > RECENT_TOKENS:
            candidates = [m for m in session.messages[:-PROTECTED_RECENT_MESSAGES] if m.importance < 8]
            if not candidates:
                return
            victim = min(candidates, key=lambda message: message.importance)
            session.messages.remove(victim)
            logger.info("Dropped low-importance turn (%d): %.60r", victim.importance, victim.content)

    def _count_recent_that_fit(self, session: Session) -> int:
        """Pick how many of the newest turns stay verbatim after a fold.

        Args:
            session: Session being summarized.

        Returns:
            At most six (three player/NPC exchanges), reduced until the kept turns fit
            :data:`src.common.budget.RECENT_TOKENS`, but never below :data:`PROTECTED_RECENT_MESSAGES` (or the
            message count, if smaller).
        """
        keep = min(6, len(session.messages))
        while keep > PROTECTED_RECENT_MESSAGES and sum(m.tokens for m in session.messages[-keep:]) > RECENT_TOKENS:
            keep -= 1
        return keep


# ==========
# Importance
# ==========


def score_importance(content: str) -> int:
    """Estimate how much a turn matters for future dialogue.

    Args:
        content: Text of the turn.

    Returns:
        Non-negative score. Short acknowledgements such as "ok" or "thanks" score zero. Anything else scores its
        token count capped at 50, plus 20 when it asks a question and 10 when it mentions a number.
    """
    if re.fullmatch(r"\W*(ok(ay)?|sure|thanks?( you)?|yes|no|yeah|nice|cool|got it|i see)\W*", content, re.IGNORECASE):
        return 0
    score = min(TOKEN_COUNTER.count(content), 50)
    if "?" in content:
        score += 20
    if re.search(r"\d", content):
        score += 10
    return score


# ============
# Summarizer
# ============


def summarize_with_llm(previous_summary: str, messages: list[dict]) -> str:
    """Condense old turns into a new rolling summary with GPT.

    Args:
        previous_summary: Summary produced so far, empty on the first call.
        messages: Chat messages being folded out of the verbatim history.

    Returns:
        New summary covering ``previous_summary`` and ``messages``.
    """
    system_prompt = (
        "You condense RPG dialogue between a player and a non-player character into a compact memory. "
        "Write in third person, past tense, as a single paragraph. Keep facts the character may need later: the "
        "player's name, goals, items, places, decisions and promises, plus any facts the character looked up. Drop "
        f"greetings and filler. Stay under {SUMMARY_TOKENS // 2} words."
    )
    transcript = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    user_prompt = f"Memory so far:\n{previous_summary or '(none)'}\n\nNew dialogue to fold in:\n{transcript}"
    response = OpenAI().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=SUMMARY_TOKENS,
    )
    return response.choices[0].message.content.strip()
