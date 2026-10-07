"""Pluggable judges: score a replayed trajectory against its baseline."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from replayagent.trajectory import Trajectory


class JudgeResult(BaseModel):
    """The verdict of one judge on a (baseline, candidate) pair."""

    judge: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Judge(ABC):
    """Base class for judges. Subclass and implement :meth:`evaluate`."""

    name: str = "judge"

    @abstractmethod
    def evaluate(self, baseline: Trajectory, candidate: Trajectory) -> JudgeResult:
        """Compare a replayed trajectory against its baseline."""
        raise NotImplementedError

    def _result(self, passed: bool, score: float, reason: str = "", **details: Any) -> JudgeResult:
        return JudgeResult(
            judge=self.name, passed=passed, score=score, reason=reason, details=details
        )


class ExactMatchJudge(Judge):
    """Passes when the final answers are exactly equal (after stripping)."""

    name = "exact_match"

    def evaluate(self, baseline: Trajectory, candidate: Trajectory) -> JudgeResult:
        a, b = baseline.final_answer(), candidate.final_answer()
        if a is None or b is None:
            return self._result(False, 0.0, "missing final answer", baseline=a, candidate=b)
        passed = a.strip() == b.strip()
        return self._result(
            passed,
            1.0 if passed else 0.0,
            "final answers match" if passed else "final answers differ",
        )


class ContainsJudge(Judge):
    """Passes when the candidate's final answer contains a string or regex."""

    name = "contains"

    def __init__(self, pattern: str, regex: bool = False, case_sensitive: bool = True) -> None:
        self.pattern = pattern
        self.regex = regex
        self.case_sensitive = case_sensitive

    def evaluate(self, baseline: Trajectory, candidate: Trajectory) -> JudgeResult:
        text = candidate.final_answer() or ""
        haystack = text if self.case_sensitive else text.lower()
        needle = self.pattern if self.case_sensitive else self.pattern.lower()
        if self.regex:
            matched = (
                re.search(self.pattern, text, 0 if self.case_sensitive else re.IGNORECASE)
                is not None
            )
        else:
            matched = needle in haystack
        return self._result(
            matched,
            1.0 if matched else 0.0,
            f"final answer {'matches' if matched else 'does not match'} {self.pattern!r}",
            pattern=self.pattern,
        )


class SemanticSimilarityJudge(Judge):
    """Offline semantic-similarity proxy via token-overlap (Jaccard) — no API needed.

    Not a true embedding similarity; a cheap, deterministic gate for CI.
    Passes when the Jaccard similarity of the final answers >= threshold.
    """

    name = "semantic_similarity"

    def __init__(self, threshold: float = 0.7) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def evaluate(self, baseline: Trajectory, candidate: Trajectory) -> JudgeResult:
        a, b = baseline.final_answer() or "", candidate.final_answer() or ""
        ta, tb = self._tokens(a), self._tokens(b)
        score = 1.0 if not ta and not tb else len(ta & tb) / max(len(ta | tb), 1)
        passed = score >= self.threshold
        return self._result(
            passed, score, f"token-overlap similarity {score:.2f} (threshold {self.threshold})"
        )


class ToolCallJudge(Judge):
    """Passes when the candidate makes the same tool calls (by name, in order)."""

    name = "tool_calls"

    def __init__(self, strict_order: bool = True) -> None:
        self.strict_order = strict_order

    def evaluate(self, baseline: Trajectory, candidate: Trajectory) -> JudgeResult:
        expected = baseline.tool_call_names()
        actual = candidate.tool_call_names()
        if self.strict_order:
            passed = expected == actual
        else:
            passed = sorted(expected) == sorted(actual)
        return self._result(
            passed,
            1.0 if passed else 0.0,
            "tool call sequences match" if passed else f"tool calls differ: {expected} vs {actual}",
            expected=expected,
            actual=actual,
        )


class LLMJudge(Judge):
    """LLM-as-judge. Calls OpenAI or Anthropic — only when an API key is present.

    Set ``OPENAI_API_KEY`` (default) or ``ANTHROPIC_API_KEY`` with
    ``provider="anthropic"``. Raises :class:`RuntimeError` when no key is found,
    so CI never silently makes network calls.
    """

    name = "llm_judge"

    def __init__(
        self,
        criteria: str = (
            "Does the candidate's final answer solve the task as well as the baseline's?"
        ),
        provider: str = "openai",
        model: str | None = None,
    ) -> None:
        if provider not in ("openai", "anthropic"):
            raise ValueError("provider must be 'openai' or 'anthropic'")
        self.criteria = criteria
        self.provider = provider
        self.model = model or ("gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest")

    def _verdict(self, baseline: Trajectory, candidate: Trajectory) -> tuple[bool, float, str]:
        prompt = (
            "You are evaluating an AI agent run against a baseline.\n\n"
            f"Criteria: {self.criteria}\n\n"
            f"Task input: {baseline.input}\n\n"
            f"Baseline final answer: {baseline.final_answer()}\n\n"
            f"Candidate final answer: {candidate.final_answer()}\n\n"
            "Reply with exactly one line in the format: PASS|score|reason or "
            "FAIL|score|reason, where score is a float between 0 and 1."
        )
        if self.provider == "openai":
            import openai  # lazy import: optional dependency

            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            line = (resp.choices[0].message.content or "").strip().splitlines()[0]
        else:
            import anthropic  # lazy import: optional dependency

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            block = resp.content[0]
            line = (getattr(block, "text", "") or "").strip().splitlines()[0]

        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3 or parts[0] not in ("PASS", "FAIL"):
            raise RuntimeError(f"Unparseable judge verdict: {line!r}")
        try:
            score = float(parts[1])
        except ValueError:
            raise RuntimeError(f"Unparseable judge score: {line!r}")
        return parts[0] == "PASS", min(max(score, 0.0), 1.0), parts[2]

    def evaluate(self, baseline: Trajectory, candidate: Trajectory) -> JudgeResult:
        key = (
            os.environ.get("OPENAI_API_KEY")
            if self.provider == "openai"
            else os.environ.get("ANTHROPIC_API_KEY")
        )
        if not key:
            key_name = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
            raise RuntimeError(f"LLMJudge needs {key_name} in the environment.")
        passed, score, reason = self._verdict(baseline, candidate)
        return self._result(passed, score, reason, model=self.model, provider=self.provider)


JUDGES: dict[str, type[Judge]] = {
    "exact": ExactMatchJudge,
    "contains": ContainsJudge,
    "similarity": SemanticSimilarityJudge,
    "tools": ToolCallJudge,
    "llm": LLMJudge,
}


def get_judge(name: str, **kwargs: Any) -> Judge:
    """Instantiate a judge by short name.

    One of: ``exact``, ``contains``, ``similarity``, ``tools``, ``llm``.
    """
    try:
        cls = JUDGES[name]
    except KeyError:
        raise ValueError(f"Unknown judge {name!r}. Choose from: {', '.join(sorted(JUDGES))}")
    return cls(**kwargs)
