"""Built-in judges: every extracted claim is reviewed before it enters the
graph. Two LLM judges score independent criteria; the panel combines their
verdicts into an accept / revise / reject decision.

* groundedness     - is the claim actually stated by the source passage?
* qualifier fidelity - are perspective/time/quote faithful to the passage?

A groundedness failure rejects the claim outright (hallucination). A
qualifier failure sends the claim back to the extractor for one judge-guided
repair round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLMClient
from .models import Claim, JudgeVerdict
from .prompts import GROUNDEDNESS_SYSTEM, QUALIFIER_SYSTEM, VERDICT_SCHEMA, judge_user

ACCEPT = "accept"
REVISE = "revise"
REJECT = "reject"


class LLMJudge:
    def __init__(self, client: LLMClient, criterion: str, system_prompt: str,
                 threshold: float = 0.7):
        self.client = client
        self.criterion = criterion
        self.system_prompt = system_prompt
        self.threshold = threshold

    def review(self, claim: Claim, chunk_text: str) -> JudgeVerdict:
        claim_json = json.dumps(
            {
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object": claim.object,
                "perspective": vars(claim.perspective),
                "time": vars(claim.time),
                "quote": claim.provenance.quote if claim.provenance else "",
            },
            indent=2,
        )
        raw = self.client.complete_json(
            self.system_prompt, judge_user(chunk_text, claim_json), VERDICT_SCHEMA
        )
        score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
        # The score is authoritative; the boolean must agree with the threshold.
        passed = score >= self.threshold and bool(raw.get("passed", False))
        return JudgeVerdict(
            criterion=self.criterion,
            score=score,
            passed=passed,
            feedback=str(raw.get("feedback", "")),
            judge=self.client.name,
        )


@dataclass
class Review:
    decision: str  # accept | revise | reject
    verdicts: list[JudgeVerdict] = field(default_factory=list)

    @property
    def feedback(self) -> str:
        return " ".join(v.feedback for v in self.verdicts if not v.passed and v.feedback)


class JudgePanel:
    def __init__(self, client: LLMClient, threshold: float = 0.7):
        self.groundedness = LLMJudge(client, "groundedness", GROUNDEDNESS_SYSTEM, threshold)
        self.qualifiers = LLMJudge(client, "qualifier_fidelity", QUALIFIER_SYSTEM, threshold)

    def review(self, claim: Claim, chunk_text: str) -> Review:
        grounded = self.groundedness.review(claim, chunk_text)
        if not grounded.passed:
            # Not supported by the source: unrepairable, drop it.
            return Review(decision=REJECT, verdicts=[grounded])
        qualifier = self.qualifiers.review(claim, chunk_text)
        verdicts = [grounded, qualifier]
        if qualifier.passed:
            return Review(decision=ACCEPT, verdicts=verdicts)
        return Review(decision=REVISE, verdicts=verdicts)
