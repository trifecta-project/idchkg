"""End-to-end pipeline tests driven by a scripted MockLLM.

The mock is queued with responses in the exact order the pipeline consumes
them: one extraction per chunk, then per claim a groundedness verdict, a
qualifier verdict, and (for revisions) a repair extraction.
"""

from idchkg.judge import ACCEPT, REJECT, REVISE, JudgePanel
from idchkg.llm.mock import MockLLM
from idchkg.models import ACCEPTED, REJECTED, REPAIRED, Chunk, Claim, Perspective, Provenance
from idchkg.pipeline import Pipeline, PipelineConfig


def _extraction(*claims):
    return {"entities": [], "claims": list(claims)}


def _claim(subject, predicate, obj, viewpoint="author", stance="asserts",
           sentiment="neutral", when=None, quote="q"):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "perspective": {"viewpoint": viewpoint, "stance": stance, "sentiment": sentiment},
        "time": {"when": when, "valid_from": None, "valid_to": None},
        "quote": quote,
    }


def _verdict(score, passed, feedback=""):
    return {"score": score, "passed": passed, "feedback": feedback}


def _run(mock, text="Coffee is a beverage. Historians dispute the colonial framing."):
    from idchkg.models import Document

    pipeline = Pipeline(
        PipelineConfig(backend="mock"),
        extractor_client=mock,
        judge_client=mock,
    )
    # Single short paragraph -> exactly one chunk -> one extract() call.
    return pipeline.run([Document(id="doc", text=text)])


def test_accepted_claim_enters_graph():
    mock = MockLLM().queue(
        _extraction(_claim("coffee", "is a", "beverage")),  # chunk 0
        _verdict(1.0, True),                                 # groundedness
        _verdict(1.0, True),                                 # qualifier
    )
    result = _run(mock)
    assert len(result.accepted) == 1
    assert result.accepted[0].status == ACCEPTED
    assert result.graph.find_entity("coffee") is not None
    # verdicts are recorded on the claim's provenance
    assert len(result.accepted[0].provenance.verdicts) == 2


def test_ungrounded_claim_is_rejected_without_repair():
    mock = MockLLM().queue(
        _extraction(_claim("coffee", "cures", "cancer")),
        _verdict(0.1, False, "not supported by the passage"),  # groundedness fails
    )
    result = _run(mock)
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert result.rejected[0].status == REJECTED
    # graph has no claim edges for a rejected statement
    assert result.graph.claims == []


def test_qualifier_failure_triggers_repair_then_accept():
    mock = MockLLM().queue(
        _extraction(_claim("coffee", "refines", "society", viewpoint="author")),
        _verdict(0.9, True),                                        # grounded ok
        _verdict(0.4, False, "viewpoint should be 'colonial writers'"),  # qualifier fails
        # repair re-extracts with the corrected viewpoint
        _extraction(_claim("coffee", "refines", "society", viewpoint="colonial writers")),
        _verdict(0.9, True),                                        # grounded ok again
        _verdict(1.0, True),                                        # qualifier now ok
    )
    result = _run(mock)
    assert len(result.repaired) == 1
    assert result.repaired[0].status == REPAIRED
    assert result.repaired[0].perspective.viewpoint == "colonial writers"


def test_qualifier_failure_rejected_when_repair_yields_nothing():
    mock = MockLLM().queue(
        _extraction(_claim("coffee", "refines", "society")),
        _verdict(0.9, True),
        _verdict(0.3, False, "bad viewpoint"),
        _extraction(),  # repair returns no claim -> reject
    )
    result = _run(mock)
    assert result.repaired == []
    assert len(result.rejected) == 1


def test_invalid_stance_is_dropped_before_judging():
    mock = MockLLM().queue(
        {"entities": [], "claims": [_claim("a", "b", "c", stance="bogus")]},
        _extraction(),
    )
    result = _run(mock)
    assert result.accepted == [] and result.rejected == []
    assert any("invalid stance" in e for e in result.errors)


def test_stats_report_shape():
    mock = MockLLM().queue(
        _extraction(_claim("coffee", "is a", "beverage")),
        _verdict(1.0, True),
        _verdict(1.0, True),
    )
    stats = _run(mock).stats()
    assert stats["claims_accepted"] == 1
    assert set(stats) >= {"entities", "claims_accepted", "claims_rejected", "documents"}


def test_panel_score_overrides_inconsistent_passed_flag():
    # judge claims passed=True but score is below threshold -> treated as fail
    mock = MockLLM().queue(_verdict(0.2, True))
    panel = JudgePanel(mock, threshold=0.7)
    claim = Claim("a", "b", "c", provenance=Provenance("d", "d#c0"))
    review = panel.review(claim, "some passage")
    assert review.decision == REJECT
