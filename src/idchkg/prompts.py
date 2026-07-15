"""Prompts and JSON schemas for the extraction and judge stages.

Schemas follow the structured-outputs constraints (every object sets
additionalProperties: false and lists all properties as required; nullable
fields use type: ["string", "null"]).
"""

from __future__ import annotations

STANCES = ["asserts", "disputes", "questions"]
SENTIMENTS = ["positive", "negative", "neutral", "ambivalent"]

# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are an information-extraction system building an "unflattened" knowledge
graph: one that preserves multiple perspectives, change over time, and exact
provenance, instead of flattening text into bare triples.

From the passage you are given, extract:

1. entities - the concepts, people, places, organisations, and events the
   passage is actually about (not every noun).
2. claims - subject/predicate/object statements made in the passage. For each
   claim also capture:
   - perspective: whose voice makes the claim. Use viewpoint "author" for the
     text's own voice, or the named source when the text attributes the claim
     to someone ("According to X...", "The 1991 report stated..."). stance is
     "asserts" (put forward as true), "disputes" (argued against), or
     "questions" (raised as uncertain). sentiment is the connotation the
     claim carries in this passage.
   - time: when the claim holds, if the passage says so. Keep the passage's
     own expression in "when" (e.g. "in the 17th century") and, when
     possible, normalise to years/ISO dates in valid_from / valid_to.
     Use null when the passage gives no temporal scope.
   - quote: the shortest verbatim span of the passage that supports the
     claim. It must appear character-for-character in the passage.

Rules:
- Extract only what the passage states. Never add outside knowledge.
- Different viewpoints on the same matter must become separate claims, each
  attributed to its own viewpoint - do not merge or reconcile them.
- Use short noun phrases for subject and object; use a concise verb phrase in
  lower case for the predicate (e.g. "originates from", "was classified as").
- Subjects and objects of claims should appear in the entities list.
"""

EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "claims"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "type", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "concept", "person", "organization",
                            "place", "event", "work", "other",
                        ],
                    },
                    "description": {"type": "string"},
                },
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "predicate", "object", "perspective", "time", "quote"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "perspective": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["viewpoint", "stance", "sentiment"],
                        "properties": {
                            "viewpoint": {"type": "string"},
                            "stance": {"type": "string", "enum": STANCES},
                            "sentiment": {"type": "string", "enum": SENTIMENTS},
                        },
                    },
                    "time": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["when", "valid_from", "valid_to"],
                        "properties": {
                            "when": {"type": ["string", "null"]},
                            "valid_from": {"type": ["string", "null"]},
                            "valid_to": {"type": ["string", "null"]},
                        },
                    },
                    "quote": {"type": "string"},
                },
            },
        },
    },
}


def extraction_user(chunk_text: str) -> str:
    return f"Extract entities and claims from this passage:\n\n<passage>\n{chunk_text}\n</passage>"


def repair_user(chunk_text: str, claim_json: str, feedback: str) -> str:
    return (
        "A previously extracted claim failed review. Re-extract a corrected "
        "version of this one claim (and its entities) from the passage, "
        "fixing the problems the reviewer describes. If the claim is simply "
        "not supported by the passage, return empty entities and claims.\n\n"
        f"<passage>\n{chunk_text}\n</passage>\n\n"
        f"<claim>\n{claim_json}\n</claim>\n\n"
        f"<reviewer_feedback>\n{feedback}\n</reviewer_feedback>"
    )


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------

VERDICT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "passed", "feedback"],
    "properties": {
        "score": {"type": "number"},
        "passed": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
}

GROUNDEDNESS_SYSTEM = """\
You are a strict reviewer of knowledge-graph claims. Judge one criterion:
GROUNDEDNESS - is the claim (subject, predicate, object) actually stated by
the passage, without added outside knowledge or over-generalisation?

Scoring: 1.0 = directly stated; 0.7 = stated with minor rewording; below 0.5 =
not supported, contradicted, or requires outside knowledge. Set passed=true
only if score >= 0.7. In feedback, explain in one or two sentences what is
wrong (or state that it is fully supported).
"""

QUALIFIER_SYSTEM = """\
You are a strict reviewer of knowledge-graph claims. Judge one criterion:
QUALIFIER FIDELITY - are the claim's perspective and time qualifiers faithful
to the passage?

Check:
- viewpoint: if the passage attributes the statement to a named source, the
  claim must use that source, not "author" - and vice versa.
- stance: "disputes"/"questions" only when the passage frames it that way.
- sentiment: matches the connotation the passage gives the claim.
- time: matches the temporal scope the passage states; must be null when the
  passage gives none; normalised dates must be consistent with "when".
- quote: must appear verbatim in the passage and support the claim.

Scoring: 1.0 = all qualifiers faithful; deduct for each unfaithful qualifier;
below 0.7 = fails. Set passed=true only if score >= 0.7. In feedback, name the
specific qualifier(s) that are wrong and what they should be.
"""


def judge_user(chunk_text: str, claim_json: str) -> str:
    return (
        f"<passage>\n{chunk_text}\n</passage>\n\n"
        f"<claim>\n{claim_json}\n</claim>\n\n"
        "Review the claim against the passage and return your verdict."
    )
