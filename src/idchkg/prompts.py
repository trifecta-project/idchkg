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
graph (after van Erp, "Unflattening Knowledge Graphs", K-CAP 2023). Instead of
flattening text into bare triples, you preserve three things current KGs lose:
the different DIMENSIONS of a concept (identity), how it CHANGES over time, and
its exact PROVENANCE.

From the passage you are given, extract:

1. entities - the concepts, people, places, organisations, and events the
   passage is actually about (not every noun). For each entity capture:
   - sense: the dimension the entity is used in, when a concept is
     multi-dimensional. The word "coffee" can name the PLANT, the BEVERAGE,
     the ACTIVITY of drinking it, or the COLONIAL COMMODITY - these are
     different identities, not one. Give a short lower-case sense label
     ("plant", "beverage", "activity", "commodity", "person", ...) or "" when
     the entity has only one obvious sense here.
   - context: the qualifying scope in which this identity holds, when the
     passage restricts it ("in the botanical sense", "as ruler of Spain
     1939-1975"), else "".

2. claims - subject/predicate/object statements made in the passage. For each
   claim also capture:
   - perspective: whose voice makes the claim. viewpoint "author" for the
     text's own voice, or the named source when attributed ("According to
     X..."). stance is "asserts", "disputes", or "questions". sentiment is the
     connotation the claim carries here.
   - time: when the claim holds. Keep the passage's own expression in "when"
     (e.g. "in the 17th century") and normalise to years/ISO dates in
     valid_from / valid_to when possible. null when the passage gives none.
     A claim that only holds during a period is exactly the change-over-time
     the graph must capture.
   - context: the scope in which the claim holds (a sense, role, or period),
     when the passage restricts it, else "".
   - quote: the shortest verbatim span of the passage that supports the claim.
     It must appear character-for-character in the passage.

Rules:
- Extract only what the passage states. Never add outside knowledge.
- Keep the different senses of a concept apart. If the passage speaks of
  coffee the plant and coffee the drink, use the entity name "coffee" with
  distinct senses - do not collapse them.
- Different viewpoints on the same matter become separate claims, each with
  its own viewpoint - do not merge or reconcile them.
- Use short noun phrases for subject and object; a concise lower-case verb
  phrase for the predicate. Subjects and objects should appear in entities.
- subject_sense / object_sense: when the subject or object is a
  multi-dimensional concept, give the sense it is used in here (matching the
  entity's sense, e.g. "plant" or "beverage"), else "".
- For sources that come from Project Gutenberg or the Internet archive, ignore 
  the preamble that states the source. 
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
                "required": ["name", "type", "sense", "description", "context"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "concept", "person", "organization",
                            "place", "event", "work", "other",
                        ],
                    },
                    "sense": {"type": "string"},
                    "description": {"type": "string"},
                    "context": {"type": "string"},
                },
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject", "subject_sense", "predicate", "object",
                    "object_sense", "perspective", "time", "context", "quote",
                ],
                "properties": {
                    "subject": {"type": "string"},
                    "subject_sense": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_sense": {"type": "string"},
                    "context": {"type": "string"},
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
QUALIFIER FIDELITY - are the claim's perspective, time, and context qualifiers
faithful to the passage?

Check:
- viewpoint: if the passage attributes the statement to a named source, the
  claim must use that source, not "author" - and vice versa.
- stance: "disputes"/"questions" only when the passage frames it that way.
- sentiment: matches the connotation the passage gives the claim.
- time: matches the temporal scope the passage states; must be null when the
  passage gives none; normalised dates must be consistent with "when".
- context: if the passage restricts the claim to a sense, role, or period
  (e.g. coffee "the plant" vs "the drink"), the context must reflect that and
  must not conflate distinct senses; empty only when the passage gives none.
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
