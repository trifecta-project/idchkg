"""Pipeline orchestration.

    ingest -> chunk -> extract (LLM) -> judge (LLM panel) -> repair (LLM,
    one round, judge-guided) -> resolve entities -> load in-memory graph

Only claims the judges accept (directly, or after one repair round) enter the
graph; everything else is kept in the run report with the judges' reasoning.

Chunks are processed concurrently (`PipelineConfig.concurrency`): each chunk's
LLM work — extraction plus judging/repair of its claims — runs in a worker
thread, but every mutation of the shared resolver and graph happens on the main
thread in document/chunk order, so the resulting graph is deterministic
regardless of concurrency.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .extract import Extractor, chunk_document
from .graph import KnowledgeGraph
from .gutenberg import strip_gutenberg_boilerplate
from .judge import ACCEPT, REVISE, JudgePanel
from .llm import LLMClient, LLMError, get_client
from .models import ACCEPTED, REJECTED, REPAIRED, Chunk, Claim, Document, EntityMention, slugify
from .resolve import EntityResolver

logger = logging.getLogger(__name__)


class _Timing:
    """Thread-safe accumulator of per-phase LLM wall time + call counts, so the
    run summary can show whether extraction or judging dominates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.extract_s = self.judge_s = self.repair_s = 0.0
        self.extract_n = self.judge_n = self.repair_n = 0

    def add(self, phase: str, seconds: float) -> None:
        with self._lock:
            setattr(self, f"{phase}_s", getattr(self, f"{phase}_s") + seconds)
            setattr(self, f"{phase}_n", getattr(self, f"{phase}_n") + 1)


def _enable_verbose_logging() -> None:
    """Make `Pipeline.run(verbose=True)` show logs even when the host app hasn't
    configured logging (e.g. a notebook). No-op if handlers already exist (the
    CLI configures logging itself via basicConfig)."""
    package = logging.getLogger("idchkg")
    package.setLevel(logging.DEBUG)
    if not logging.getLogger().handlers and not package.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        package.addHandler(handler)


@dataclass
class PipelineConfig:
    backend: str = "anthropic"          # anthropic | ollama | mock
    model: str | None = None            # backend default when None
    judge_backend: str | None = None    # defaults to `backend`
    judge_model: str | None = None      # defaults to `model`
    judge_threshold: float = 0.7
    max_repair_rounds: int = 1
    chunk_chars: int = 1600
    concurrency: int = 4                # worker threads for LLM calls across chunks
    aliases: dict[str, str] = field(default_factory=dict)
    # DBpedia entity linking (opt-in; makes network calls to a Spotlight server).
    link_dbpedia: bool = False
    spotlight_url: str | None = None    # None -> DBpediaLinker default (public API)
    spotlight_confidence: float = 0.5
    spotlight_concurrency: int = 4


@dataclass
class PipelineResult:
    graph: KnowledgeGraph
    accepted: list[Claim] = field(default_factory=list)
    repaired: list[Claim] = field(default_factory=list)
    rejected: list[Claim] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "documents": self.graph.document_count(),
            "entities": self.graph.entity_count(),
            "claims_accepted": len(self.accepted),
            "claims_repaired": len(self.repaired),
            "claims_rejected": len(self.rejected),
            "extraction_errors": len(self.errors),
        }


@dataclass
class _ClaimOutcome:
    """Result of judging (and maybe repairing) one claim — computed off-thread,
    applied to the graph on the main thread. Carries no graph/resolver state."""

    claim: Claim | None = None
    accepted: bool = False
    was_repaired: bool = False
    mentions: list[EntityMention] = field(default_factory=list)  # from repair re-extraction
    error: str | None = None


@dataclass
class _ChunkWork:
    """All LLM work for one chunk (extraction mentions + per-claim outcomes)."""

    mentions: list[EntityMention] = field(default_factory=list)
    outcomes: list[_ClaimOutcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        extractor_client: LLMClient | None = None,
        judge_client: LLMClient | None = None,
        linker=None,
    ):
        self.config = config or PipelineConfig()
        # DBpedia linker: injected (tests pass a stub), or built on demand when
        # link_dbpedia is set. None means the linking step is skipped.
        self.linker = linker
        if self.linker is None and self.config.link_dbpedia:
            from .link import DBpediaLinker

            kwargs = {"confidence": self.config.spotlight_confidence}
            if self.config.spotlight_url:
                kwargs["spotlight_url"] = self.config.spotlight_url
            self.linker = DBpediaLinker(**kwargs)
        self._extractor_client = extractor_client or get_client(
            self.config.backend, self.config.model
        )
        self._judge_client = judge_client or (
            self._extractor_client
            if self.config.judge_backend is None and self.config.judge_model is None
            else get_client(
                self.config.judge_backend or self.config.backend,
                self.config.judge_model or self.config.model,
            )
        )
        self.extractor = Extractor(self._extractor_client)
        self.panel = JudgePanel(self._judge_client, threshold=self.config.judge_threshold)

    # ------------------------------------------------------------------ input

    @staticmethod
    def load_documents(paths: list[str | Path]) -> list[Document]:
        documents = []
        seen_ids: set[str] = set()
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                # Recurse so nested corpora (e.g. corpora/food, corpora/maritime)
                # are all picked up. Skip README/index docs - they describe the
                # corpus, they aren't part of it.
                files = [
                    f for f in sorted(path.rglob("*.txt")) + sorted(path.rglob("*.md"))
                    if f.stem.casefold() not in {"readme", "index"}
                ]
            else:
                files = [path]
            for file in files:
                # Derive a stable id from the path so same-named files in
                # different subfolders don't collide.
                rel = file.stem if not path.is_dir() else "-".join(file.relative_to(path).with_suffix("").parts)
                doc_id = slugify(rel)
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                # Strip Project Gutenberg header/footer boilerplate so we don't
                # extract and judge license text (a no-op on non-Gutenberg files).
                text = strip_gutenberg_boilerplate(file.read_text(encoding="utf-8"))
                documents.append(
                    Document(
                        id=doc_id,
                        text=text,
                        title=file.stem.replace("_", " "),
                        metadata={"path": str(file)},
                    )
                )
        return documents

    # -------------------------------------------------------------------- run

    def run(self, documents: list[Document], verbose: bool = False, graph=None) -> PipelineResult:
        # `graph` is a build sink: an in-memory KnowledgeGraph (default) or a
        # streaming GraphStore (store.py) that writes to SQLite as it goes, so a
        # large corpus need not be held in RAM. Both share the build API.
        if verbose:
            _enable_verbose_logging()
        graph = graph if graph is not None else KnowledgeGraph()
        resolver = EntityResolver(aliases=self.config.aliases)
        result = PipelineResult(graph=graph)
        self._timing = _Timing()  # per-run; read by the worker methods below

        chunks: list[Chunk] = []
        for document in documents:
            graph.add_document(document)
            chunks.extend(chunk_document(document, max_chars=self.config.chunk_chars))

        started = time.perf_counter()
        logger.info(
            "building graph: %d document(s), %d chunk(s), backend=%s, concurrency=%d",
            len(documents), len(chunks), self.config.backend, self.config.concurrency,
        )

        # LLM work per chunk runs concurrently; results are applied in order.
        for chunk, work in zip(chunks, self._map(self._process_chunk, chunks)):
            result.errors.extend(work.errors)
            for mention in work.mentions:
                resolver.add_mention(mention)
            for outcome in work.outcomes:
                self._apply_outcome(outcome, graph, resolver, result)

        # Optionally link entities to DBpedia (context-aware) before they are
        # finalised, so the populated `dbpedia` field is persisted by both the
        # in-memory graph and the streaming SQLite store in one write.
        if self.linker is not None:
            started_link = time.perf_counter()
            n = self.linker.link_entities(resolver.entities, concurrency=self.config.spotlight_concurrency)
            logger.info("DBpedia linking: %d entities linked in %.1fs", n, time.perf_counter() - started_link)

        # Register contextual / near-identity links between senses (Challenge 1).
        for link in resolver.identity_links:
            graph.add_identity_link(link)
        graph.finalize_entities(resolver.entities)

        self._log_summary(time.perf_counter() - started, result)
        return result

    def _log_summary(self, wall: float, result: PipelineResult) -> None:
        t = self._timing
        # Per-phase times are cumulative across worker threads, so their sum can
        # exceed the wall time when concurrency>1 — the ratio still shows which
        # phase dominates (usually judging: 2 model calls per claim vs 1 extract).
        logger.info(
            "run complete in %.1fs — extraction: %d calls / %.1fs (%.1fs avg); "
            "judging: %d reviews / %.1fs (%.1fs avg, <=2 model calls each); repairs: %d / %.1fs",
            wall,
            t.extract_n, t.extract_s, t.extract_s / max(t.extract_n, 1),
            t.judge_n, t.judge_s, t.judge_s / max(t.judge_n, 1),
            t.repair_n, t.repair_s,
        )
        logger.info(
            "accepted=%d repaired=%d rejected=%d errors=%d entities=%d",
            len(result.accepted), len(result.repaired), len(result.rejected),
            len(result.errors), result.graph.entity_count(),
        )

    # --------------------------------------------------------------- internals

    def _map(self, fn, items: list):
        """Map `fn` over `items`, concurrently when configured. Results keep
        input order; run sequentially for a single item or concurrency<=1 so
        LLM-call order stays deterministic (the scripted mock relies on it)."""
        if self.config.concurrency <= 1 or len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            return list(executor.map(fn, items))

    def _process_chunk(self, chunk: Chunk) -> _ChunkWork:
        """All LLM work for one chunk — runs in a worker thread, touches no
        shared state (only the per-chunk objects it creates)."""
        started = time.perf_counter()
        try:
            extraction = self.extractor.extract(chunk)
        except LLMError as exc:
            logger.warning("chunk %s: extraction failed: %s", chunk.id, exc)
            return _ChunkWork(errors=[f"{chunk.id}: extraction failed: {exc}"])
        elapsed = time.perf_counter() - started
        self._timing.add("extract", elapsed)
        logger.info("chunk %s: %d claim(s) extracted in %.1fs", chunk.id, len(extraction.claims), elapsed)
        work = _ChunkWork(
            mentions=list(extraction.entities),
            errors=[f"{chunk.id}: {e}" for e in extraction.errors],
        )
        for claim in extraction.claims:
            work.outcomes.append(self._review_claim(claim, chunk))
        return work

    def _review_claim(self, claim: Claim, chunk: Chunk) -> _ClaimOutcome:
        """Judge one claim, with up to one judge-guided repair round. Pure: no
        graph/resolver mutation — returns an outcome to apply on the main thread."""
        repairs_left = self.config.max_repair_rounds
        current = claim
        was_repaired = False
        mentions: list[EntityMention] = []

        while True:
            started = time.perf_counter()
            try:
                review = self.panel.review(current, chunk.text)
            except LLMError as exc:
                logger.warning("%s: judging failed: %s", current.id, exc)
                return _ClaimOutcome(error=f"{current.id}: judging failed: {exc}")
            self._timing.add("judge", time.perf_counter() - started)
            if current.provenance:
                current.provenance.verdicts.extend(review.verdicts)

            if review.decision == ACCEPT:
                return _ClaimOutcome(claim=current, accepted=True, was_repaired=was_repaired, mentions=mentions)

            if review.decision == REVISE and repairs_left > 0:
                repairs_left -= 1
                started = time.perf_counter()
                try:
                    extraction = self.extractor.repair(chunk, current, review.feedback)
                except LLMError as exc:
                    logger.warning("%s: repair failed: %s", current.id, exc)
                    return _ClaimOutcome(error=f"{current.id}: repair failed: {exc}")
                self._timing.add("repair", time.perf_counter() - started)
                if extraction and extraction.claims:
                    mentions.extend(extraction.entities)
                    repaired = extraction.claims[0]
                    if repaired.provenance and current.provenance:
                        # Keep the original verdict history, and preserve the
                        # supporting quote when the repair round drops it (repair
                        # fixes qualifiers, not the evidence) - otherwise the
                        # claim would enter the graph with no provenance quote.
                        repaired.provenance.verdicts = list(current.provenance.verdicts)
                        if not repaired.provenance.quote and current.provenance.quote:
                            repaired.provenance.quote = current.provenance.quote
                    current = repaired
                    was_repaired = True
                    continue
                # repair produced nothing: fall through to rejection

            return _ClaimOutcome(claim=current, accepted=False, was_repaired=was_repaired, mentions=mentions)

    def _apply_outcome(self, outcome, graph, resolver, result) -> None:
        """Apply a claim outcome to the shared resolver + graph (main thread)."""
        if outcome.error:
            result.errors.append(outcome.error)
            return
        for mention in outcome.mentions:  # entities from a repair re-extraction
            resolver.add_mention(mention)
        claim = outcome.claim
        if outcome.accepted:
            claim.status = REPAIRED if outcome.was_repaired else ACCEPTED
            subject = resolver.resolve(claim.subject, claim.subject_sense)
            obj = resolver.resolve(claim.object, claim.object_sense)
            graph.add_claim(claim, subject, obj)
            (result.repaired if outcome.was_repaired else result.accepted).append(claim)
        else:
            claim.status = REJECTED
            result.rejected.append(claim)
        logger.debug("[%s] %s --%s--> %s", claim.status, claim.subject, claim.predicate, claim.object)
