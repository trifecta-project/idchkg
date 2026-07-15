"""Pipeline orchestration.

    ingest -> chunk -> extract (LLM) -> judge (LLM panel) -> repair (LLM,
    one round, judge-guided) -> resolve entities -> load in-memory graph

Only claims the judges accept (directly, or after one repair round) enter the
graph; everything else is kept in the run report with the judges' reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .extract import Extractor, chunk_document
from .graph import KnowledgeGraph
from .judge import ACCEPT, REVISE, JudgePanel
from .llm import LLMClient, LLMError, get_client
from .models import ACCEPTED, REJECTED, REPAIRED, Chunk, Claim, Document, slugify
from .resolve import EntityResolver


@dataclass
class PipelineConfig:
    backend: str = "anthropic"          # anthropic | ollama | mock
    model: str | None = None            # backend default when None
    judge_backend: str | None = None    # defaults to `backend`
    judge_model: str | None = None      # defaults to `model`
    judge_threshold: float = 0.7
    max_repair_rounds: int = 1
    chunk_chars: int = 1600
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    graph: KnowledgeGraph
    accepted: list[Claim] = field(default_factory=list)
    repaired: list[Claim] = field(default_factory=list)
    rejected: list[Claim] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "documents": len(self.graph.to_dict()["documents"]),
            "entities": len(self.graph.entities),
            "claims_accepted": len(self.accepted),
            "claims_repaired": len(self.repaired),
            "claims_rejected": len(self.rejected),
            "extraction_errors": len(self.errors),
        }


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        extractor_client: LLMClient | None = None,
        judge_client: LLMClient | None = None,
    ):
        self.config = config or PipelineConfig()
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
        for raw in paths:
            path = Path(raw)
            files = sorted(path.glob("*.txt")) + sorted(path.glob("*.md")) if path.is_dir() else [path]
            for file in files:
                documents.append(
                    Document(
                        id=slugify(file.stem),
                        text=file.read_text(encoding="utf-8"),
                        title=file.stem.replace("_", " "),
                        metadata={"path": str(file)},
                    )
                )
        return documents

    # -------------------------------------------------------------------- run

    def run(self, documents: list[Document], verbose: bool = False) -> PipelineResult:
        graph = KnowledgeGraph()
        resolver = EntityResolver(aliases=self.config.aliases)
        result = PipelineResult(graph=graph)

        for document in documents:
            graph.add_document(document)
            for chunk in chunk_document(document, max_chars=self.config.chunk_chars):
                if verbose:
                    print(f"[extract] {chunk.id}")
                try:
                    extraction = self.extractor.extract(chunk)
                except LLMError as exc:
                    result.errors.append(f"{chunk.id}: extraction failed: {exc}")
                    continue
                result.errors.extend(f"{chunk.id}: {e}" for e in extraction.errors)
                for mention in extraction.entities:
                    resolver.add_mention(mention)
                for claim in extraction.claims:
                    self._judge_and_load(claim, chunk, graph, resolver, result, verbose)
        return result

    def _judge_and_load(
        self,
        claim: Claim,
        chunk: Chunk,
        graph: KnowledgeGraph,
        resolver: EntityResolver,
        result: PipelineResult,
        verbose: bool,
    ) -> None:
        repairs_left = self.config.max_repair_rounds
        current = claim
        was_repaired = False

        while True:
            try:
                review = self.panel.review(current, chunk.text)
            except LLMError as exc:
                result.errors.append(f"{current.id}: judging failed: {exc}")
                return
            if current.provenance:
                current.provenance.verdicts.extend(review.verdicts)

            if review.decision == ACCEPT:
                current.status = REPAIRED if was_repaired else ACCEPTED
                subject = resolver.resolve(current.subject)
                obj = resolver.resolve(current.object)
                graph.add_claim(current, subject, obj)
                (result.repaired if was_repaired else result.accepted).append(current)
                if verbose:
                    print(f"  [{current.status}] {current.subject} --{current.predicate}--> {current.object}")
                return

            if review.decision == REVISE and repairs_left > 0:
                repairs_left -= 1
                try:
                    extraction = self.extractor.repair(chunk, current, review.feedback)
                except LLMError as exc:
                    result.errors.append(f"{current.id}: repair failed: {exc}")
                    extraction = None
                if extraction and extraction.claims:
                    for mention in extraction.entities:
                        resolver.add_mention(mention)
                    # keep the original verdict history on the repaired claim
                    repaired = extraction.claims[0]
                    if repaired.provenance and current.provenance:
                        repaired.provenance.verdicts = list(current.provenance.verdicts)
                    current = repaired
                    was_repaired = True
                    if verbose:
                        print(f"  [repairing] {claim.id}: {review.feedback[:100]}")
                    continue
                # repair produced nothing: fall through to rejection

            current.status = REJECTED
            result.rejected.append(current)
            if verbose:
                print(f"  [rejected] {current.subject} --{current.predicate}--> {current.object}")
            return
