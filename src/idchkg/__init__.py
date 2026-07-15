"""idchkg - a local, LLM-based knowledge graph pipeline with built-in judges,
inspired by "Unflattening Knowledge Graphs" (Marieke van Erp, K-CAP 2023)."""

from .graph import KnowledgeGraph
from .models import Claim, Entity, Perspective, Provenance, TemporalScope
from .pipeline import Pipeline, PipelineConfig, PipelineResult

__all__ = [
    "Claim",
    "Entity",
    "KnowledgeGraph",
    "Perspective",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "Provenance",
    "TemporalScope",
]

__version__ = "0.1.0"
