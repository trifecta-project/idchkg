"""Optional visualization: render a KnowledgeGraph as an interactive
yFiles Graphs for Jupyter widget.

The "unflattened" graph is a networkx MultiDiGraph whose nodes are of four
kinds (entity | claim | perspective | document) and whose edges carry a `kind`
(subject | object | perspective | source | near_identity | evolves). This
module maps those onto yFiles node/edge styling so the structure is legible:

* entities are sized by mention count (Challenge 3: head vs long tail),
* claim nodes are coloured by stance (asserts / disputes / questions),
* each node kind gets its own colour, and every edge is labelled by its kind.

Requires the `viz` extra:  pip install -e '.[viz]'
"""

from __future__ import annotations

from .graph import KnowledgeGraph

# One colour per node kind.
_KIND_COLORS = {
    "entity": "#2b7a9b",       # teal
    "claim": "#b0b7bd",        # grey (overridden by stance below)
    "perspective": "#8e6fb0",  # purple
    "document": "#9aa0a6",     # slate
}

# Claim nodes are coloured by the stance of their perspective.
_STANCE_COLORS = {
    "asserts": "#4c9a52",    # green
    "disputes": "#c0504d",   # red
    "questions": "#e0a03c",  # amber
}


def _props(element: dict) -> dict:
    return element.get("properties", {})


def _node_label(node: dict) -> str:
    p = _props(node)
    kind = p.get("kind")
    if kind == "entity":
        sense = p.get("sense")
        return f"{p.get('label', '?')}\n({sense})" if sense else p.get("label", "?")
    if kind == "claim":
        return p.get("predicate", "claim")
    if kind == "perspective":
        return p.get("viewpoint", "perspective")
    if kind == "document":
        return p.get("title", "document")
    return str(node.get("id"))


def _node_color(node: dict) -> str:
    p = _props(node)
    if p.get("kind") == "claim":
        return _STANCE_COLORS.get(p.get("stance"), _KIND_COLORS["claim"])
    return _KIND_COLORS.get(p.get("kind"), "#cccccc")


def _node_scale(node: dict) -> float:
    """Scale entity nodes by mention count so head entities stand out from the
    long tail (Challenge 3). Non-entity nodes keep the default size."""
    p = _props(node)
    if p.get("kind") == "entity":
        count = int(p.get("mention_count", 0) or 0)
        return 1.0 + 0.15 * min(count, 20)
    return 1.0


def _edge_label(edge: dict) -> str:
    return _props(edge).get("kind", "")


def entity_subgraph(kg: KnowledgeGraph, label: str):
    """A focused networkx subgraph around one concept: every sense of ``label``,
    the claims that mention them, and each claim's other endpoint, perspective,
    and source document. Handy for large graphs where the full view is too busy.

    Pass the result straight to ``to_yfiles_widget(...)``.
    """
    graph = kg.g
    keep: set = set()
    for entity in kg.senses_of(label):
        keep.add(entity.id)
        for claim in kg.claims_about(entity):
            keep.add(claim.id)
    # Expand each claim to its immediate neighbours (subject/object/persp/doc).
    for node in list(keep):
        if node in graph:
            keep.update(graph.successors(node))
    return graph.subgraph(keep).copy()


def to_yfiles_widget(kg, *, layout: str = "organic", directed: bool = True):
    """Build a yFiles ``GraphWidget`` for a ``KnowledgeGraph`` (or a raw
    networkx graph). Return it as the last expression in a notebook cell to
    render it.

    ``layout`` is any yFiles layout name: organic | hierarchic | circular |
    orthogonal | radial | tree.
    """
    try:
        from yfiles_jupyter_graphs import GraphWidget
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yFiles Graphs for Jupyter is not installed. "
            "Run: pip install -e '.[viz]'  (or: pip install yfiles_jupyter_graphs)"
        ) from exc

    graph = kg.g if isinstance(kg, KnowledgeGraph) else kg

    widget = GraphWidget()
    widget.import_graph(graph)
    widget.directed = directed
    widget.set_node_label_mapping(_node_label)
    widget.set_node_color_mapping(_node_color)
    widget.set_node_scale_factor_mapping(_node_scale)
    widget.set_edge_label_mapping(_edge_label)
    widget.set_graph_layout(layout)
    return widget
