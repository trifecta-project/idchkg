# Representative corpora — food history & maritime history

These two domains are a natural fit for "unflattening": the same commodities
(coffee, sugar, tea, spices) are simultaneously plants, drinks, and colonial
trade goods (**identity**), their meaning and value shift over centuries
(**change**), and they are described very differently by contemporary merchants
versus modern historians (**perspective**) — all of which the pipeline is built
to surface, with **provenance** on every claim.

This folder has two parts:

1. **`food/` and `maritime/`** — a small, ready-to-run sample corpus bundled in
   the repo (see *Provenance* below).
2. **This catalogue** — pointers to the real, larger, properly-licensed corpora
   and datasets you can scale to.

## Run the bundled sample

```bash
# Recurses into food/ and maritime/
idchkg build examples/corpora --backend ollama --model llama3.1 -o out

idchkg query out/graph.json sugar   --view identity
idchkg query out/graph.json spices  --view change
idchkg query out/graph.json         --view longtail
idchkg query out/graph.json         --view conflicts
```

The sample docs are deliberately interlinked (sugar ↔ Atlantic trade, spices ↔
VOC ↔ spice route, tea ↔ East India Company), so a single build exercises
cross-document entity resolution, multiple senses of one word, opposing
viewpoints, and temporal drift.

### Provenance of the bundled sample

> The files in `food/` and `maritime/` are **concise factual summaries written
> for this repository as test fixtures**, not verbatim primary sources. They
> state well-established history in neutral prose so the pipeline has realistic,
> multi-dimensional material to extract. For research use, ingest the real
> corpora below instead — and note that the whole point of the pipeline's
> provenance tracking is that it records exactly which source each claim came
> from, so mixing curated and primary text stays auditable.

## Real corpora — food history

| Corpus / source | What it is | License / access | Fits |
| --- | --- | --- | --- |
| [Project Gutenberg — "Books about Food · History"](https://www.gutenberg.org/ebooks/subject/25611) | Public-domain food-history books & primary texts | Public domain | full text |
| [*All About Coffee*, W. H. Ukers (1922)](https://www.gutenberg.org/ebooks/28500) | Encyclopedic history of coffee — origins, trade, colonialism | Public domain | identity, change, provenance |
| [*The Book of Tea*, Okakura (1906)](https://www.gutenberg.org/ebooks/769) | Tea as ritual/philosophy — a strong "perspective" source | Public domain | perspective, identity |
| [*Coffee and Chicory*, P. L. Simmonds](https://www.gutenberg.org/ebooks/57340) | Production, adulteration, botany vs. commodity | Public domain | identity |
| [*Tea Drinking in 18th-Century America*, Roth](https://www.gutenberg.org/ebooks/46775) | Etiquette/equipage; tea's arrival & contestation | Public domain | change, perspective |
| [Feeding America: Historic American Cookbook Project (MSU)](https://digital.lib.msu.edu/projects/cookbooks/) | Digitized historical American cookbooks | Public domain / educational | change over time |
| [Biodiversity Heritage Library](https://www.biodiversitylibrary.org/) | Botanical/agricultural texts on crop plants | Mostly public domain | identity (plant sense) |
| [Wikipedia — History of food](https://en.wikipedia.org/wiki/Category:History_of_food) (Columbian exchange, history of coffee/sugar/tea, spice trade) | Encyclopedic, cross-linked, current | CC BY-SA (attribute) | all four |

## Real corpora — maritime history

| Corpus / source | What it is | License / access | Fits |
| --- | --- | --- | --- |
| [SlaveVoyages](https://www.slavevoyages.org/) | Trans-Atlantic & Intra-American slave-trade voyage databases | Open (cite) | provenance, change; structured data |
| [CLIWOC](https://webs.ucm.es/info/cliwoc/) | Climatological DB of the World's Oceans 1750–1854, from ~280k ship-logbook days | Open research data | provenance, change; ship logs |
| [Old Weather](https://www.oldweather.org/) | Crowd-transcribed Royal Navy / Arctic ship logbooks | Open (Zooniverse) | provenance; primary logs |
| [ICOADS](https://icoads.noaa.gov/) | Comprehensive Ocean–Atmosphere Data Set — historical ship observations | Public (NOAA) | provenance; structured obs |
| [*The Influence of Sea Power upon History*, Mahan (1890)](https://www.gutenberg.org/ebooks/13529) | Foundational (and opinionated) naval-history text | Public domain | perspective |
| [*Two Years Before the Mast*, Dana (1840)](https://www.gutenberg.org/ebooks/2055) | First-hand sailor's narrative | Public domain | perspective, provenance |
| [Hakluyt, *Principal Navigations*](https://www.gutenberg.org/ebooks/author/1109) | Early-modern voyage & trade narratives | Public domain | identity, change |
| [Wikipedia — Maritime history](https://en.wikipedia.org/wiki/Category:Maritime_history) (VOC, Age of Sail, triangular trade, Hanseatic League, spice trade) | Encyclopedic, cross-linked | CC BY-SA (attribute) | all four |

## Fetching (run in an unrestricted environment)

Network egress is locked down in some sandboxes (Gutenberg, Wikipedia, and the
dataset hosts above may be blocked). On a machine with open access, the helper
downloads a curated set of public-domain plain-text books into
`food/raw/` and `maritime/raw/`:

```bash
python examples/corpora/fetch.py           # downloads the curated Gutenberg set
idchkg build examples/corpora/food/raw --backend ollama -o out-food
```

For the structured datasets (SlaveVoyages, CLIWOC, ICOADS) you'll want a small
adapter that renders rows/voyages into short natural-language documents before
ingesting — those are tabular, and the pipeline consumes text.

## Licensing note

Public-domain texts can be redistributed freely. Wikipedia and some datasets are
CC BY-SA / require citation — attribute them if you redistribute derived graphs.
The bundled `food/` and `maritime/` samples are original prose and carry no such
restriction.
