from pathlib import Path

from idchkg.pipeline import Pipeline


def test_load_documents_recurses_and_excludes_readme(tmp_path: Path):
    (tmp_path / "food").mkdir()
    (tmp_path / "maritime").mkdir()
    (tmp_path / "food" / "sugar.txt").write_text("Sugar is sweet.")
    (tmp_path / "maritime" / "voc.txt").write_text("The VOC traded spices.")
    (tmp_path / "README.md").write_text("This describes the corpus, not part of it.")

    docs = Pipeline.load_documents([tmp_path])
    ids = sorted(d.id for d in docs)
    assert ids == ["food-sugar", "maritime-voc"]  # README excluded, nested picked up


def test_load_documents_ids_are_unique_across_subfolders(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.txt").write_text("one")
    (tmp_path / "b" / "notes.txt").write_text("two")
    docs = Pipeline.load_documents([tmp_path])
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)) == 2  # same filename, distinct ids


def test_load_single_file(tmp_path: Path):
    f = tmp_path / "one.txt"
    f.write_text("hello")
    docs = Pipeline.load_documents([f])
    assert len(docs) == 1 and docs[0].id == "one"


def test_bundled_corpora_are_loadable():
    docs = Pipeline.load_documents(["examples/corpora"])
    ids = {d.id for d in docs}
    assert "food-sugar" in ids and "maritime-voc" in ids
    assert all(d.text.strip() for d in docs)  # no empty corpus files
    assert "readme" not in ids
