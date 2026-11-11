import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertForTokenClassification, BertTokenizerFast, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from error_detection.steps import data as data_module  # noqa: E402
from error_detection.steps.data import (IGNORE, ResampledDataset, TokenCollator, build_training_rows, chunk_row, corrected,  # noqa: E402
                                        load_rows, sampling_weights, word_kinds)
from error_detection.steps.evaluation import (average_precision, best_threshold, build_lexicon, false_positive_sample,  # noqa: E402
                                              lexicon_scores, mlm_word_scores, predict_words, prf, summarise, token_metrics)
from error_detection.steps.model import WeightedTokenTrainer, build_model, weighted_token_loss  # noqa: E402
from error_detection.steps.vendor.normalize import normalize_word  # noqa: E402

LETTERS = list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")


@pytest.fixture()
def tokenizer(tmp_path):
    chars = LETTERS + ['"', "'", ".", ":", ","]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    return BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=False)


def tiny_config(tokenizer):
    return BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                      max_position_embeddings=128)


@pytest.fixture()
def model_dir(tmp_path, tokenizer):
    torch.manual_seed(0)
    path = tmp_path / "tiny"
    BertForTokenClassification(tiny_config(tokenizer)).save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return str(path)


def make_row(tokens, labels, fixes, row_id="r1", version="V", source="real"):
    row = {"id": row_id, "source": source, "ref": "Ref 1:1", "version": version, "tokens": tokens, "labels": labels, "fixes": fixes}
    row["kinds"] = word_kinds(row)
    return row


def test_normalize_word_strips_points_and_tags_and_splits_maqaf():
    assert normalize_word("כָּל־הַזְּבָחִים") == "כל הזבחים"
    assert normalize_word("<b>שָׁלוֹם</b>") == "שלום"
    assert normalize_word("&lt;span class=\"x\"&gt;") == ""
    assert normalize_word('א"ל') == 'א"ל'


def test_load_rows_maps_omission_label_to_ok_and_records_kinds(tmp_path, monkeypatch):
    row = {"id": "a", "source": "real", "ref": "R", "version": "V", "tokens": ["אב", "גד", "הו"], "labels": [1, 0, 2],
           "fixes": [{"i1": 0, "i2": 1, "before": ["אב"], "after": ["אג"], "kind": "typo", "labelled": True},
                     {"i1": 2, "i2": 2, "before": [], "after": ["זה"], "kind": "omission", "labelled": True}]}
    path = tmp_path / "real.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    monkeypatch.setattr(data_module, "dataset_file", lambda repo, filename: str(path))
    [loaded] = load_rows("real", "train", "unused")
    assert loaded["labels"] == [1, 0, 0]
    assert loaded["kinds"] == ["typo", None, None]


def test_corrected_applies_fixes_right_to_left_including_splits_and_joins():
    row = make_row(["אבגד", "הו", "ז", "חט"], [1, 1, 1, 0], [
        {"i1": 0, "i2": 1, "before": ["אבגד"], "after": ["אב", "גד"], "kind": "spacing", "labelled": True},
        {"i1": 1, "i2": 3, "before": ["הו", "ז"], "after": ["הוז"], "kind": "spacing", "labelled": True},
    ])
    fixed = corrected(row)
    assert fixed["tokens"] == ["אב", "גד", "הוז", "חט"]
    assert fixed["labels"] == [0, 0, 0, 0]
    assert fixed["source"] == "corrected" and fixed["id"] == "r1:corrected"


def test_corrected_clips_a_fix_that_runs_past_the_window():
    row = make_row(["אב", "גד"], [0, 1], [{"i1": 1, "i2": 5, "before": ["גד"], "after": ["גה"], "kind": "typo", "labelled": True}])
    assert corrected(row)["tokens"] == ["אב", "גה"]


def test_chunk_row_labels_first_piece_and_never_splits_a_word(tokenizer):
    tokens = ["<b>אבג</b>", "דָּה", "<br>", "וזח"] * 6
    labels = [0, 1, 0, 0] * 6
    row = make_row(tokens, labels, [])
    chunks = chunk_row(tokenizer, row, max_length=12)
    assert len(chunks) > 1
    seen = []
    for chunk in chunks:
        assert len(chunk["input_ids"]) <= 12
        assert chunk["input_ids"][0] == tokenizer.cls_token_id and chunk["input_ids"][-1] == tokenizer.sep_token_id
        firsts = [i for i, label in enumerate(chunk["labels"]) if label != IGNORE]
        assert len(firsts) == len(chunk["word_index"])
        for p, w in zip(firsts, chunk["word_index"]):
            assert chunk["labels"][p] == labels[w]
        seen += chunk["word_index"]
    # "<br>" normalizes to nothing and is dropped; every other word appears exactly once, in order.
    assert seen == [i for i, t in enumerate(tokens) if t != "<br>"]


def test_sampling_weights_by_source():
    chunks = [{"source": "real"}, {"source": "corrected"}, {"source": "synthetic"}, {"source": "synthetic"}]
    w = sampling_weights(chunks, synthetic_weight=0.5, negative_weight=1.0)
    assert np.allclose(w, np.array([1, 1, 0.5, 0.5]) / 3.0)
    assert np.allclose(sampling_weights(chunks, synthetic_weight=0.0)[2:], 0.0)


def test_build_training_rows_pairs_every_real_row():
    real = [make_row(["אב"], [1], [{"i1": 0, "i2": 1, "before": ["אב"], "after": ["אג"], "kind": "typo", "labelled": True}])]
    synthetic = [make_row(["גד"], [0], [], row_id="s", source="synthetic")]
    rows = build_training_rows(real, synthetic, paired_negatives=True)
    assert [r["source"] for r in rows] == ["real", "corrected", "synthetic"]
    assert len(build_training_rows(real, synthetic, paired_negatives=False)) == 2


def test_resampled_dataset_is_deterministic():
    chunks = [{"i": i} for i in range(5)]
    w = np.ones(5) / 5
    assert [d["i"] for d in ResampledDataset(chunks, 20, w, seed=3)] == [d["i"] for d in ResampledDataset(chunks, 20, w, seed=3)]


def test_collator_pads_with_ignore():
    batch = TokenCollator(0)([{"input_ids": [2, 5, 3], "labels": [IGNORE, 1, IGNORE]}, {"input_ids": [2, 3], "labels": [IGNORE, IGNORE]}])
    assert batch["input_ids"].tolist() == [[2, 5, 3], [2, 3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[IGNORE, 1, IGNORE], [IGNORE, IGNORE, IGNORE]]


def test_average_precision_and_threshold():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([1, 0, 1, 0])
    assert average_precision(scores, labels) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert best_threshold(scores, labels) == pytest.approx(0.7)
    assert prf(scores >= 0.7, labels)["recall"] == 1.0


def test_token_metrics_ignores_unlabelled_positions():
    logits = np.array([[[0.0, 5.0], [5.0, 0.0], [0.0, 9.0]]])
    labels = np.array([[1, 0, IGNORE]])
    m = token_metrics((logits, labels))
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["positive_rate"] == pytest.approx(0.5)


def test_weighted_loss_up_weights_errors():
    logits = torch.zeros(1, 2, 2)
    labels = torch.tensor([[1, 0]])
    flat = weighted_token_loss(logits, labels, 1.0)
    assert weighted_token_loss(logits, labels, 5.0) == pytest.approx(flat.item())  # equal per-token loss: weighting does not change the mean
    logits = torch.tensor([[[0.0, -2.0], [0.0, -2.0]]])
    assert weighted_token_loss(logits, labels, 5.0) > weighted_token_loss(logits, labels, 1.0)


def test_trainer_runs_and_keeps_best(tmp_path, tokenizer, model_dir):
    model, tok = build_model(model_dir)
    rows = [make_row(["אבג", "דה", "וז"], [0, 1, 0], [{"i1": 1, "i2": 2, "before": ["דה"], "after": ["דו"], "kind": "typo", "labelled": True}],
                     row_id=f"r{i}") for i in range(8)]
    chunks = [c for row in build_training_rows(rows, [], True) for c in chunk_row(tok, row, 32)]
    args = TrainingArguments(output_dir=str(tmp_path / "out"), per_device_train_batch_size=4, per_device_eval_batch_size=4, max_steps=4,
                             eval_strategy="steps", eval_steps=2, save_strategy="no", report_to="none", disable_tqdm=True, use_cpu=True)
    trainer = WeightedTokenTrainer(model=model, args=args, train_dataset=ResampledDataset(chunks, 16, None),
                                   eval_dataset=ResampledDataset(chunks, len(chunks)), data_collator=TokenCollator(tok.pad_token_id),
                                   compute_metrics=token_metrics, pos_weight=5.0)
    trainer.train()
    assert trainer.best_metric is not None and trainer.best_step in (2, 4)
    assert trainer.restore_best()


def test_predictions_and_baselines_share_word_order(tokenizer, model_dir):
    torch.manual_seed(0)
    model, tok = build_model(model_dir)
    row = make_row(["אבג", "דה", "<br>", "וזח"], [0, 1, 0, 0], [{"i1": 1, "i2": 2, "before": ["דה"], "after": ["דו"], "kind": "typo", "labelled": True}])
    rows = build_training_rows([row], [], True)
    chunks = [c for r in rows for c in chunk_row(tok, r, 32)]
    words = predict_words(model, tok, chunks, "cpu")
    assert [w["label"] for w in words] == [0, 1, 0, 0, 0, 0]
    assert words[1]["kind"] == "typo"

    lexicon = build_lexicon([corrected(row)], min_count=1)
    lex = lexicon_scores(chunks, {r["id"]: r for r in rows}, lexicon)
    assert lex == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # "דה" is not in the corrected text; "דו" is

    mlm = BertForMaskedLM(tiny_config(tok)).eval()
    assert len(mlm_word_scores(mlm, tok, chunks, "cpu")) == len(words)

    report = summarise(words, scores=lex)
    assert report["recall_by_kind"]["typo"]["recall"] == 1.0
    sample = false_positive_sample(words, {r["id"]: r for r in rows}, threshold=0.0, n=10)
    assert all(s["word"] in row["tokens"] or s["word"] in corrected(row)["tokens"] for s in sample)


def test_score_segments_aggregates_per_version_and_book(tmp_path, monkeypatch, model_dir):
    from error_detection.steps import scoring

    segments = [{"id": f"V{v}|Book {i}", "ref": f"Book {i}", "version": f"V{v}", "book": "Book", "license": "CC0",
                 "tokens": ["אבג", "דה", "<br>", "וזח", "טי"]} for v in (1, 2) for i in range(5)]
    path = tmp_path / "segments.jsonl"
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in segments) + "\n")
    monkeypatch.setattr(scoring, "dataset_file", lambda repo, filename: str(path))
    rows = scoring.load_segments("unused")
    assert rows[0]["labels"] == [0] * 5 and rows[0]["source"] == "sample"

    model, tok = build_model(model_dir)
    counts, top = scoring.score_segments(model, tok, rows, "cpu", 32, threshold=0.0, block=3)
    assert len(counts) == 10 and all(c[0] == 4 and c[1] == 4 for c in counts.values())  # "<br>" dropped; threshold 0 flags every word
    versions = scoring.aggregate(rows, counts, ("version",))
    assert [(r["version"], r["words"], r["flagged_per_1k"]) for r in sorted(versions, key=lambda r: r["version"])] == [("V1", 20, 1000.0), ("V2", 20, 1000.0)]
    books = scoring.aggregate(rows, counts, ("version", "book"))
    assert {(r["version"], r["book"]) for r in books} == {("V1", "Book"), ("V2", "Book")}
    examples = top.rows()
    assert {e["version"] for e in examples} == {"V1", "V2"} and all(len([e for e in examples if e["version"] == v]) == 5 for v in ("V1", "V2"))
    assert scoring.to_csv(versions).splitlines()[0].startswith("version,segments,words,flagged")


def test_bootstrap_ci_brackets_the_rate():
    from error_detection.steps.scoring import bootstrap_ci

    words = np.full(50, 100.0)
    flagged = np.r_[np.zeros(25), np.full(25, 4.0)]
    lo, hi = bootstrap_ci(words, flagged)
    assert lo < 20.0 < hi


def test_segment_label_scope_keeps_neighbours_as_unlabelled_context(tokenizer):
    tokens = ["אב", "גד", "הו", "זח", "טי", "כל"]          # segment is tokens 2..4; the rest are neighbouring segments
    row = make_row(tokens, [1, 0, 0, 1, 0, 1], [{"i1": 3, "i2": 4, "before": ["זח"], "after": ["זט"], "kind": "typo", "labelled": True}])
    row["segment_span"] = [2, 4]
    [all_chunk] = chunk_row(tokenizer, row, 32, "all")
    [seg_chunk] = chunk_row(tokenizer, row, 32, "segment")
    assert all_chunk["word_index"] == [0, 1, 2, 3, 4, 5] and all_chunk["word_in_segment"] == [False, False, True, True, False, False]
    assert seg_chunk["word_index"] == [2, 3]
    assert list(seg_chunk["input_ids"]) == list(all_chunk["input_ids"])     # same input, fewer labels
    assert [int(l) for l in seg_chunk["labels"] if l != IGNORE] == [0, 1]
    syn = dict(row, source="synthetic")
    assert chunk_row(tokenizer, syn, 32, "segment")[0]["word_index"] == [0, 1, 2, 3, 4, 5]   # synthetic labels are all kept
    with pytest.raises(ValueError):
        chunk_row(tokenizer, row, 32, "nope")


def test_corrected_moves_the_segment_end_with_fixes_inside_it():
    row = make_row(["אב", "גדהו", "זח", "טי"], [0, 1, 0, 0],
                   [{"i1": 1, "i2": 2, "before": ["גדהו"], "after": ["גד", "הו"], "kind": "spacing", "labelled": True}])
    row["segment_span"] = [1, 3]
    fixed = corrected(row)
    assert fixed["tokens"] == ["אב", "גד", "הו", "זח", "טי"] and fixed["segment_span"] == [1, 4]


def test_evaluation_reports_segment_words_separately():
    from error_detection.steps.evaluation import score_and_summarise

    words = [{"chunk": "a#0", "word_index": i, "label": l, "kind": "typo" if l else None, "version": "V", "score": s, "in_segment": seg}
             for i, (l, s, seg) in enumerate([(1, 0.9, True), (0, 0.1, True), (0, 0.8, False), (1, 0.7, False)])]
    rep = score_and_summarise(words, words)
    assert rep["test"]["words"] == 4 and rep["segment"]["test"]["words"] == 2
    assert rep["segment"]["test"]["pr_auc"] == pytest.approx(1.0)


def judged_row():
    # tokens: 0 A, 1 B(fixed), 2 C, 3 D(judged error), 4 E(judged undecided)
    return {"id": "j1", "source": "real", "ref": "Book 1:1", "version": "V", "tokens": ["אב", "גד", "הו", "זח", "טי"],
            "labels": [0, 1, 0, 0, 0], "segment_span": [0, 5], "judged": {"3": 1, "4": -1},
            "fixes": [{"i1": 1, "i2": 2, "before": ["גד"], "after": ["ג", "ד"], "kind": "spacing", "labelled": True}]}


def test_judged_labels_load_and_shift_into_corrected_copy(tmp_path, monkeypatch):
    path = tmp_path / "real" / "train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(judged_row(), ensure_ascii=False) + "\n")
    monkeypatch.setattr(data_module, "dataset_file", lambda repo, filename: str(tmp_path / filename))
    row = load_rows("real", "train", "gs://x/y")[0]
    assert row["labels"] == [0, 1, 0, 1, IGNORE]
    assert row["kinds"][3] == "judged"
    fixed = corrected(row)
    # the split fix adds a token, so the judged words move one to the right
    assert fixed["tokens"] == ["אב", "ג", "ד", "הו", "זח", "טי"]
    assert fixed["labels"] == [0, 0, 0, 0, 1, IGNORE]


def test_undecided_judged_word_is_not_scored(tokenizer):
    row = {**judged_row(), "labels": [0, 1, 0, 1, IGNORE], "kinds": [None, "spacing", None, "judged", None]}
    chunk = chunk_row(tokenizer, row, max_length=64)[0]
    assert chunk["word_index"] == [0, 1, 2, 3]
    assert sum(1 for label in chunk["labels"] if label != IGNORE) == 4


def test_folds_partition_rows_by_version_and_book():
    from error_detection.steps.data import fold_of, select_fold
    rows = [{"ref": f"Book{b} {c}:1", "version": f"V{v}"} for b in range(6) for c in range(1, 4) for v in range(3)]
    held = [select_fold(rows, 4, k, held_out=True) for k in range(4)]
    assert sum(len(h) for h in held) == len(rows)
    for k in range(4):
        assert len(select_fold(rows, 4, k, held_out=False)) + len(held[k]) == len(rows)
    # chapters of one book in one version always share a fold
    assert len({fold_of(r, 4) for r in rows if r["ref"].startswith("Book2 ") and r["version"] == "V1"}) == 1
    assert select_fold(rows, 0, 0, held_out=False) == rows
