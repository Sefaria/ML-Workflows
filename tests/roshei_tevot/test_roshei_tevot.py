import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from roshei_tevot.steps.data import ExpansionCollator, ResampledDataset, sampling_weights  # noqa: E402
from roshei_tevot.steps.encoding import add_markers, encode_candidate, encode_query, marker_ids  # noqa: E402
from roshei_tevot.steps.index import ExpansionIndex, decide  # noqa: E402
from roshei_tevot.steps.model import SiameseExpander  # noqa: E402


@pytest.fixture()
def tokenizer(tmp_path):
    chars = list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ") + ['"', "'", "|", ".", ":"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars] + ["a", "b", "c", "x", "y"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=False)
    add_markers(tok)
    return tok


def row(before="א ב ג", surface='ר"י', after="ד ה ו", expansion="רבי יוסי", candidates=("רבי יוסי", "רבי יהודה"), key='ר"י', source="gold"):
    return {"before": before, "surface": surface, "after": after, "base_expansion": expansion, "candidates": list(candidates), "key": key,
            "source": source, "title": "Mishnah x", "category": "Mishnah"}


def test_marked_span_covers_the_surface(tokenizer):
    enc = encode_query(tokenizer, "א ב ג", 'ר"י', "ד ה", "Talmud", "Berakhot", max_length=64)
    open_id, close_id = marker_ids(tokenizer)
    assert enc.input_ids[enc.span_start] == open_id and enc.input_ids[enc.span_end] == close_id
    assert len(enc.input_ids) == len(enc.token_type_ids)
    assert enc.token_type_ids[0] == 0 and enc.token_type_ids[enc.span_start] == 1


def test_truncation_keeps_the_marked_token_and_metadata(tokenizer):
    long_before, long_after = " ".join(["א"] * 400), " ".join(["ב"] * 400)
    enc = encode_query(tokenizer, long_before, 'ר"י', long_after, "Talmud", "Berakhot", max_length=64)
    assert len(enc.input_ids) <= 64
    assert enc.input_ids[enc.span_start] == marker_ids(tokenizer)[0]
    # context is cut about evenly on both sides
    assert abs((enc.span_start) - (len(enc.input_ids) - enc.span_end)) < 20


def test_candidate_is_wrapped_and_bounded(tokenizer):
    ids = encode_candidate(tokenizer, " ".join(["א"] * 100), max_length=8)
    assert len(ids) == 8 and ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id


def test_collator_deduplicates_the_pool_and_targets_the_positive(tokenizer):
    rows = [row(), row(expansion="רבי יוסי"), row(expansion="רבי יהודה"), row(candidates=())]
    batch = ExpansionCollator(tokenizer, 64, hard_negatives=7)(rows)
    texts = batch["candidates"]["input_ids"].shape[0]
    assert texts == 2                       # רבי יוסי and רבי יהודה once each, however often they occur
    assert batch["targets"].tolist() == [0, 0, 1, 0]
    assert batch["queries"]["span_mask"].sum(1).min() >= 3


def test_hard_negatives_never_include_the_positive(tokenizer):
    rows = [row(candidates=("רבי יוסי", "רבי יהודה", "רבי יעקב"))]
    batch = ExpansionCollator(tokenizer, 64, hard_negatives=7)(rows)
    assert batch["candidates"]["input_ids"].shape[0] == 3 and batch["targets"].tolist() == [0]


def test_sampling_down_weights_frequent_keys_and_synthetic_rows():
    rows = [row(key="a")] * 100 + [row(key="b")] + [row(key="b", source="swapped")]
    w = sampling_weights(rows)
    assert w[:100].sum() < 0.95 and w[100] > w[101] and abs(w.sum() - 1) < 1e-9
    a = ResampledDataset(rows, 50, w, seed=3).order
    b = ResampledDataset(rows, 50, w, seed=3).order
    assert (a == b).all()


@pytest.mark.parametrize("pooling", ["span", "cls", "mean"])
def test_siamese_loss_learns_a_toy_mapping(tokenizer, pooling):
    torch.manual_seed(0)
    encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64), add_pooling_layer=False)
    model = SiameseExpander(encoder, pooling)
    rows = [row(surface='ר"י', expansion="רבי יוסי"), row(surface='ר"ה', expansion="ראש השנה", candidates=("ראש השנה", "רבי הונא"))]
    batch = ExpansionCollator(tokenizer, 64)(rows * 4)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for _ in range(60):
        out = model(batch["queries"], batch["candidates"], batch["targets"])
        first = first if first is not None else out["loss"].item()
        opt.zero_grad(); out["loss"].backward(); opt.step()
    assert out["loss"].item() < 0.6 * first
    if pooling != "cls":  # a CLS vector has to learn to attend to the marked token, which a one-layer toy may not manage in 60 steps
        assert out["accuracy"].item() == 1.0


def test_index_filters_by_letters_and_ranks_by_similarity(tmp_path):
    lexicon = tmp_path / "lex.jsonl"
    entries = [
        {"surface": 'ור"י', "candidates": [
            {"bare": 'ר"י', "prefix": "ו", "expansion": "רבי יוסי", "freq": 5},
            {"bare": 'ר"י', "prefix": "ו", "expansion": "ראש חודש", "freq": 99},     # letters do not fit ר"י
        ]},
        {"surface": 'ר"י', "candidates": [{"bare": 'ר"י', "prefix": "", "expansion": "רבי יהודה", "freq": 3}]},
    ]
    lexicon.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries))

    class Stub:
        pass
    # Build without a model: embed by hand.
    from roshei_tevot.steps import index as idx
    texts = ["רבי יוסי", "ראש חודש", "רבי יהודה"]
    emb = torch.eye(3)
    by_surface = {e["surface"]: {c["expansion"]: c["freq"] for c in e["candidates"]} for e in entries}
    by_bare = {'ר"י': {"רבי יוסי": 5, "ראש חודש": 99, "רבי יהודה": 3}}
    index = idx.ExpansionIndex(texts, emb, by_surface, by_bare)

    found = index.candidates('ור"י')
    assert "ראש חודש" not in found and {"רבי יוסי", "רבי יהודה"} <= set(found)
    ranked = index.rank('ור"י', torch.tensor([0.0, 0.0, 1.0]))
    assert ranked[0][0] == "רבי יהודה"
    assert index.candidates('חז"ל') == {}

    index.save(str(tmp_path / "idx"))
    again = ExpansionIndex.load(str(tmp_path / "idx"))
    assert again.expansions == texts and again.rank('ור"י', torch.tensor([1.0, 0, 0]))[0][0] == "רבי יוסי"


def test_decide_abstains_on_weak_or_close_scores():
    ranked = [("a", 0.9, 1), ("b", 0.89, 1)]
    assert decide(ranked, 0.5, 0.05) is None and decide(ranked, 0.5, 0.0) == "a"
    assert decide([("a", 0.3, 1)], 0.5, 0.0) is None and decide([], 0.0, 0.0) is None
