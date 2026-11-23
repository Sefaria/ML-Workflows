import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from roshei_tevot.steps.encoding import add_markers  # noqa: E402
from roshei_tevot.steps.index import ExpansionIndex, Lexicon  # noqa: E402
from roshei_tevot.steps.predict import RosheiTevot  # noqa: E402
from roshei_tevot.publish_hf import build_senses  # noqa: E402


@pytest.fixture()
def tokenizer(tmp_path):
    chars = list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ") + ['"', "'", "|", ".", ":", ","]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=False)
    add_markers(tok)
    return tok


def make(tokenizer, senses=None, prior_weight=0.0):
    torch.manual_seed(0)
    model = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64), add_pooling_layer=False)
    lexicon = Lexicon(by_surface={'ר"ש': {"רבי יוסי": 30, "רבי יהודה": 10, "רבי יהודא": 5}}, by_bare={'ר"ש': {"רבי יוסי": 30, "רבי יהודה": 10, "רבי יהודא": 5}})
    expansions = list(lexicon.expansions())
    index = ExpansionIndex(lexicon, expansions, torch.nn.functional.normalize(torch.randn(len(expansions), 32), dim=-1))
    return RosheiTevot(model, tokenizer, {"logit_scale": 20.0, "pooling": "span"}, index, senses, prior_weight)


def test_find_detects_abbreviations_and_skips_letter_names(tokenizer):
    rt = make(tokenizer)
    _, tokens = rt.find('אמר ר"ש ומ"ם')
    surfaces = [t["surface"] for t in tokens]
    assert 'ר"ש' in surfaces
    assert 'מ"ם' not in surfaces                                  # a letter name, never an abbreviation


def test_token_offsets_point_at_the_surface(tokenizer):
    rt = make(tokenizer)
    text, tokens = rt.find('אמר ר"ש')
    for t in tokens:
        assert text[t["start"]:t["end"]] == t["surface"]


def test_scores_are_probabilities_over_the_readings_and_sorted(tokenizer):
    rt = make(tokenizer, prior_weight=0.5)
    out = rt.expand('אמר ר"ש אבגד')
    token = next(t for t in out["tokens"] if t["surface"] == 'ר"ש')
    assert token["n_readings"] == 4                        # the three lexicon readings and the citation reading `רמז ש'`
    assert sum(g["probability"] for g in token["senses"]) == pytest.approx(1.0, abs=1e-4)
    probs = [g["probability"] for g in token["senses"]]
    assert probs == sorted(probs, reverse=True)


def test_readings_of_one_sense_are_one_answer(tokenizer):
    senses = {'ר"ש': {"0": {"label": "Rabbi Yehuda", "texts": ["רבי יהודה", "רבי יהודא"], "expansions": ["רבי יהודה", "רבי יהודא"]},
                      "1": {"label": "Rabbi Yose", "texts": ["רבי יוסי"], "expansions": ["רבי יוסי"]}}}
    rt = make(tokenizer, senses)
    token = next(t for t in rt.expand('אמר ר"ש')["tokens"] if t["surface"] == 'ר"ש')
    assert len(token["senses"]) == 3                       # Rabbi Yose, Rabbi Yehuda (two spellings, one answer) and the citation reading `רמז ש'`
    yehuda = next(g for g in token["senses"] if g["sense"] == 0)
    assert yehuda["label"] == "Rabbi Yehuda" and {r["text"] for r in yehuda["readings"]} == {"רבי יהודה", "רבי יהודא"}


def test_the_frequency_prior_moves_probability_to_the_frequent_reading(tokenizer):
    flat = make(tokenizer, prior_weight=0.0).score([{"surface": 'ר"ש', "before": "א", "after": "ב"}])[0]
    prior = make(tokenizer, prior_weight=3.0).score([{"surface": 'ר"ש', "before": "א", "after": "ב"}])[0]
    share = lambda r: next(g["probability"] for g in r["senses"] if g["text"] == "רבי יוסי")       # noqa: E731  (freq 30, the most frequent)
    assert share(prior) > share(flat)


def test_a_bare_numeral_gets_no_citation_readings(tokenizer):
    rt = make(tokenizer)
    assert 'סק"ג' not in rt.candidates('קי"ב') and all(not t.startswith("קיצור") for t in rt.candidates('קי"ב'))
    assert rt.candidates("סי'") == {} or all("י'" not in t for t in rt.candidates("סי'"))      # a truncated word is not unit + numeral


def test_a_token_with_nothing_to_offer_has_no_senses(tokenizer):
    rt = make(tokenizer)
    assert rt.score([{"surface": 'בלה"בלה', "before": "א", "after": "ב"}])[0] == {"senses": [], "n_readings": 0}


def test_build_senses_collects_classes_from_train_rows():
    rows = ['{"key": "ר\\"ש", "candidates": ["רבי יוסי", "רבי יהודה"], "candidate_senses": [1, 0], "sense": 0, "sense_label": "Rabbi Yehuda", "sense_texts": ["רבי יהודה"]}',
            '{"key": "ר\\"ש", "candidates": ["רבי יוסי"], "candidate_senses": [1], "sense": null}', '{"candidates": []}']
    senses = build_senses(rows)
    assert senses['ר"ש']["0"] == {"label": "Rabbi Yehuda", "texts": ["רבי יהודה"], "expansions": ["רבי יהודה"]}
    assert senses['ר"ש']["1"]["expansions"] == ["רבי יוסי"]
