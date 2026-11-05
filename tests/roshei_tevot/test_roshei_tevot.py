import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from roshei_tevot.steps.data import CrossCollator, ResampledDataset, SiameseCollator, attach_candidates, sampling_weights  # noqa: E402
from roshei_tevot.steps.encoding import add_markers, encode_candidate, encode_query, marker_ids  # noqa: E402
from roshei_tevot.steps.evaluation import evaluate_splits, outcomes, summarise  # noqa: E402
from roshei_tevot.steps.index import ExpansionIndex, Lexicon, decide  # noqa: E402
from roshei_tevot.steps.model import CrossExpander, SiameseExpander  # noqa: E402
from roshei_tevot.steps.vendor.spelling import same_expansion, stem_key  # noqa: E402


@pytest.fixture()
def tokenizer(tmp_path):
    chars = list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ") + ['"', "'", "|", ".", ":"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars] + ["a", "b", "c", "x", "y"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=False)
    add_markers(tok)
    return tok


def tiny_encoder(tokenizer):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    return BertModel(config, add_pooling_layer=False)


def row(surface='ר"י', target="רבי יוסי", candidates=("רבי יוסי", "רבי יהודה"), key='ר"י', source="gold", before="א ב ג", after="ד ה ו"):
    return {"before": before, "surface": surface, "after": after, "base_expansion": target, "target": target, "candidates": list(candidates),
            "freqs": [10 - i for i in range(len(candidates))], "key": key, "source": source, "title": "משנה", "category": "משנה"}


LEXICON = Lexicon(
    by_surface={'ור"י': {"רבי יוסי": 5, "ראש חודש": 99}, 'ר"י': {"רבי יהודה": 30}},
    by_bare={'ר"י': {"רבי יוסי": 5, "רבי יהודה": 30, "ראש חודש": 99}},
)


# ---- encoding ----
def test_marked_span_covers_the_surface(tokenizer):
    enc = encode_query(tokenizer, "א ב ג", 'ר"י', "ד ה", "תלמוד", "ברכות", max_length=64)
    open_id, close_id = marker_ids(tokenizer)
    assert enc.input_ids[enc.span_start] == open_id and enc.input_ids[enc.span_end] == close_id
    assert enc.token_type_ids[0] == 0 and enc.token_type_ids[enc.span_start] == 1


def test_truncation_keeps_the_marked_token_and_cuts_both_sides(tokenizer):
    enc = encode_query(tokenizer, " ".join(["א"] * 400), 'ר"י', " ".join(["ב"] * 400), "תלמוד", "ברכות", max_length=64, context_words=400)
    assert len(enc.input_ids) <= 64 and enc.input_ids[enc.span_start] == marker_ids(tokenizer)[0]
    assert abs(enc.span_start - (len(enc.input_ids) - enc.span_end)) < 20


def test_cross_input_puts_the_candidate_first_and_keeps_the_span(tokenizer):
    plain = encode_query(tokenizer, "א ב", 'ר"י', "ג", "תלמוד", "ברכות", max_length=64)
    cross = encode_query(tokenizer, "א ב", 'ר"י', "ג", "תלמוד", "ברכות", max_length=64, candidate="רבי יוסי")
    assert cross.input_ids[1] == tokenizer.convert_tokens_to_ids("ר")
    assert cross.input_ids[cross.span_start] == marker_ids(tokenizer)[0] and cross.span_start > plain.span_start


def test_candidate_is_wrapped_and_bounded(tokenizer):
    ids = encode_candidate(tokenizer, " ".join(["א"] * 100), max_length=8)
    assert len(ids) == 8 and ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id


# ---- spelling folds ----
@pytest.mark.parametrize("a,b", [("ודרבי שמעון", "רבי שמעון"), ("מגזרה שוה", "גזירה שוה"), ("רבי יעקב", "רבי יעקב")])
def test_folds_merge_prefix_and_plene_variants(a, b):
    assert same_expansion(a, b)


@pytest.mark.parametrize("a,b", [("רבי יוסי", "רבי יהודה"), ("רבא", "רבי"), ("רבי יוסה", "רבי יוסי")])
def test_folds_keep_different_words_apart(a, b):
    assert not same_expansion(a, b)


# ---- lexicon lookup and candidate mining ----
def test_lookup_uses_prefix_readings_and_drops_letters_that_do_not_fit():
    found = LEXICON.candidates('ור"י')
    assert set(found) == {"רבי יוסי", "רבי יהודה"} and found["רבי יהודה"] == 30


def test_attach_candidates_matches_folded_gold_and_drops_single_candidate_rows():
    rows = [
        {**row(surface='ור"י'), "base_expansion": "ורבי יוסי"},     # gold carries the prefix; the fold maps it to the lexicon spelling
        {**row(surface='חז"ל'), "base_expansion": "חכמינו זכרונם לברכה"},  # nothing in the lexicon
    ]
    kept, report = attach_candidates(rows, LEXICON, min_candidates=2, training=False)
    assert len(kept) == 1 and kept[0]["target"] == "רבי יוסי" and kept[0]["candidates"][0] == "רבי יהודה"
    assert report["dropped_below_min_candidates"] == 1


def test_training_keeps_an_unmatched_gold_as_an_extra_candidate():
    rows = [{**row(surface='ור"י'), "base_expansion": "רבי יוסה"}]
    train, _ = attach_candidates(rows, LEXICON, training=True)
    evaluation, report = attach_candidates(rows, LEXICON, training=False)
    assert train[0]["target"] == "רבי יוסה" and "רבי יוסה" in train[0]["candidates"]
    assert evaluation[0]["target"] is None and report["gold_not_in_lexicon_lookup"] == 1


# ---- collators ----
def test_siamese_pool_is_deduplicated_and_marks_each_rows_candidates(tokenizer):
    rows = [row(), row(), row(target="רבי יהודה"), row(candidates=("רבי יוסי", "רבי יעקב"))]
    batch = SiameseCollator(tokenizer, 64)(rows)
    assert batch["candidates"]["input_ids"].shape[0] == 3        # יוסי, יהודה, יעקב once each
    assert batch["targets"].tolist() == [0, 0, 1, 0]
    assert batch["own"][0].tolist() == [True, True, False] and batch["own"][3].tolist() == [True, False, True]


def test_negatives_never_include_a_spelling_of_the_target(tokenizer):
    batch = SiameseCollator(tokenizer, 64)([row(candidates=("רבי יוסי", "ורבי יוסי", "רבי יהודה"))])
    assert batch["candidates"]["input_ids"].shape[0] == 2


def test_cross_collator_groups_target_first(tokenizer):
    batch = CrossCollator(tokenizer, 64)([row(), row(candidates=("רבי יוסי", "רבי יהודה", "רבי יעקב"))])
    assert batch["group"].tolist() == [0, 0, 1, 1, 1] and batch["rows"] == 2


def test_sampling_down_weights_frequent_keys_and_synthetic_rows():
    rows = [row(key="a")] * 100 + [row(key="b")] + [row(key="b", source="swapped")]
    w = sampling_weights(rows)
    assert w[:100].sum() < 0.95 and w[100] > w[101] and abs(w.sum() - 1) < 1e-9
    assert (ResampledDataset(rows, 50, w, seed=3).order == ResampledDataset(rows, 50, w, seed=3).order).all()


# ---- models ----
def _train(model, batch, steps=80):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for _ in range(steps):
        out = model(**batch)
        first = first if first is not None else out["loss"].item()
        opt.zero_grad(); out["loss"].backward(); opt.step()
    return first, out


TOY = [row(surface='ר"י', target="רבי יוסי", candidates=("רבי יוסי", "רבי הונא")), row(surface='ר"ה', target="ראש השנה", candidates=("ראש השנה", "רבי הונא"))]


@pytest.mark.parametrize("pooling", ["span", "mean"])
def test_siamese_learns_a_toy_mapping(tokenizer, pooling):
    first, out = _train(SiameseExpander(tiny_encoder(tokenizer), pooling), SiameseCollator(tokenizer, 64)(TOY * 4))
    assert out["loss"].item() < 0.6 * first and out["accuracy"].item() == 1.0


def test_cross_encoder_learns_a_toy_mapping(tokenizer):
    first, out = _train(CrossExpander(tiny_encoder(tokenizer)), CrossCollator(tokenizer, 64)(TOY * 4))
    assert out["loss"].item() < 0.6 * first and out["accuracy"].item() == 1.0


def test_cross_loss_ignores_padding_in_uneven_groups(tokenizer):
    model = CrossExpander(tiny_encoder(tokenizer)).eval()
    batch = CrossCollator(tokenizer, 64, hard_negatives=0)([row(), row()])   # one sequence per row: loss is exactly 0
    assert model(**batch)["loss"].item() == pytest.approx(0.0, abs=1e-6)


# ---- evaluation ----
def test_prior_reorders_close_calls():
    rows = [row(target="רבי יהודה", candidates=("רבי יוסי", "רבי יהודה"))]
    rows[0]["freqs"] = [1, 1000]
    scores = [[1.0, 0.9]]
    assert outcomes(rows, scores, 0.0)[0]["rank"] == 1 and outcomes(rows, scores, 1.0)[0]["rank"] == 0


def test_unanswerable_rows_count_against_end_to_end_only():
    s = summarise([{"rank": 0, "frequency_correct": True, "confidence": 1.0}, {"rank": None, "frequency_correct": False, "confidence": 0.0}])
    assert s["top1"] == 1.0 and s["end_to_end_top1"] == 0.5 and s["answerable_fraction"] == 0.5


def test_evaluate_splits_picks_lambda_on_val_and_checks_success(tokenizer):
    model = SiameseExpander(tiny_encoder(tokenizer)).eval()
    rows = {"val": TOY, "test_seen_keys": TOY}
    report = evaluate_splits(model, tokenizer, rows, "cpu", 20.0, (0.0, 1.0), max_length=64, context_words=10)
    assert report["prior_selected_on"] == "val" and report["prior_weight"] in (0.0, 1.0)
    assert set(report["success"]["checks"]) == {"val", "test_seen_keys"} and isinstance(report["success"]["passes"], bool)


def test_index_ranks_with_prior_and_round_trips(tmp_path):
    index = ExpansionIndex(LEXICON, ["רבי יוסי", "ראש חודש", "רבי יהודה"], torch.eye(3))
    assert index.rank('ור"י', torch.tensor([1.0, 0, 0]))[0][0] == "רבי יוסי"
    assert index.rank('ור"י', torch.tensor([1.0, 0, 0]), prior_weight=1.0)[0][0] == "רבי יהודה"
    index.save(str(tmp_path / "idx"))
    again = ExpansionIndex.load(str(tmp_path / "idx"))
    assert again.rank('ור"י', torch.tensor([0, 0, 1.0]))[0][0] == "רבי יהודה"


def test_decide_abstains_on_weak_or_close_scores():
    ranked = [("a", 0.9, 1), ("b", 0.89, 1)]
    assert decide(ranked, 0.5, 0.05) is None and decide(ranked, 0.5, 0.0) == "a"
    assert decide([("a", 0.3, 1)], 0.5, 0.0) is None and decide([], 0.0, 0.0) is None


def test_stem_key_is_a_dedup_key():
    assert stem_key("ורבי יוסי") == stem_key("רבי יוסי")


def test_any_spelling_of_the_target_counts_as_correct():
    rows = [row(target="רבי יוסי", candidates=("רבי יוסי", "ורבי יוסי", "רבי יהודה"))]
    assert outcomes(rows, [[0.1, 0.9, 0.2]], 0.0)[0]["rank"] == 0


@pytest.mark.parametrize("architecture", ["siamese", "cross"])
def test_trainer_checkpoint_resumes(tokenizer, tmp_path, architecture):
    """Spot evictions depend on this: a checkpoint written mid-run must load back into a fresh model and continue."""
    from transformers import TrainingArguments
    from roshei_tevot.steps.training import ExpansionTrainer

    def make():
        model = SiameseExpander(tiny_encoder(tokenizer)) if architecture == "siamese" else CrossExpander(tiny_encoder(tokenizer))
        collator = SiameseCollator(tokenizer, 64) if architecture == "siamese" else CrossCollator(tokenizer, 64)
        args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=2, max_steps=4, save_strategy="steps", save_steps=2,
                                 report_to="none", remove_unused_columns=False, use_cpu=True, disable_tqdm=True, logging_steps=1)
        return ExpansionTrainer(model=model, args=args, train_dataset=ResampledDataset(TOY * 4, 8), data_collator=collator, processing_class=tokenizer)

    first = make()
    first.train()
    resumed = make()
    resumed.args.max_steps = 6
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-4"))
    assert resumed.state.global_step == 6   # loading the checkpoint raises on any missing or mismatched weight
