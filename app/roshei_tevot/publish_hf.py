"""Assemble a self-contained release of a roshei tevot model and (optionally) upload it to Hugging Face. Run locally: the Prefect pods have no HF token.

    cd app
    python -m roshei_tevot.publish_hf --run rt-numbers-v5b-siamese-2026-11-23 --index rt-numbers-v5b-2026-11-23 \\
        --dataset gs://sefaria-ml-models/roshei-tevot/data/berel-roshei-tevot-numbers-v5b --card /path/to/model_card.md \\
        --out /tmp/rt_release                       # build + check only
    python -m roshei_tevot.publish_hf ... --upload Sefaria/berel-roshei-tevot-siamese --message "..."     # also upload (uses your `hf auth login`)

Needs `gcloud` logged in (the model, the index and the dataset's train.jsonl are read from gs://sefaria-ml-models/roshei-tevot/) and, to upload, `huggingface_hub`.
The release holds, so that the model, its lexicon and the code that makes candidates can never be paired wrongly:
    <model files>                    config.json, model.safetensors, tokenizer, roshei_tevot_config.json   (runs/<run>/final)
    index/lexicon.json, embeddings.npy   the lexicon and one embedding per reading                          (index/<index>)
    senses.json                      sense classes, from the dataset's train rows                           (build_senses)
    roshei_tevot/                    the inference code: predict.py, index.py, encoding.py, nonabbrev.py, citations.py, vendor/   (copied from steps/)
    README.md                        the model card (--card)
Use it with `sys.path.insert(0, folder)`; `from roshei_tevot.predict import RosheiTevot; RosheiTevot.from_pretrained(folder)`.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BUCKET = "gs://sefaria-ml-models/roshei-tevot"
STEPS = Path(__file__).parent / "steps"
CODE = ["predict.py", "index.py", "encoding.py", "nonabbrev.py", "citations.py"]      # data.py, model.py, training.py, evaluation.py stay: they need the training stack


def build_senses(lines) -> dict:
    """Stream the training rows once and collect the lexicon's sense classes: key -> sense index -> {label, texts, expansions}.
    A row carries `candidates` + `candidate_senses` (a sense index per candidate) and, for its label, `sense`, `sense_label` (English gloss) and `sense_texts`."""
    senses = defaultdict(lambda: defaultdict(lambda: {"label": "", "texts": set(), "expansions": set()}))
    for line in lines:
        row = json.loads(line)
        key = row.get("key")
        if not key:
            continue
        for exp, s in zip(row.get("candidates") or [], row.get("candidate_senses") or []):
            if s is not None:
                senses[key][s]["expansions"].add(exp)
        s = row.get("sense")
        if s is not None:
            entry = senses[key][s]
            entry["label"] = entry["label"] or row.get("sense_label") or ""
            entry["texts"].update(row.get("sense_texts") or [])
    return {k: {str(i): {"label": e["label"], "texts": sorted(e["texts"]), "expansions": sorted(e["expansions"])} for i, e in v.items()} for k, v in senses.items()}


def gcloud(*args: str) -> None:
    subprocess.run(["gcloud", "storage", *args], check=True)


def assemble(run: str, index: str, dataset: str, card: Path, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    gcloud("cp", "-r", f"{BUCKET}/runs/{run}/final/*", str(out))
    (out / "_SUCCESS").unlink(missing_ok=True)
    (out / "index").mkdir()
    gcloud("cp", f"{BUCKET}/index/{index}/embeddings.npy", f"{BUCKET}/index/{index}/lexicon.json", str(out / "index"))
    print("collecting sense classes from the dataset's train rows ...")
    cat = subprocess.Popen(["gcloud", "storage", "cat", f"{dataset}/train.jsonl"], stdout=subprocess.PIPE, text=True, encoding="utf-8")
    senses = build_senses(cat.stdout)
    if cat.wait():
        raise RuntimeError("could not read train.jsonl from the dataset")
    (out / "senses.json").write_text(json.dumps(senses, ensure_ascii=False), encoding="utf-8")
    code = out / "roshei_tevot"
    (code / "vendor").mkdir(parents=True)
    (code / "__init__.py").write_text("")
    for name in CODE:
        shutil.copyfile(STEPS / name, code / name)
    for f in (STEPS / "vendor").iterdir():
        if f.is_file():
            shutil.copyfile(f, code / "vendor" / f.name)
    shutil.copyfile(card, out / "README.md")
    return out


def verify(folder: Path) -> None:
    """Load the release as a user would, from a clean import path, and expand a sentence."""
    script = f'''
import sys
sys.path.insert(0, {str(folder)!r})
from roshei_tevot.predict import RosheiTevot          # the copy in the release (this runs isolated, with only the folder on the path)
rt = RosheiTevot.from_pretrained({str(folder)!r})
result = rt.expand('אמר ר"ש בן לקיש בפ"ב דשבת מ"ט, ובסק"ג כתב הרמ"א וע"ל סי\\' קכ"ה.')
assert result["tokens"], "no abbreviation found in the check sentence"
for t in result["tokens"]:
    if t["senses"]:
        total = sum(g["probability"] for g in t["senses"])
        assert abs(total - 1) < 1e-3, (t["surface"], total)
    print(" ", t["surface"], "->", t["senses"][0]["text"] if t["senses"] else "(no reading offered)")
'''
    subprocess.run([sys.executable, "-I", "-c", script], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name under gs://sefaria-ml-models/roshei-tevot/runs/ (its final/ is the model)")
    p.add_argument("--index", required=True, help="index name under .../index/ (embeddings.npy + lexicon.json built from that model)")
    p.add_argument("--dataset", required=True, help="gs:// folder of the dataset the model was trained on (its train.jsonl gives the sense classes)")
    p.add_argument("--card", required=True, type=Path, help="the model card, copied as README.md")
    p.add_argument("--out", required=True, type=Path, help="folder to assemble the release in (replaced)")
    p.add_argument("--upload", metavar="REPO", help="after checking, upload the folder to this Hugging Face model repo")
    p.add_argument("--message", default="", help="commit message for the upload")
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    out = assemble(a.run, a.index, a.dataset, a.card, a.out)
    if not a.no_verify:
        verify(out)
    print("release folder:", out)
    if a.upload:
        from huggingface_hub import HfApi

        if not a.message:
            raise SystemExit("--message is required with --upload")
        info = HfApi().upload_folder(folder_path=str(out), repo_id=a.upload, repo_type="model", commit_message=a.message)
        print("uploaded:", info)


if __name__ == "__main__":
    main()
