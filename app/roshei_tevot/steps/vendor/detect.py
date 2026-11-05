# Vendored from Sefaria/translation-engine experiments/expand_roshei_tevot/detect.py @ b8160e4f; only imports were made relative. Re-sync rather than edit.
"""Rosh-tevot detector. Recall over precision: anything that looks like an abbreviation is emitted, including
numerals and one-letter geresh forms; classification and prefix parsing are left ambiguous on purpose.

Row per token: surface, normalised `abbr` (no-prefix reading), `parses` (every prefix split), `cls`, numeric info."""
import re
from .hebrew import PREFIX_LETTERS, clean, numeral_value

# letters (+ internal gershayim, optional trailing geresh).  Gershayim only count between letters.
TOKEN = re.compile(r"[א-ת]+(?:\"[א-ת]+)*'?")
HEB = re.compile(r"[א-ת]")

CITATIONS = {'ע"א', 'ע"ב', 'ב"ק', 'ב"מ', 'ב"ב', 'ר"ה', 'מ"ק', 'ע"ז', 'ב"ר', 'שמ"ר', 'ויק"ר', 'במ"ר', 'דב"ר',
             'שה"ש', 'קה"ר', 'איכ"ר', 'ד"ה', 'ע"ג', 'ת"ר', 'ס"ק', 'פ"ק', 'פ"ג'}
CITATION_RE = re.compile(r"^(?:פ|פי|ה)\"[א-ת]{1,2}$")   # פ"א, פי"ב (perek), ה"ג (halakha)
TETRAGRAMMATON = {"ה'", 'ה"', "ד'"}
GERESH_MAX_LETTERS = 7   # plan says 1-3, but prefixed truncations (דמתני', בירוש', בקונט') are common; recall over precision

def parses(core):
    """All (prefix, abbr) splits of `core` where abbr is itself shaped like an abbreviation."""
    out = []
    for n in range(0, 4):
        prefix, rest = core[:n], core[n:]
        if any(c not in PREFIX_LETTERS for c in prefix):
            break
        if '"' in rest:
            a, _, b = rest.partition('"')
            if not (a and b):                     # quote must sit between letters of the abbreviation itself
                continue
        elif rest.endswith("'"):
            if len(HEB.findall(rest)) < 1:
                continue
        else:
            continue
        out.append((prefix, rest))
    return out

def quote_artifact(tok, before):
    """Quotation marks glued to a word, not an abbreviation: ב"תחומין" (3+ letters after the gershayim) or 'אשתך' (geresh
    that closes a quote opened right before the token). Still emitted, with its own class, so precision can be measured."""
    if '"' in tok and len(tok.rsplit('"', 1)[1].rstrip("'")) >= 3:
        return True
    return tok.endswith("'") and before in ("'", '"')

def classify(abbr):
    if abbr in TETRAGRAMMATON:
        return "tetragrammaton"
    if abbr in CITATIONS or CITATION_RE.match(abbr):
        return "citation"
    letters = abbr.replace('"', "").replace("'", "")
    if abbr.endswith("'") and len(letters) == 1:
        return "one_letter_geresh"
    if numeral_value(abbr) is not None:
        return "gematria_only"        # upgraded to ambiguous_numeric once the lexicon knows it as an abbreviation
    return "abbr"

def is_candidate(tok):
    if '"' in tok:
        return True
    return tok.endswith("'") and 1 <= len(tok) - 1 <= GERESH_MAX_LETTERS

def detect(text):
    """Yield one dict per candidate token in `text` (already cleaned), in order, with the word index."""
    for wi, m in enumerate(TOKEN.finditer(text)):
        tok = m.group()
        if not is_candidate(tok):
            continue
        ps = parses(tok)
        if not ps:
            continue
        art = quote_artifact(tok, text[m.start() - 1] if m.start() else "")
        abbr = ps[0][1]                      # shortest-prefix reading
        row_cls = next((classify(a) for _, a in ps if classify(a) in ("citation", "tetragrammaton")), classify(abbr))
        num = numeral_value(abbr)
        yield {"surface": tok, "start": m.start(), "abbr": abbr, "cls": "quote_artifact" if art else row_cls,
               "numeric_value": num, "is_valid_gematria": num is not None,
               "parses": [{"prefix": p, "abbr": a, "cls": classify(a)} for p, a in ps]}

if __name__ == "__main__":
    for s in ['ור"ת כתב וכו\' והרמב"ם י"ד ט"ו ה\' ר\' אמר ע"ב בפ"א "בראשית" מהר"ם']:
        for r in detect(clean(s)):
            print(r["surface"], r["cls"], r["numeric_value"], [p["prefix"] + "|" + p["abbr"] for p in r["parses"]])
