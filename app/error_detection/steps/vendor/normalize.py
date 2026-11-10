# Vendored from Sefaria/translation-engine experiments/version_quality_from_history/classify.py at commit db47231 (TAG, NIQQUD).
# Re-sync from there rather than editing here: the dataset was built with these patterns.
import re

# Escaped tags first: search-highlight spans leaked into some texts as &lt;span ...&gt;.
TAG = re.compile(r"&lt;.*?&gt;|<[^>]*>|&nbsp;|&[a-z]+;")
# Hebrew points and cantillation (U+0591-U+05C7), excluding maqaf (U+05BE) and sof pasuq (U+05C3), which are punctuation.
NIQQUD = re.compile("[֑-ׇֽֿׁׂׅׄ]")
MAQAF = "־"


def normalize_word(word: str) -> str:
    """What the model sees for one dataset token: tags and niqqud removed, maqaf split into a space (the tokenizer keeps the pieces in
    the same word). Punctuation and quote marks stay, since geresh and gershayim are evidence for abbreviation errors."""
    return TAG.sub("", NIQQUD.sub("", word)).replace(MAQAF, " ").strip()
