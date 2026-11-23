"""Tokens written with gershayim that are never abbreviations: Hebrew letter names (אל"ף, נו"ן, מ"ם ...) and divine names (הוי"ה, אדנ"י, אלהי"ם).

The comparison run showed the model reading these at p~1.0 ("Yoreh Deah", "לסימן ח'") and Dicta guessing too. Left out on purpose, because they are also real abbreviations:
יו"ד (יורה דעה), למ"ד (למאן דאמר), רי"ש (רבי ישמעאל), דל"ת (דלא תעשה), כת"ר (כבוד תורתו), שד"י, ה"א, ו"ו, פ"א, ת"ו. Matching ignores quote marks and up to two attached prefix letters.
"""
import re

NAMES = {"אלף", "בית", "גימל", "זין", "חית", "טית", "כף", "מם", "נון", "סמך", "עין", "צדי", "קוף", "שין", "תיו", "אלהים", "אדני", "הויה", "אהיה"}
MARK = re.compile(r"[\"'״׳]")
PREFIX = "ובהכלמש"


def is_non_abbreviation(surface: str) -> bool:
    if '"' not in surface and "״" not in surface:
        return False
    bare = MARK.sub("", surface)
    for n in range(0, 3):
        if n and (len(bare) <= n + 2 or bare[n - 1] not in PREFIX):
            break
        if bare[n:] in NAMES:
            return True
    return False
