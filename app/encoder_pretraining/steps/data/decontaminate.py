import re
from typing import Iterable


WHITESPACE = re.compile(r"\s+")


def _words(text: str) -> list[str]:
    return WHITESPACE.sub(" ", text.lower()).split()


class NgramDecontaminator:
    """Flags documents that share any run of n consecutive words with a set of held-out texts.

    Used to keep an evaluation set out of the pretraining corpus, so a downstream probe
    measures generalization and not memorization of its own text.
    """

    def __init__(self, held_out_texts: Iterable[str], n: int = 10):
        self.n = n
        self._index: set[int] = set()
        for text in held_out_texts:
            words = _words(text)
            for start in range(len(words) - n + 1):
                self._index.add(hash(tuple(words[start : start + n])))

    def __len__(self) -> int:
        return len(self._index)

    def contaminated(self, text: str) -> bool:
        words = _words(text)
        index = self._index
        n = self.n
        return any(hash(tuple(words[start : start + n])) in index for start in range(len(words) - n + 1))
