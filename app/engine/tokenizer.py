"""A small, dependency-free tokenizer.

Word-and-punctuation level with a numeric split. Not byte-pair encoding, and
it makes no attempt to pretend otherwise — but it is deterministic, reversible
enough for display, and fast enough to tokenize a prompt on every request.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Order matters: numbers, then word runs, then single punctuation characters.
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:[''][A-Za-z]+)*|[^\sA-Za-z\d]")

NO_SPACE_BEFORE = frozenset(".,!?;:%)]}'\"…—")
NO_SPACE_AFTER = frozenset("([{$#'\"—")


class Tokenizer:
    """Maps text to integer ids and back.

    Ids are stable for a given corpus: the vocabulary is sorted so the same
    corpus always produces the same id for the same token. That is what makes a
    saved seed able to reproduce a run.
    """

    def __init__(self, corpus_tokens: list[str], specials: Sequence[str] = ()) -> None:
        vocab = sorted(set(corpus_tokens))
        self.itos: list[str] = vocab
        self.stoi: dict[str, int] = {token: index for index, token in enumerate(vocab)}

        # Specials are appended after the corpus vocabulary so every real token
        # keeps its index, then the id attributes are resolved *last*.
        for special in specials:
            self.add_special(special)

        self._refresh_specials()

    def add_special(self, token: str) -> int:
        """Register a special token and return its id."""
        if token not in self.stoi:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1
        return self.stoi[token]

    def _refresh_specials(self) -> None:
        """Resolve the special ids from the mapping.

        This must run after every mutation. Getting it wrong is subtle and
        expensive: if ``eos_id`` is left pointing at index 0, the first comma
        the sampler produces looks like end-of-sequence and generation stops
        after a single token.
        """
        self.unk_id: int = self.stoi.get("<unk>", 0)
        self.bos_id: int = self.stoi.get("<bos>", self.unk_id)
        self.eos_id: int = self.stoi.get("<eos>", self.unk_id)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def split(self, text: str) -> list[str]:
        """Tokenize without mapping to ids — useful for building a vocabulary."""
        return _TOKEN_RE.findall(text)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in self.split(text)]

    def decode(self, ids: list[int]) -> str:
        """Detokenize with basic spacing rules.

        Reconstructing exact original whitespace is impossible once text has
        been split, so this applies the conventional English rule: no space
        before closing punctuation, no space after opening punctuation.
        """
        pieces: list[str] = []
        for _index, token_id in enumerate(ids):
            token = self.itos[token_id] if 0 <= token_id < len(self.itos) else "<unk>"
            if token in {"<bos>", "<eos>", "<unk>"}:
                continue
            if not pieces:
                pieces.append(token)
                continue
            previous = pieces[-1]
            if token in NO_SPACE_BEFORE or previous in NO_SPACE_AFTER:
                pieces[-1] = previous + token
            else:
                pieces.append(token)
        return " ".join(pieces).strip()

    def render_token(self, token_id: int) -> str:
        """Single token as it should appear when appended during streaming."""
        if 0 <= token_id < len(self.itos):
            return self.itos[token_id]
        return ""
