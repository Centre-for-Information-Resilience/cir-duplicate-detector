"""
Code adapted from: https://github.com/AiLECS/pyMIH

Licensed under the MIT License:
Copyright 2018 Artificial Intelligence for Law Enforcement and Community Safety (AiLECS) Lab, a joint Australian Federal Police and Monash Univeristy initiative

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""  # noqa

import re
from math import floor

from bitarray import bitarray


class MIHIndex:
    """
    Implementation of MIH algorithm (see https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf
        (refer brief doc at https://github.com/facebook/ThreatExchange/blob/master/hashing/hashing.pdf)
    """

    def __init__(self, hashsize: int = 256):
        """
        :param hashsize: Hash size in bits (not: NOT string length!)
        """
        if hashsize % 8 != 0:
            raise ValueError("hashsize " + str(hashsize) + " is not a multiple of 8")
        self._hashlength = hashsize
        self._items = None
        self._index = None
        self._categories: list[str] = []
        self._trained = False
        self._pattern = re.compile("^[a-f0-9]{" + str(int(hashsize / 4)) + "}$", re.IGNORECASE)
        self._floor = None
        self._words = None
        self._wordlength = None
        self._threshold = None
        self._cache = None

    def _getwindow(
        self, word: bitarray, distance: int = 2, _position: int = 0, _entries: None | set = None
    ) -> set[str]:
        """
        Calculate and return a set of all hex combinations within the provided hamming distance

        :param word: candidate word as bitarray. If string provided, converts to bitarray internally.
        :param distance: Maximum hamming distance (inclusive). Defaults to 2
        :param _position: Position to commence comparison. For internal (recursive) use only.
        :param _entries: Set of possible entries. For internal (recursive) use only.
        :return: set of hex strings within provided hamming distance of candidate word.
        """
        if _entries is None:
            _entries = set()

        if _position == len(word):
            _entries.add(word.tobytes().hex())
            return _entries
        if distance > 0:
            temp = word[_position]
            for i in [True, False]:
                word[_position] = i
                distOffset = 0
                if temp != word[_position]:
                    distOffset = -1
                _entries.update(self._getwindow(word, distance + distOffset, _position + 1, _entries))
            word[_position] = temp
        else:
            _entries.update(self._getwindow(word, distance, _position + 1, _entries))
        return _entries

    def update(self, new: str | list[str], category: str) -> None:
        """
        Add more entries for indexing, together with category/class. Only accepts iterables of hex strings

        Keyword arguments:
        new -- iterable of hex strings to add
        category -- category/class name for the added strings (e.g. 'ignorable')
        """
        offset = -1
        for i in range(0, len(self._categories), 1):
            if self._categories[i] == category:
                offset = i
                break
        if offset == -1:
            self._categories.append(category)
            offset = len(self._categories) - 1

        if isinstance(new, str):
            new = [new]

        if self._items is None:
            self._items = {}

        for x in new:
            if self._pattern.fullmatch(x):
                if x not in self._items:
                    self._items[x] = [offset]
                else:
                    if offset not in self._items[x]:
                        self._items[x].append(offset)
            else:
                raise ValueError("Not a valid hex string: " + str(x))
            self._trained = False

    # train the index.
    def train(self, wordLength: int = 16, threshold: int = 32) -> None:
        """
        Train and initialise the index. (i.e. make it queryable)
        :param wordLength: Word length (in bits). Defaults to 16
        :param threshold:  Threshold for matching - i.e. hamming distance needs to be <= to guarantee entry
            to be returned. Defaults to 32
        :return: None
        """
        if self._trained:
            raise ValueError("Index already trained. Rebuild object to train again")

        self._index = []
        for k, v in self._items.items():
            b = bitarray()
            b.frombytes(bytes.fromhex(k))
            if len(b) != self._hashlength:
                raise ValueError("Invalid hash length encountered: " + k)
            else:
                self._index.append((b, v))
        self._items = None

        self._floor = floor(threshold / wordLength)

        self._words = []
        for i in range(0, self._hashlength, wordLength):
            w = {}
            for c in range(0, len(self._index)):
                hex = self._index[c][0][i : i + wordLength].tobytes().hex()
                if hex not in w:
                    w[hex] = {c}
                else:
                    w[hex].add(c)
            for k in w:
                w[k] = frozenset(w[k])
            self._words.append(w)

        self._trained = True
        self._wordlength = wordLength
        self._threshold = threshold

    # Check hamming distance - iterate through list. Returns None if distance exceeded
    @staticmethod
    def gethamming(hash1: str, hash2: str, maxhd: int | None = None) -> int | None:
        """
        Calculate hamming distance between hashes
        :param hash1: Candidate hash
        :param hash2: Comparison hash
        :param maxhd: Maximum hamming distance. If none, defaults to length of hash1
        :return: Hamming distance if <= maxhd, else None
        """
        # TODO: make this use fuzywuzzy or similar for speed
        if maxhd is None:
            maxhd = len(hash1)
        hd = 0
        for b1, b2 in zip(hash1, hash2, strict=True):
            if b1 != b2:
                hd += 1
                if hd > maxhd:
                    return None
        return hd

    def query(self, h: str):
        """
        Query index for candidates within pre-set hamming distance threshold (set at train())
        :param h: Candidate hash (hex string)
        :return: Generator returning matches as tuples (<hex>, <hamming distance>) format
        """
        if not self._trained:
            raise ValueError("Index not trained yet")
        b = bitarray()
        b.frombytes(bytes.fromhex(h))
        if len(b) != self._hashlength:
            raise ValueError("Invalid hash length encountered: " + h)

        candidates = set()
        slot = 0
        for i, slot in enumerate(range(0, self._hashlength, self._wordlength)):
            window = self._getwindow(b[i : i + self._wordlength])
            for w in window:
                if w in self._words[slot]:
                    for x in self._words[slot][w]:
                        if x not in candidates:
                            r = self.gethamming(b, self._index[x][0], self._threshold)
                            if r is not None:
                                cats = []
                                for d in self._index[x][1]:
                                    cats.append(self._categories[d])
                                yield self._index[x][0].tobytes().hex(), cats, r
                            candidates.add(x)
